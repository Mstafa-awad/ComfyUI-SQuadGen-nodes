import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import folder_paths
import comfy.model_management
import comfy.utils


def run_worker(command, root, out):
    logfile = out / 'run.log'
    with logfile.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        try:
            while process.poll() is None:
                comfy.model_management.throw_exception_if_processing_interrupted()
                time.sleep(0.5)
        finally:
            if process.poll() is None:
                if os.name == 'nt':
                    subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                                   capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
                else:
                    process.terminate()
                process.wait()
    report_path = out / 'report.json'
    if process.returncode != 0 or not report_path.is_file():
        tail = logfile.read_text(encoding='utf-8', errors='replace')[-4000:]
        raise RuntimeError(f'Mesh worker failed. Full log: {logfile}\n{tail}')
    return json.loads(report_path.read_text(encoding='utf-8'))


class SQuadGenRemesh:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'mesh_path': ('STRING', {'default': '', 'tooltip': 'Absolute path to a GLB, OBJ, PLY or STL mesh.'}),
            'output_prefix': ('STRING', {'default': 'SQuadGen'}),
            'resolution': ('INT', {'default': 8192, 'min': 256, 'max': 16384, 'step': 256,
                                   'tooltip': 'Model latents, not target face count. The paper uses 8192 for inference.'}),
            'seed': ('INT', {'default': 0, 'min': 0, 'max': 0xffffffffffffffff}),
            'solver': (['dopri5', 'heun2', 'euler'],),
            'steps': ('INT', {'default': 50, 'min': 2, 'max': 1000,
                              'tooltip': 'Integration points; dopri5 chooses its internal steps adaptively.'}),
            'latent_smoothing': ('BOOLEAN', {'default': True}),
            'preprocess_faces': ('INT', {'default': 0, 'min': 0, 'max': 10000000, 'step': 1000,
                                         'tooltip': 'Optional triangle budget before extraction remeshing. 0 disables it. Clean-surface conditioning uses a repaired proxy instead of the original geometry.'}),
            'edge_percent': ('FLOAT', {'default': 0.2, 'min': 0.05, 'max': 5.0, 'step': 0.05,
                                       'tooltip': 'Extraction surface edge length as percent of bounding box diagonal. Upstream uses 0.2.'}),
            'subdivisions': ('INT', {'default': 3, 'min': 0, 'max': 7,
                                     'tooltip': 'QuadTools output subdivisions. Higher values produce more quads.'}),
            'query_chunk': ('INT', {'default': 2048, 'min': 128, 'max': 16384, 'step': 128}),
        }, 'optional': {
            'precision': (['fp32', 'bf16'], {'default': 'fp32', 'tooltip': 'FP32 matches upstream default. BF16 can alter generated layouts.'}),
            'generations': ('INT', {'default': 4, 'min': 1, 'max': 16,
                                     'tooltip': 'The paper generates four layouts and selects by loop simplicity. More generations take longer.'}),
            'extraction_preset': (['standard', 'divided', 'detail', 'auto', 'artist'], {'default': 'standard',
                                   'tooltip': 'Artist is an experimental search for simple loops with noise cleanup and defect penalties. Standard matches upstream; no preset guarantees artist-ready topology.'}),
            'face_budget': ('INT', {'default': 0, 'min': 0, 'max': 2147483647, 'step': 1000,
                                    'tooltip': '0 disables the face budget. Artist layout uses no count target or budget.'}),
            'target_quads': ('INT', {'default': 0, 'min': 0, 'max': 2147483647, 'step': 1000,
                                     'tooltip': 'Approximate output quad count. Added vertices are projected to the original high-poly surface. 0 keeps the native layout.'}),
            'conditioning': (['original','clean_surface'], {'default':'original',
                              'tooltip':'Clean surface closes small scan defects before learning the layout. It changes the conditioning surface, not your input file; intended for topology-focused experiments.'}),
            'blender_path': ('STRING', {'default':'',
                                      'tooltip':'Used only for clean-surface layout conditioning.'}),
        }}

    RETURN_TYPES = ('STRING', 'STRING', 'STRING')
    RETURN_NAMES = ('quad_obj_path', 'quad_ply_path', 'report')
    FUNCTION = 'remesh'
    CATEGORY = '3D/SQuadGen'
    OUTPUT_NODE = True
    DESCRIPTION = 'Generate a learned quad layout with Microsoft SQuadGen and QuadTools. Writes OBJ/PLY with loop scores, face ratios and geometric quality diagnostics.'

    def remesh(self, mesh_path, output_prefix, resolution, seed, solver, steps,
               latent_smoothing, preprocess_faces, edge_percent, subdivisions, query_chunk,
               precision='fp32', generations=4, extraction_preset='standard', face_budget=0, target_quads=0,
               conditioning='original',blender_path=''):
        source = Path(mesh_path.strip().strip('"')).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f'Mesh does not exist: {source}')
        if source.suffix.lower() not in {'.glb', '.gltf', '.obj', '.ply', '.stl', '.off'}:
            raise ValueError('Use a GLB, GLTF, OBJ, PLY, STL or OFF mesh.')
        if solver not in {'dopri5', 'heun2', 'euler'}:
            raise ValueError(f'Unsupported solver: {solver}')
        if precision not in {'fp32', 'bf16'} or extraction_preset not in {'detail', 'standard', 'divided', 'auto', 'artist'}:
            raise ValueError('Unsupported precision or extraction preset.')
        if conditioning not in {'original','clean_surface'}:
            raise ValueError('Unsupported layout conditioning.')
        if conditioning == 'clean_surface' and not Path(blender_path).expanduser().is_file():
            raise FileNotFoundError('Set blender_path to the Blender executable for clean-surface conditioning.')
        if not re.fullmatch(r'[A-Za-z0-9_-]+', output_prefix):
            raise ValueError('Output prefix must contain only letters, numbers, underscores or hyphens.')
        root = Path(__file__).resolve().parent
        ckpts = root / 'ckpts'
        for name in ('sqvae_geomae.safetensors', 'sqdiffuse.safetensors'):
            if not (ckpts / name).is_file():
                raise FileNotFoundError(f'Place {name} in {ckpts}')
        if not (root / 'QuadTools/build/Release/QuadExtraction.exe').is_file():
            raise FileNotFoundError('QuadExtraction.exe is missing. See this node pack README.')
        destination = Path(folder_paths.get_output_directory()) / output_prefix
        destination.mkdir(parents=True, exist_ok=True)
        out = destination / f'{source.stem}_{time.time_ns()}'
        out.mkdir()
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        candidates = []
        progress = comfy.utils.ProgressBar(generations)
        for index in range(generations):
            gen_out = out / f'gen_{index:03d}'
            gen_out.mkdir()
            command = [sys.executable, '-u', str(root / 'comfy_worker.py'),
                       '--input', str(source), '--output', str(gen_out), '--ckpts', str(ckpts),
                       '--res', str(resolution), '--seed', str(seed + index), '--solver', solver,
                       '--steps', str(steps), '--smoothing', str(int(latent_smoothing)),
                       '--preprocess_faces', str(preprocess_faces), '--edge_percent', str(edge_percent),
                       '--subdivisions', str(subdivisions), '--query_chunk', str(query_chunk),
                       '--precision', precision, '--extraction_preset', extraction_preset,
                       '--max_faces', str(face_budget), '--target_quads', str(target_quads)]
            command.extend(['--conditioning',conditioning,'--blender_path',blender_path])
            if index:
                command.extend(['--prepared_dir', str(out / 'gen_000')])
            logfile = gen_out / 'run.log'
            print(f'[SQuadGen] Generation {index + 1}/{generations}: {logfile}', flush=True)
            candidates.append(run_worker(command, root, gen_out))
            progress.update(1)
        def selection_score(item):
            if extraction_preset != 'artist':
                return (item['quad_fraction'], item['loop_score_face_weighted'])
            health = item['quad_quality']
            return (item['loop_score_face_weighted'] * item['quad_fraction'] *
                    (1-health.get('fraction_folded_quads',0)) *
                    (1-health.get('fraction_edge_ratio_above_5',0)) *
                    math.exp(-item.get('surface_error_mean_percent_diagonal',0)),)
        report = max(candidates, key=selection_score)
        report['generation_candidates'] = [
            {'seed': item['settings']['seed'], 'loop_score_face_weighted': item['loop_score_face_weighted'],
             'quad_fraction': item['quad_fraction'], 'quad_quality': item['quad_quality'],
             'obj': item['obj']} for item in candidates]
        report['selection'] = ('loop simplicity penalized by folded/slender/non-quad faces and original-surface error'
                               if extraction_preset == 'artist' else
                               'quad fraction, then face-weighted loop simplicity; geometric defects are reported separately')
        summary = json.dumps(report, indent=2)
        (out / 'report.json').write_text(summary, encoding='utf-8')
        folds = report['quad_quality'].get('fraction_folded_quads')
        fold_text = f'{folds:.1%}' if folds is not None else 'N/A'
        return {'ui': {'text': [f"Loop simplicity: {report['loop_score_face_weighted']:.1%}; {report['output_faces']:,} faces; {report['quad_fraction']:.1%} quads; folded quads: {fold_text}\n{report['obj']}"]},
                'result': (report['obj'], report['ply'], summary)}

NODE_CLASS_MAPPINGS = {'SQuadGenRemesh': SQuadGenRemesh}
NODE_DISPLAY_NAME_MAPPINGS = {'SQuadGenRemesh': 'SQuadGen • Experimental Quad Layout'}
