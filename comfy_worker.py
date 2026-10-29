"""Local SQuadGen inference with quad-preserving mesh export."""
import argparse
import gc
import json
import shutil
import subprocess
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pymeshlab
import point_cloud_utils as pcu
import torch
import trimesh
from omegaconf import OmegaConf
from plyfile import PlyData
from safetensors.torch import load_file, save_file
from torch.nn.attention import SDPBackend, sdpa_kernel

from squadgen.network.models_sit import create_sqdiffuse_from_config, StackedRandomGenerator
from squadgen.network.models_vae_joint import create_joint_vae_from_config
from squadgen.network.transport import create_transport, Sampler
from squadgen.util.util import (create_batch_from_data, sample_points_on_mesh,
                               spatial_smoothing, de_norm_pca)
from audit_quads import audit
from quad_refine import refine_to_target, write_quad_ply
from cdf_cleanup import smooth_fields


def log(message):
    print(message, flush=True)


def extract_quads(binary, surface, feature_path, out, subdivisions, preset='standard', max_faces=0):
    attempts = []
    best = None
    reference = trimesh.load(surface, process=False)
    reference_points, _ = trimesh.sample.sample_surface(reference, 2048, seed=0)
    diagonal = np.linalg.norm(reference.extents)
    feature_variants = [('raw', feature_path)]
    if preset == 'artist':
        with np.load(feature_path) as archive:
            fields = {key: np.array(archive[key]) for key in archive.files}
        cleaned = smooth_fields(reference, fields)
        cleaned_path = out/'fields_cleaned.npz'
        np.savez(cleaned_path, **cleaned)
        feature_variants.append(('cleaned', cleaned_path))
    del reference
    presets = {'standard': [(0, 8, 0.1)], 'divided': [(1, 4, 0.2)], 'detail': [(1, 8, 0.1)],
               'artist': [(0, 8, 0.1), (0, 4, 0.05), (0, 12, 0.15)],
               'auto': [(0, 8, 0.1), (1, 4, 0.2), (1, 2, 0.1)]}
    configurations = [(variant, feature, division, rings, threshold)
                      for variant, feature in feature_variants
                      for division, rings, threshold in presets[preset]]
    for variant, feature, division, rings, threshold in configurations:
        suffix = f'_{variant}' if preset == 'artist' else ''
        path = out / f'layout_div{division}_rings{rings}{suffix}.ply'
        command = [str(binary), '-i', str(surface), '-f', str(feature), '-o', str(path),
                   f'--ringsize={rings}', '--verbose=true', '-a', '150', f'--div={division}',
                   '-t', str(threshold), '-s', str(subdivisions), '-r', '15', '--maxfaces', str(max_faces)]
        if preset == 'artist':
            command.append('--chartfit=true')
        process = subprocess.run(command)
        attempt = {'division': division, 'rings': rings, 'threshold': threshold, 'fields': variant,
                   'returncode': process.returncode}
        attempts.append(attempt)
        if process.returncode != 0 or not path.is_file():
            log(f'Extraction preset failed: {attempt}')
            continue
        ply = PlyData.read(str(path))
        faces = ply['face']['vertex_indices']
        if not len(faces):
            continue
        fraction = sum(len(f) == 4 for f in faces) / len(faces)
        attempt.update(faces=len(faces), quad_fraction=fraction)
        quality_tool = binary.parent / 'QuadQuality.exe'
        subprocess.run([str(quality_tool), '-i', str(path), '-j', str(out)], check=True)
        quality = json.loads(path.with_name(path.stem + '_quadquality.json').read_text())
        total = sum(row['NumOfFaces'] for row in quality)
        score = sum(min(row['Fratio'], row['Eratio']) * row['NumOfFaces'] for row in quality) / total if total else 0.0
        health = audit(path)
        vertices = np.column_stack([ply['vertex'][key] for key in ('x','y','z')]).astype(np.float64)
        triangles = np.array([[int(f[0]),int(f[i]),int(f[i+1])] for f in faces for i in range(1,len(f)-1)], dtype=np.int32)
        distances, _, _ = pcu.closest_points_on_mesh(reference_points, vertices, triangles)
        shape_error = float(distances.mean()/diagonal*100)
        attempt.update(loop_score_face_weighted=score, quad_quality=health,
                       surface_error_mean_percent_diagonal=shape_error, surface_error_samples=2048)
        if preset == 'artist':
            rank = (score * fraction * (1-health.get('fraction_folded_quads',0)) *
                    (1-health.get('fraction_edge_ratio_above_5',0)) * np.exp(-shape_error), -shape_error)
        else:
            rank = (fraction, -shape_error, score)
        if best is None or rank > best[0]:
            best = (rank, path, attempt)
    if best is None:
        raise RuntimeError('QuadExtraction failed for every preset; see run.log.')
    final = out / 'extracted_quad.ply'
    shutil.copyfile(best[1], final)
    selection = ('loop simplicity penalized by folded/slender/non-quad faces and surface error'
                 if preset == 'artist' else 'quad fraction, then sampled surface fidelity, then loop simplicity')
    return final, {'preset': preset, 'max_faces': max_faces, 'selected': best[2], 'attempts': attempts,
                   'selection': selection}


def run(args):
    started = time.perf_counter()
    torch.set_grad_enabled(False)
    if not torch.cuda.is_available():
        raise RuntimeError('SQuadGen requires an NVIDIA CUDA GPU for this workflow.')
    torch.set_num_threads(8)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parent
    binary = root / 'QuadTools' / 'build' / 'Release' / 'QuadExtraction.exe'
    source = Path(args.input).resolve()
    input_ply = out / 'input.ply'
    samples = out / 'sampled_points.npz'
    surface = out / 'surface.ply'
    if args.prepared_dir:
        prepared = Path(args.prepared_dir).resolve()
        previous = json.loads((prepared / 'report.json').read_text(encoding='utf-8-sig'))
        if Path(previous['input']).resolve() != source:
            raise ValueError('Prepared geometry belongs to a different input mesh.')
        for key in ('res', 'preprocess_faces', 'edge_percent', 'remesh_iterations'):
            if previous['settings'].get(key) != getattr(args, key):
                raise ValueError(f'Prepared geometry uses a different {key}.')
        if previous['settings'].get('conditioning','original') != args.conditioning:
            raise ValueError('Prepared geometry uses a different layout-conditioning method.')
        original_faces, original_vertices = previous['input_triangles'], previous['input_vertices']
        for name in ('input.ply', 'sampled_points.npz', 'surface.ply'):
            shutil.copyfile(prepared / name, out / name)
        log(f'Reusing prepared geometry: {prepared}')
    else:
        log(f'Loading mesh: {source}')
        mesh = trimesh.load(source, force='mesh', process=False)
        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True)
        mesh.remove_unreferenced_vertices()
        original_faces = len(mesh.faces)
        original_vertices = len(mesh.vertices)
        mesh.export(input_ply)
        log(f'Input: {original_vertices:,} vertices, {original_faces:,} triangles')
        condition_ply = input_ply
        if args.conditioning == 'clean_surface':
            blender = Path(args.blender_path).expanduser().resolve()
            if not blender.is_file():
                raise FileNotFoundError(f'Blender is required to prepare the layout surface: {blender}')
            proxy_input = out/'conditioning_input.npz'
            np.savez(proxy_input,vertices=mesh.vertices.astype(np.float32),faces=mesh.faces.astype(np.int32))
            del mesh
            gc.collect()
            condition_ply = out/'conditioning.ply'
            log('Preparing a cleaned surface for layout conditioning; original mesh stays in input.ply')
            subprocess.run([str(blender),'--background','--factory-startup','--python-exit-code','1',
                            '--python',str(root/'prepare_layout_proxy.py'),'--','--input',str(proxy_input),
                            '--output',str(condition_ply)],check=True)
            mesh = trimesh.load(condition_ply,force='mesh',process=True)
        log('Sampling surface and farthest points')
        sample_points_on_mesh(str(condition_ply), str(samples), n_fps_list=[args.res, 4 * args.res])
        log('Preparing extraction surface')
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(vertex_matrix=mesh.vertices, face_matrix=mesh.faces))
        if args.preprocess_faces and len(mesh.faces) > args.preprocess_faces:
            ms.meshing_decimation_quadric_edge_collapse(targetfacenum=args.preprocess_faces,
                                                       preservenormal=True, preservetopology=True)
        ms.meshing_remove_duplicate_vertices()
        ms.meshing_remove_duplicate_faces()
        ms.meshing_remove_unreferenced_vertices()
        remesh_options = {'targetlen': pymeshlab.PercentageValue(args.edge_percent), 'featuredeg': 10.0}
        if args.remesh_iterations is not None:
            remesh_options['iterations'] = args.remesh_iterations
        ms.meshing_isotropic_explicit_remeshing(**remesh_options)
        ms.save_current_mesh(str(surface), save_vertex_normal=False)
        del mesh, ms
        gc.collect()

    with np.load(samples) as data:
        batch = create_batch_from_data(dict(data), args.res)
    batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    inv = batch['invT'][0]
    working = trimesh.load(surface, process=False)
    centers = working.vertices[working.faces].mean(axis=1)
    queries = ((centers - inv[:, 3]) @ np.linalg.inv(inv[:, :3]).T).astype(np.float32)
    log(f'Extraction surface: {len(working.faces):,} triangles')
    del centers
    gc.collect()

    config = OmegaConf.load(root / 'squadgen/network/config/sqvae_geomae.yaml')
    config.model.params.latent_vae_params.num_latents = args.res
    config.model.params.cond_vae_params.num_latents = args.res
    log('Loading autoencoder')
    with torch.device('meta'):
        ae = create_joint_vae_from_config(config.model.params)
    ae.load_state_dict(load_file(str(Path(args.ckpts) / 'sqvae_geomae.safetensors')), assign=True)
    ae.eval().cuda()
    ae.latent_vae.out_keys_list = [k for k in ae.latent_vae.out_keys_list if not k.endswith('_edge')]
    model_config = OmegaConf.load(root / 'squadgen/network/config/sqdiffuse.yaml')
    log('Loading SQ-Diffuse')
    with torch.device('meta'):
        model = create_sqdiffuse_from_config(model_config.model.params, config.model.params)
    model.load_state_dict(load_file(str(Path(args.ckpts) / 'sqdiffuse.safetensors')), assign=True)
    model.eval()
    gc.collect()
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION), torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.precision == 'bf16'):
        log('Encoding high-poly geometry')
        condition = model.get_latent_and_condition(batch, ae, is_mode=True, is_run_sqvae=False)['condition_params']
        ae.cpu()
        torch.cuda.empty_cache()
        model.cuda()
        rng = StackedRandomGenerator('cuda', [args.seed])
        latents = rng.randn([1, model.n_latents, model.channels], device='cuda')
        if args.smoothing:
            pts = torch.cat([batch[f'xyz_fps_{args.res}'], 0.1 * batch[f'normal_fps_{args.res}']], dim=-1)
            latents = spatial_smoothing(pts, latents, k=32, ring_size=2)
        log(f'Generating layout: {args.res} latents, solver={args.solver}')
        sampler = Sampler(create_transport('Linear', 'velocity', None))
        sample = sampler.sample_ode(sampling_method=args.solver, num_steps=args.steps)
        evaluations = 0
        def predict(x, t, **kwargs):
            nonlocal evaluations
            evaluations += 1
            if evaluations % 10 == 0:
                log(f'Diffusion evaluation {evaluations}, t={t[0].item():.3f}')
            return model(x, t, **kwargs)
        latent = model.denormalize_latents(sample(latents, predict, condition_params=condition)[-1]).float()
        save_file({'latent': latent.cpu()}, str(out / 'latent.safetensors'))
        del model, condition, batch, latents
        gc.collect()
        torch.cuda.empty_cache()
        ae.latent_vae.cuda()
        log('Decoding chart distance fields')
        features = ae.latent_vae.return_last_features(latent)
        pieces = []
        for start in range(0, len(queries), args.query_chunk):
            points = torch.from_numpy(queries[start:start + args.query_chunk]).cuda()[None]
            predicted = ae.latent_vae.query_last_features(features, points)
            pieces.append({k: v[0].float().cpu().numpy() for k, v in predicted.items()})
            if start % (args.query_chunk * 20) == 0:
                log(f'Decoded {min(start + args.query_chunk, len(queries)):,}/{len(queries):,}')
        fields = {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    fields = de_norm_pca(fields, inv)
    fields['invT'] = inv
    if not all(np.isfinite(v).all() for v in fields.values()):
        raise RuntimeError('The model generated non-finite chart fields.')
    feature_path = out / 'extract_mesh.npz'
    np.savez(feature_path, **fields)
    del ae, features, latent
    gc.collect()
    torch.cuda.empty_cache()

    log('Extracting quad topology with upstream QuadTools')
    quad_ply, extraction = extract_quads(binary, surface, feature_path, out, args.subdivisions, args.extraction_preset, args.max_faces)
    ply = PlyData.read(str(quad_ply))
    vertices = np.column_stack([ply['vertex'][k] for k in ('x', 'y', 'z')])
    faces = ply['face']['vertex_indices']
    if not len(faces) or not np.isfinite(vertices).all():
        raise RuntimeError('QuadExtraction produced empty or invalid geometry.')
    base_quality = audit(quad_ply)
    density_refinement = {'target': args.target_quads, 'applied': False}
    if args.target_quads > len(faces) and all(len(face) == 4 for face in faces):
        shutil.copyfile(quad_ply, out / 'layout_before_density.ply')
        log(f'Refining output toward {args.target_quads:,} quads on the original high-poly surface')
        original_surface = trimesh.load(source, force='mesh', process=False)
        vertices, faces, density_refinement = refine_to_target(
            vertices, np.asarray(list(faces)), original_surface.vertices, original_surface.faces,
            args.target_quads, args.max_faces)
        write_quad_ply(quad_ply, vertices, faces)
        del original_surface
        gc.collect()
    sizes, counts = np.unique([len(f) for f in faces], return_counts=True)
    histogram = {str(k): int(v) for k, v in zip(sizes, counts)}
    edges = np.array([sorted((int(f[i]), int(f[(i + 1) % len(f)])))
                      for f in faces for i in range(len(f))])
    _, edge_counts = np.unique(edges, axis=0, return_counts=True)
    obj = out / 'extracted_quad.obj'
    with obj.open('w', encoding='utf-8') as stream:
        stream.write('# SQuadGen quad mesh; original input coordinates\n')
        for v in vertices:
            stream.write(f'v {v[0]:.9g} {v[1]:.9g} {v[2]:.9g}\n')
        for f in faces:
            stream.write('f ' + ' '.join(str(int(i) + 1) for i in f) + '\n')
    report = {
        'input': str(source), 'input_vertices': original_vertices, 'input_triangles': original_faces,
        'surface_triangles': len(working.faces), 'output_vertices': len(vertices),
        'output_faces': len(faces), 'face_sizes': histogram,
        'quad_fraction': histogram.get('4', 0) / len(faces),
        'boundary_edges': int((edge_counts == 1).sum()),
        'nonmanifold_edges': int((edge_counts > 2).sum()),
        'repeated_vertex_faces': sum(len(set(f)) != len(f) for f in faces),
        'triangle_equivalent_reduction': original_faces / sum((int(k) - 2) * v for k, v in histogram.items()),
        'elapsed_seconds': time.perf_counter() - started,
        'settings': vars(args), 'obj': str(obj), 'ply': str(quad_ply),
        'extraction': extraction,
        'quad_quality': audit(quad_ply),
        'loop_score_face_weighted': extraction['selected']['loop_score_face_weighted'],
        'base_quad_quality': base_quality,
        'density_refinement': density_refinement,
    }
    source_mesh = trimesh.load(source,force='mesh',process=False)
    validation_points,_ = trimesh.sample.sample_surface(source_mesh,6000,seed=97)
    triangles = np.array([[int(f[0]),int(f[i]),int(f[i+1])] for f in faces for i in range(1,len(f)-1)],dtype=np.int32)
    distances,_,_ = pcu.closest_points_on_mesh(validation_points,vertices.astype(np.float64),triangles)
    diagonal = np.linalg.norm(source_mesh.extents)
    report['surface_error_mean_percent_diagonal'] = float(distances.mean()/diagonal*100)
    report['surface_error_p99_percent_diagonal'] = float(np.percentile(distances,99)/diagonal*100)
    report['surface_error_reference'] = 'original input, independent area-uniform samples'
    quality = root / 'QuadTools/build/Release/QuadQuality.exe'
    if quality.is_file():
        subprocess.run([str(quality), '-i', str(quad_ply), '-j', str(out)], check=True)
        rows = json.loads((out / 'extracted_quad_quadquality.json').read_text())
        count = sum(row['NumOfFaces'] for row in rows)
        report['loop_score_face_weighted'] = sum(min(row['Fratio'], row['Eratio'])*row['NumOfFaces'] for row in rows)/count if count else 0
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    log(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--ckpts', required=True)
    parser.add_argument('--res', type=int, default=8192)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--solver', choices=['dopri5', 'euler', 'heun2'], default='dopri5')
    parser.add_argument('--smoothing', type=int, default=1)
    parser.add_argument('--query_chunk', type=int, default=2048)
    parser.add_argument('--preprocess_faces', type=int, default=0)
    parser.add_argument('--edge_percent', type=float, default=0.2)
    parser.add_argument('--subdivisions', type=int, default=3)
    parser.add_argument('--precision', choices=['fp32', 'bf16'], default='fp32')
    parser.add_argument('--remesh_iterations', type=int, default=None)
    parser.add_argument('--prepared_dir', default=None)
    parser.add_argument('--conditioning',choices=['original','clean_surface'],default='original')
    parser.add_argument('--blender_path',default='')
    parser.add_argument('--extraction_preset', choices=['detail', 'standard', 'divided', 'auto', 'artist'], default='standard')
    parser.add_argument('--max_faces', type=int, default=0)
    parser.add_argument('--target_quads', type=int, default=0)
    run(parser.parse_args())
