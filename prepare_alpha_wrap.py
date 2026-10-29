"""Optional destructive-to-detail preprocessing, written to a new mesh file."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pymeshlab
import trimesh

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('input')
parser.add_argument('output', help='New PLY/OBJ/GLB file; existing files are refused')
parser.add_argument('--alpha-percent', type=float, default=.5)
parser.add_argument('--offset-percent', type=float, default=.05)
args = parser.parse_args()
source, output = Path(args.input).expanduser().resolve(), Path(args.output).expanduser().resolve()
if output.exists(): parser.error('Choose a new output path; no existing mesh is overwritten')
if not source.is_file(): parser.error('Input mesh does not exist')
mesh = trimesh.load(source, force='mesh', process=False)
ms = pymeshlab.MeshSet(); ms.add_mesh(pymeshlab.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces)))
started = time.perf_counter()
ms.generate_alpha_wrap(alpha=pymeshlab.PercentageValue(args.alpha_percent), offset=pymeshlab.PercentageValue(args.offset_percent))
result = ms.current_mesh(); mesh = trimesh.Trimesh(result.vertex_matrix(), result.face_matrix(), process=True)
counts = np.bincount(mesh.edges_unique_inverse)
output.parent.mkdir(parents=True, exist_ok=True); mesh.export(output)
report = {'input': source.name, 'output': output.name, 'alpha_percent_diagonal': args.alpha_percent,
          'offset_percent_diagonal': args.offset_percent, 'vertices': len(mesh.vertices), 'triangles': len(mesh.faces),
          'watertight': bool(mesh.is_watertight), 'boundary_edges': int((counts == 1).sum()),
          'nonmanifold_edges': int((counts > 2).sum()), 'elapsed_seconds': time.perf_counter()-started,
          'warning': 'Wrapping changes the surface, can round features and remove small details; it is not a lossless repair.'}
output.with_suffix(output.suffix+'.preparation.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
