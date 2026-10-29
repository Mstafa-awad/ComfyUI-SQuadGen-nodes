import argparse

import numpy as np
import trimesh


parser = argparse.ArgumentParser()
parser.add_argument('source')
parser.add_argument('output')
args = parser.parse_args()
mesh = trimesh.load(args.source, force='mesh', process=False)
np.savez(args.output, vertices=mesh.vertices.astype(np.float32),
         faces=mesh.faces.astype(np.int32), normals=mesh.vertex_normals.astype(np.float32))
print(f'Saved {len(mesh.vertices)} vertices / {len(mesh.faces)} triangles', flush=True)
