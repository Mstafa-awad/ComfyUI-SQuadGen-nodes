import argparse
from pathlib import Path
import sys

import bpy
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
data = np.load(args.input)
v, f = data['vertices'], data['faces']
mesh = bpy.data.meshes.new('Layout conditioning source')
mesh.vertices.add(len(v))
mesh.vertices.foreach_set('co',v.ravel())
mesh.loops.add(f.size)
mesh.loops.foreach_set('vertex_index',f.ravel())
mesh.polygons.add(len(f))
mesh.polygons.foreach_set('loop_start',np.arange(len(f),dtype=np.int32)*3)
mesh.polygons.foreach_set('loop_total',np.full(len(f),3,dtype=np.int32))
mesh.update(calc_edges=True)
obj = bpy.data.objects.new('Layout conditioning source', mesh)
bpy.context.collection.objects.link(obj)
bpy.context.view_layer.objects.active = obj
obj.select_set(True)
mesh.remesh_voxel_size = float(np.linalg.norm(np.ptp(v,axis=0))*.005)
mesh.remesh_voxel_adaptivity = 0
print('Closing small scan defects for layout conditioning; voxel size',mesh.remesh_voxel_size,flush=True)
bpy.ops.object.voxel_remesh()
relax = obj.modifiers.new('Conditioning surface relaxation','SMOOTH')
relax.factor = .5
relax.iterations = 3
bpy.ops.object.modifier_apply(modifier=relax.name)
mesh = obj.data
mesh.calc_loop_triangles()
positions = np.empty(len(mesh.vertices)*3,dtype=np.float32)
mesh.vertices.foreach_get('co',positions)
triangles = np.empty(len(mesh.loop_triangles)*3,dtype=np.int32)
mesh.loop_triangles.foreach_get('vertices',triangles)
triangles = triangles.reshape(-1,3)
path = Path(args.output)
path.parent.mkdir(parents=True,exist_ok=True)
with path.open('wb') as stream:
    stream.write((f'ply\nformat binary_little_endian 1.0\nelement vertex {len(mesh.vertices)}\nproperty float x\nproperty float y\nproperty float z\nelement face {len(triangles)}\nproperty list uchar int vertex_indices\nend_header\n').encode())
    stream.write(positions.astype('<f4').tobytes())
    records = np.empty(len(triangles),dtype=[('n','u1'),('v','<i4',(3,))])
    records['n'] = 3
    records['v'] = triangles
    stream.write(records.tobytes())
print('Saved layout-only proxy:',len(mesh.vertices),'vertices',len(triangles),'triangles',flush=True)
