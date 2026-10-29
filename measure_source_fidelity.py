"""Independent source-to-quad distance and geometric-normal validation."""
import argparse
import json
from pathlib import Path

import numpy as np
import point_cloud_utils as pcu
import trimesh


parser=argparse.ArgumentParser()
parser.add_argument('source')
parser.add_argument('quads')
parser.add_argument('--seed',type=int,default=107)
parser.add_argument('--save-samples',action='store_true')
parser.add_argument('--reference-normals',choices=['vertex','face'],default='vertex')
args=parser.parse_args()
source=np.load(args.source);target=np.load(args.quads)
sv,sf=source['vertices'].astype(float),source['faces'].astype(np.int32)
v,q=target['vertices'].astype(float),target['faces'].astype(np.int32)
mesh=trimesh.Trimesh(vertices=sv,faces=sf,process=False)
sn=source['normals'] if 'normals' in source else np.array(mesh.vertex_normals)
points,ids=trimesh.sample.sample_surface(mesh,60000,seed=args.seed)
bary=trimesh.triangles.points_to_barycentric(sv[sf[ids]],points)
normals=np.array(mesh.face_normals[ids]) if args.reference_normals=='face' else (sn[sf[ids]]*bary[:,:,None]).sum(1)
normals/=np.linalg.norm(normals,axis=1,keepdims=True)
triangles=np.concatenate((q[:,[0,1,2]],q[:,[0,2,3]])).astype(np.int32)
distance,ids,_=pcu.closest_points_on_mesh(points,v,triangles)
output_normals=np.cross(v[triangles[:,1]]-v[triangles[:,0]],v[triangles[:,2]]-v[triangles[:,0]])
output_normals/=np.maximum(np.linalg.norm(output_normals,axis=1,keepdims=True),1e-30)
angles=np.degrees(np.arccos(np.clip((output_normals[ids]*normals).sum(1),-1,1)))
target_mesh=trimesh.Trimesh(vertices=v,faces=triangles,process=False)
output_points,output_ids=trimesh.sample.sample_surface(target_mesh,60000,seed=args.seed+1000)
output_distance,source_ids,output_bary=pcu.closest_points_on_mesh(output_points,sv,sf)
nearest_normals=np.array(mesh.face_normals[source_ids]) if args.reference_normals=='face' else (sn[sf[source_ids]]*output_bary[:,:,None]).sum(1)
nearest_normals/=np.maximum(np.linalg.norm(nearest_normals,axis=1,keepdims=True),1e-30)
output_angles=np.degrees(np.arccos(np.clip((output_normals[output_ids]*nearest_normals).sum(1),-1,1)))
diagonal=np.linalg.norm(np.ptp(sv,axis=0))
report={'quads':len(q),'samples':len(points),'seed':args.seed,'reference_normal_mode':args.reference_normals,
        'source_distance_mean':float(distance.mean()),'source_distance_p95':float(np.percentile(distance,95)),
        'source_distance_p99':float(np.percentile(distance,99)),'source_distance_max_sample':float(distance.max()),
        'source_distance_p99_percent_diagonal':float(np.percentile(distance,99)/diagonal*100),
        'normal_angle_median_degrees':float(np.median(angles)),
        'normal_angle_p95_degrees':float(np.percentile(angles,95)),
        'normal_angle_p99_degrees':float(np.percentile(angles,99)),'fraction_opposed_normals':float((angles>90).mean()),
        'output_distance_mean':float(output_distance.mean()),'output_distance_p95':float(np.percentile(output_distance,95)),
        'output_distance_p99':float(np.percentile(output_distance,99)),'output_distance_max_sample':float(output_distance.max()),
        'output_distance_p99_percent_diagonal':float(np.percentile(output_distance,99)/diagonal*100),
        'output_normal_angle_p95_degrees':float(np.percentile(output_angles,95)),
        'output_fraction_opposed_normals':float((output_angles>90).mean())}
Path(args.quads).with_name('independent_fidelity.json').write_text(json.dumps(report,indent=2))
if args.save_samples:
    np.savez(Path(args.quads).with_name('fidelity_samples.npz'),source_points=points,output_points=output_points,
             source_distance=distance,source_angles=angles,output_distance=output_distance,output_angles=output_angles)
print(json.dumps(report,indent=2),flush=True)
