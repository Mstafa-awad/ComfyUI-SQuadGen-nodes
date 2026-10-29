"""Conforming quad refinement with projection onto the original mesh."""
import numpy as np
import point_cloud_utils as pcu
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from plyfile import PlyData, PlyElement


def write_quad_ply(path, vertices, faces):
    vertex_data = np.empty(len(vertices), dtype=[('x','f4'),('y','f4'),('z','f4')])
    for i, key in enumerate(('x','y','z')):
        vertex_data[key] = vertices[:,i]
    face_data = np.empty(len(faces), dtype=[('vertex_indices','i4',(4,))])
    face_data['vertex_indices'] = faces
    PlyData([PlyElement.describe(vertex_data,'vertex'), PlyElement.describe(face_data,'face')]).write(str(path))


def split_quads(vertices, faces, marked_edges, edges, face_edges, surface_vertices, surface_faces, project=True):
    added = []
    midpoint_ids = np.full(len(edges), -1, dtype=np.int64)
    selected = np.flatnonzero(marked_edges)
    midpoint_ids[selected] = len(vertices) + np.arange(len(selected))
    added.extend(vertices[edges[selected]].mean(axis=1))
    result = []
    for face, ids in zip(faces, face_edges):
        mask = marked_edges[ids]
        a, b, c, d = face
        if not mask.any():
            result.append(face)
        elif mask.all():
            center = len(vertices) + len(added)
            added.append(vertices[face].mean(axis=0))
            m0, m1, m2, m3 = midpoint_ids[ids]
            result.extend(((a,m0,center,m3), (m0,b,m1,center), (center,m1,c,m2), (m3,center,m2,d)))
        elif mask[0] and mask[2] and mask.sum() == 2:
            m0, m2 = midpoint_ids[ids[[0,2]]]
            result.extend(((a,m0,m2,d), (m0,b,c,m2)))
        elif mask[1] and mask[3] and mask.sum() == 2:
            m1, m3 = midpoint_ids[ids[[1,3]]]
            result.extend(((a,b,m1,m3), (m3,m1,c,d)))
        else:
            raise RuntimeError('Refinement would create a T-junction.')
    added = np.asarray(added, dtype=np.float64)
    if len(added):
        if project:
            _, triangle_ids, barycentric = pcu.closest_points_on_mesh(added, surface_vertices, surface_faces)
            added = pcu.interpolate_barycentric_coords(surface_faces, triangle_ids, barycentric, surface_vertices)
        vertices = np.concatenate((vertices, added), axis=0)
    return vertices, np.asarray(result, dtype=np.int64)


def refine_to_target(vertices, faces, surface_vertices, surface_faces, target_faces, max_faces):
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    surface_vertices = np.ascontiguousarray(surface_vertices, dtype=np.float64)
    surface_faces = np.ascontiguousarray(surface_faces, dtype=np.int32)
    target = min(target_faces, max_faces)
    initial_faces = len(faces)
    rounds = []
    if target <= initial_faces:
        return vertices, faces, {'target': target_faces, 'budget': max_faces, 'initial_faces': initial_faces,
                                 'output_faces': len(faces), 'rounds': rounds}
    while len(faces) < target:
        raw_edges = np.stack((faces, np.roll(faces, -1, axis=1)), axis=-1)
        edges, inverse = np.unique(np.sort(raw_edges.reshape(-1,2), axis=1), axis=0, return_inverse=True)
        face_edges = inverse.reshape(-1,4)
        if len(faces)*4 <= target:
            marked = np.ones(len(edges), dtype=bool)
        else:
            opposite = np.concatenate((face_edges[:,[0,2]], face_edges[:,[1,3]]), axis=0)
            graph = coo_matrix((np.ones(len(opposite)), (opposite[:,0], opposite[:,1])), shape=(len(edges),len(edges))).tocsr()
            count, labels = connected_components(graph, directed=False)
            lengths = np.linalg.norm(vertices[edges[:,0]]-vertices[edges[:,1]], axis=1)
            priorities = np.bincount(labels, weights=lengths) / np.bincount(labels)
            face_labels = labels[face_edges]
            face_marks = np.zeros(len(faces), dtype=np.int8)
            selected = np.zeros(count, dtype=bool)
            predicted_count = len(faces)
            costs = np.array([1,0,2,0,4])
            for group in np.argsort(-priorities, kind='stable'):
                affected = np.flatnonzero(np.any(face_labels == group, axis=1))
                increment = np.count_nonzero(face_labels[affected] == group, axis=1)
                before = face_marks[affected]
                after = before + increment
                delta = int((costs[after]-costs[before]).sum())
                if predicted_count + delta > max_faces:
                    continue
                selected[group] = True
                face_marks[affected] = after
                predicted_count += delta
                if predicted_count >= target:
                    break
            marked = selected[labels]
            if not marked.any():
                break
        previous = len(faces)
        vertices, faces = split_quads(vertices, faces, marked, edges, face_edges, surface_vertices, surface_faces)
        rounds.append({'before': previous, 'after': len(faces)})
        if len(faces) <= previous:
            break
    return vertices, faces, {'target': target_faces, 'budget': max_faces, 'initial_faces': initial_faces,
                             'output_faces': len(faces), 'rounds': rounds,
                             'projection': 'new vertices projected to original input triangles; base vertices preserved'}
