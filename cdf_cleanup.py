import numpy as np
from scipy import sparse


def smooth_fields(mesh, fields, iterations=4):
    """Reduce scalar prediction noise on the face graph without moving geometry."""
    adjacent = mesh.face_adjacency
    normals = mesh.face_normals
    dot = (normals[adjacent[:, 0]]*normals[adjacent[:, 1]]).sum(axis=1)
    weights = np.exp(-np.maximum(0, 1-dot)/.05)
    ids = np.arange(len(mesh.faces))
    graph = sparse.coo_matrix((np.r_[weights, weights, np.ones(len(ids))],
                              (np.r_[adjacent[:, 0], adjacent[:, 1], ids],
                               np.r_[adjacent[:, 1], adjacent[:, 0], ids])),
                             shape=(len(ids), len(ids))).tocsr()
    graph = sparse.diags(1/np.asarray(graph.sum(axis=1)).ravel())@graph
    answer = dict(fields)
    for key in ('cdf', 'dcdf'):
        value = np.array(fields[key], dtype=np.float32)
        for i in range(iterations):
            value = value+(.5 if i%2 == 0 else -.53)*(graph@value-value)
        answer[key] = np.clip(value, 0, 1).astype(np.float32)
    return answer
