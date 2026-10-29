import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyParseError


def audit(path):
    try:
        ply = PlyData.read(str(path), known_list_len={'face': {'vertex_indices': 4}})
        quads = ply['face']['vertex_indices']
    except PlyParseError:
        ply = PlyData.read(str(path))
        quads = np.array([f for f in ply['face']['vertex_indices'] if len(f) == 4], dtype=np.int32)
    vertices = np.column_stack([ply['vertex'][k] for k in ('x', 'y', 'z')]).astype(np.float64)
    faces = ply['face']['vertex_indices']
    report = {'file': str(path), 'faces': len(faces), 'quad_fraction': len(quads) / len(faces)}
    if not len(quads):
        return report
    ratio = np.empty(len(quads))
    minimum_angles = np.empty(len(quads))
    normal_dot = np.empty(len(quads))
    plane_error = np.empty(len(quads))
    for start in range(0, len(quads), 65536):
        stop = start+65536
        p = vertices[quads[start:stop]]
        forward = np.roll(p, -1, axis=1) - p
        backward = np.roll(p, 1, axis=1) - p
        lengths = np.linalg.norm(forward, axis=-1)
        ratio[start:stop] = lengths.max(axis=1) / np.maximum(lengths.min(axis=1), 1e-15)
        cosines = (forward * backward).sum(axis=-1) / np.maximum(np.linalg.norm(forward, axis=-1) * np.linalg.norm(backward, axis=-1), 1e-15)
        angles = np.rad2deg(np.arccos(np.clip(cosines, -1, 1)))
        minimum_angles[start:stop] = angles.min(axis=1)
        n1 = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        n2 = np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0])
        normal_dot[start:stop] = (n1*n2).sum(axis=1) / np.maximum(np.linalg.norm(n1, axis=1) * np.linalg.norm(n2, axis=1), 1e-15)
        plane_error[start:stop] = np.abs(((p[:, 3] - p[:, 0])*n1).sum(axis=1)) / np.maximum(np.linalg.norm(n1, axis=1) * lengths.max(axis=1), 1e-15)
    report.update(
        edge_ratio_median=float(np.median(ratio)), edge_ratio_p95=float(np.percentile(ratio, 95)),
        fraction_edge_ratio_above_5=float((ratio > 5).mean()),
        minimum_corner_angle_p05=float(np.percentile(minimum_angles, 5)),
        fraction_quads_corner_below_10deg=float((minimum_angles < 10).mean()),
        fraction_folded_quads=float((normal_dot < 0).mean()),
        nonplanarity_p95_relative_edge=float(np.percentile(plane_error, 95)),
    )
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+')
    args = parser.parse_args()
    for name in args.files:
        path = Path(name)
        report = audit(path)
        path.with_suffix('.audit.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
