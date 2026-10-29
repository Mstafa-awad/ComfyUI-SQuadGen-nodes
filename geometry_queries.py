"""Stable array-shaped mesh queries, including PCU's broken singleton case."""
import numpy as np
import point_cloud_utils as pcu


def closest_points(points,vertices,faces):
    points=np.ascontiguousarray(points,dtype=np.float64)
    if points.ndim!=2 or points.shape[1]!=3:raise ValueError('Query points must have shape (N,3)')
    count=len(points)
    if not count:return np.empty(0),np.empty(0,dtype=np.int32),np.empty((0,3))
    # This installed PCU build reads a one-row point matrix incorrectly. Merely
    # reshaping its outputs cannot repair the returned distance/barycentrics.
    query=np.repeat(points,2,axis=0) if count==1 else points
    distance,ids,bary=pcu.closest_points_on_mesh(query,vertices,faces)
    return np.asarray(distance).reshape(-1)[:count],np.asarray(ids).reshape(-1)[:count],np.asarray(bary).reshape(-1,3)[:count]
