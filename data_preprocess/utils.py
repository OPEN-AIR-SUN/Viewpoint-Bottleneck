import os
import numpy as np
import pandas as pd
from plyfile import PlyData, PlyElement
from numpy.linalg import matrix_rank, inv


def read_plyfile(filepath):
    """Read ply file and return it as numpy array. Returns None if emtpy."""
    with open(filepath, 'rb') as f:
        plydata = PlyData.read(f)
    if plydata.elements:
        return pd.DataFrame(plydata.elements[0].data).values


def save_point_cloud(points_3d, filename, binary=True, with_label=False, verbose=True):
    """Save an RGB point cloud as a PLY file.

    Args:
      points_3d: Nx6 matrix where points_3d[:, :3] are the XYZ coordinates and points_3d[:, 4:] are
          the RGB values. If Nx3 matrix, save all points with [128, 128, 128] (gray) color.
    """
    assert points_3d.ndim == 2
    if with_label:
        assert points_3d.shape[1] == 7
        python_types = (float, float, float, int, int, int, int)
        npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                     ('blue', 'u1'), ('label', 'u1')]
    else:
        if points_3d.shape[1] == 3:
            gray_concat = np.tile(np.array([128], dtype=np.uint8), (points_3d.shape[0], 3))
            points_3d = np.hstack((points_3d, gray_concat))
        assert points_3d.shape[1] == 6
        python_types = (float, float, float, int, int, int)
        npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                     ('blue', 'u1')]
    if binary is True:
        # Format into NumPy structured array
        vertices = []
        for row_idx in range(points_3d.shape[0]):
            cur_point = points_3d[row_idx]
            vertices.append(tuple(dtype(point) for dtype, point in zip(python_types, cur_point)))
        vertices_array = np.array(vertices, dtype=npy_types)
        el = PlyElement.describe(vertices_array, 'vertex')

        # Write
        PlyData([el]).write(filename)
    else:
        # PlyData([el], text=True).write(filename)
        with open(filename, 'w') as f:
            f.write('ply\n'
                    'format ascii 1.0\n'
                    'element vertex %d\n'
                    'property float x\n'
                    'property float y\n'
                    'property float z\n'
                    'property uchar red\n'
                    'property uchar green\n'
                    'property uchar blue\n'
                    'property uchar alpha\n'
                    'end_header\n' % points_3d.shape[0])
            for row_idx in range(points_3d.shape[0]):
                X, Y, Z, R, G, B = points_3d[row_idx]
                f.write('%f %f %f %d %d %d 0\n' % (X, Y, Z, R, G, B))
    if verbose is True:
        print('Saved point cloud to: %s' % filename)
