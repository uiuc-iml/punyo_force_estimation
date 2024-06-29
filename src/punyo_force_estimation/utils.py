import meshio
import numpy as np
import open3d as o3d
from .force_module.deformation_estimator import DeformationEstimation

import os
REFERENCE_MESH_PATH = os.path.join(os.path.dirname(__file__), "ref_data/equalized.vtk")

def pointcloud(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def make_o3d_mesh(points, triangles):
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(points)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    o3d_mesh.compute_vertex_normals()
    return o3d_mesh

def load_data(working_dir, n, subsample=True):
    # Conversion from hPa to Pa
    # google: hectopascal
    pressure = np.load(f"{working_dir}/pressure_{n}.npy")*100
    pointcloud = np.load(f"{working_dir}/pc_{n}.npy").reshape(-1, 3)
    if subsample:
        idx = np.random.randint(len(pointcloud), size=len(pointcloud)//10)
        pointcloud = pointcloud[idx, :]
    rgb = np.load(f"{working_dir}/rgb_{n}.npy")
    return pressure, pointcloud, rgb

def load_frames(working_dir, reference_frames):
    reference_rgbs = []
    reference_pcds = []
    reference_pressures = []
    for i, frame in enumerate(reference_frames):
        pressure, pcd, rgb = load_data(working_dir, frame)
        reference_rgbs.append(rgb)
        reference_pcds.append(pcd)
        reference_pressures.append(pressure)
    return reference_rgbs, reference_pcds, reference_pressures

"""
gets reference mesh and pressure

return: (deformation_estimator, undeformed_mesh_points, reference_pressure)
"""
def get_reference_mesh(reference_rgbs, reference_pcds, reference_pressures, points, triangles=None, boundary=None, smooth_iters=10):
    deformation_estimator = DeformationEstimation(reference_rgbs[0], reference_pcds[0],
                            PC_ROTATION_MATRIX, mesh_plane_z, points, fixed_points=boundary)

    undeformed_mesh_points_lst = np.empty((len(reference_rgbs), len(points), 3))
    reference_pressure = 0
    for i, (rgb, pcd, pressure) in enumerate(zip(reference_rgbs, reference_pcds, reference_pressures)):
        undeformed_mesh_points, _ = deformation_estimator.estimate(rgb, pcd)
        undeformed_mesh_points_lst[i, :, :] = undeformed_mesh_points
        reference_pressure += pressure

    reference_pressure = reference_pressure / len(reference_rgbs)

    

    undeformed_mesh_points = np.nanmean(undeformed_mesh_points_lst, axis=0) @ PC_ROTATION_MATRIX
    if smooth_iters > 0:
        undeformed_mesh_points = smooth_mesh_taubin(undeformed_mesh_points, triangles, boundary, num_iters=smooth_iters)

    return deformation_estimator, undeformed_mesh_points, reference_pressure


def smooth_mesh_taubin(points, triangles, fix_points=[], num_iters=10):
    tmp_mesh = o3d.geometry.TriangleMesh()
    tmp_mesh.vertices = o3d.utility.Vector3dVector(points)
    tmp_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    tmp_mesh.compute_vertex_normals()
    smoothed_points = np.asarray(tmp_mesh.filter_smooth_taubin(number_of_iterations=num_iters).vertices)

    for n in fix_points:
        #undeformed_mesh_points[n] = pc_rotM @ reference_mesh.points[n].reshape(3)
        smoothed_points[n] = points[n].reshape(3)

    return smoothed_points

def unpack_mesh(mesh_fn):
    mesh = meshio.read(mesh_fn)
    points = mesh.points.astype(np.float32)
    triangles = np.array(mesh.cells_dict['triangle'], dtype=np.int32)
    boundary = np.array(mesh.cells_dict['vertex'], dtype=np.int32)
    boundary_mask = np.zeros(len(points), dtype=np.uint8)
    for n in boundary:
        boundary_mask[n] = 1
    return points, triangles, boundary, boundary_mask

pc_rotation, mesh_plane_z = 25*np.pi/180, 0.054
PC_ROTATION_MATRIX = np.array([
    [np.cos(pc_rotation), 0, -np.sin(pc_rotation)],
    [0, 1, 0],
    [np.sin(pc_rotation), 0, np.cos(pc_rotation)],
])

def estimate_tri_normals(vertices, triangles):
    normals = np.zeros((len(vertices), 3))
    for tri_idx, (i, j, k) in enumerate(triangles):
        a = vertices[j] - vertices[i]
        b = vertices[k] - vertices[i]
        normal = np.cross(a, b)
        normals[i] += normal
        normals[j] += normal
        normals[k] += normal
    return normals / np.linalg.norm(normals, axis=1).reshape(-1, 1)

def estimate_point_areas(vertices, triangles):
    areas = np.zeros(len(vertices))
    for tri_idx, (i, j, k) in enumerate(triangles):
        a = vertices[j] - vertices[i]
        b = vertices[k] - vertices[i]
        area = np.linalg.norm(np.cross(a, b))
        areas[i] += area
        areas[j] += area
        areas[k] += area
    return areas/6

# actually pretty good lol
FT_Y_SCALING = 1
FT_X_SCALING = 1/1.126629595411067
FT_Z_SCALING = 1/1.388220845057926
