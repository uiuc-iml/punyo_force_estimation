import time
import numpy as np
import scipy
import matplotlib.pyplot as plt
from pathlib import Path
import open3d as o3d
import imageio
import trimesh

from punyo_force_estimation.force_module.force_from_punyo import ForceFromPunyo
from punyo_force_estimation.utils import load_frames, load_data, unpack_mesh, PC_ROTATION_MATRIX, mesh_plane_z

sx = 385.263
sy = 385.263
image_center = [307.943, 241.596]
intrinsic = o3d.camera.PinholeCameraIntrinsic(640, 480, sx, sy, image_center[0], image_center[1])

def rgbd_to_pc(color_img, depth_img, depth_scale=10000.0, depth_max=1.0, gray_img=False):
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_img, depth_img, depth_scale=depth_scale,
                                                              depth_trunc=depth_max, convert_rgb_to_intensity=gray_img)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    # visualize pcd
    # o3d.visualization.draw_geometries([pcd])
    return np.asarray(pcd.points), np.asarray(pcd.colors)

def compute_A(vertices, triangles, points):
    tri_mesh = trimesh.Trimesh(vertices, triangles)
    # find closest triangle
    closest_tri, dist, closest_tri_idx = trimesh.proximity.closest_point(tri_mesh, points)
    triangles_nearest = tri_mesh.triangles[closest_tri_idx]
    barycentric = trimesh.triangles.points_to_barycentric(triangles_nearest, points)

    vertices_flatten = vertices.flatten()
    points_flatten = points.flatten()
    A = np.zeros((points_flatten.shape[0], vertices_flatten.shape[0]))
    point_idx = np.arange(points_flatten.shape[0] // 3)
    for pt_idx, bary, tri_idx in zip(point_idx, barycentric, closest_tri_idx):
        tri = triangles[tri_idx]
        for i, tri_v in enumerate(tri):
            A[pt_idx*3:pt_idx*3+3, tri_v*3:tri_v*3+3] = np.eye(3) * bary[i]
    return A, closest_tri_idx


working_dir = "data/shoe"

total_frame = 148
rgbs = [imageio.imread(f"{working_dir}/raw/punyo_color_{i}.png") for i in range(total_frame)]
depths = [imageio.imread(f"{working_dir}/raw/punyo_depth_{i}.png") for i in range(total_frame)]
pressure_scale = 100
pressures = np.load(f"{working_dir}/raw/pressure.npy") * pressure_scale
rgbs_o3d = [o3d.geometry.Image(rgbs[i]) for i in range(total_frame)]
depths_o3d = [o3d.geometry.Image(depths[i]) for i in range(total_frame)]
pointclouds = [rgbd_to_pc(rgbs_o3d[i], depths_o3d[i])[0] for i in range(total_frame)]
# downsample pointclouds
pointclouds = [pointclouds[i][::10, :] for i in range(total_frame)]

reference_frames = [0, 1, 2, 3, 4]
reference_rgbs = [rgbs[i] for i in reference_frames]
reference_pcds = [pointclouds[i] for i in reference_frames]
reference_pressures = [pressures[i] for i in reference_frames]

ref_dir = "src/punyo_force_estimation/ref_data"

points, triangles, boundary, boundary_mask = unpack_mesh(f"{ref_dir}/equalized.vtk")
force_estimator = ForceFromPunyo(reference_rgbs, reference_pcds, reference_pressures, points, triangles, boundary, 
                                    rest_internal_force=None, precompile=False, verbose=True)


punyo_rest = force_estimator.undeformed_points.numpy()

vsf_curr = np.array([[0.03989953, 0.0002754, 0.08455362]])

boundary_mask_flatten = np.repeat(boundary_mask[..., None], 3, axis=1).reshape(-1)
boundary_mask_flatten = boundary_mask_flatten == 1

# TODO: load K_b matrix (K_v matrix is the stiffness of VSF)
K_B = force_estimator.force_predictor.static_K.toarray()
K_B = K_B[~boundary_mask_flatten,:][:,~boundary_mask_flatten]
K_V = np.array(10000).repeat(3)
K_V = np.diag(K_V)

# TODO: solve the least square problem

K_B_eigen_val, K_B_eigen_vec = np.linalg.eig(K_B)
K_B_eigen_val = np.real(K_B_eigen_val)
K_B_eigen_vec = np.real(K_B_eigen_vec)
K_B_eigen_val = np.maximum(K_B_eigen_val, 0.0)
L_B = np.diag(np.sqrt(K_B_eigen_val)) @ K_B_eigen_vec.T
assert np.isclose(L_B.T @ L_B, K_B).all() # if error, K_B is not positive semi-definite, probably a bug in the code
K_B = L_B.T @ L_B


K_V = np.maximum(K_V, 0.0)
L_V = np.sqrt(K_V)
assert np.isclose(L_V.T @ L_V, K_V).all()

num_steps = 10
for i in range(num_steps):
    vsf_rest = np.mean(punyo_rest, axis=0, keepdims=True) * i / num_steps + vsf_curr * (num_steps - i) / num_steps
    # vsf_curr = vsf_rest.copy()
    # while True:
    #     tri_mesh = trimesh.Trimesh(punyo_rest, triangles)
    #     # find closest triangle
    #     closest_tri, dist, closest_tri_idx = trimesh.proximity.closest_point(tri_mesh, vsf_curr)
    #     if dist[0] < 1e-4:
    #         break
    #     vsf_curr += np.array([0.0, 0.0, 0.00003])
    # print(vsf_curr)

    # TODO: compute A matrix (check whether distance is reasonable)
    A, closest_tri_idx = compute_A(punyo_rest, triangles, vsf_curr) # triangles_nearest for test
    A = A[:,~boundary_mask_flatten]
    C = vsf_curr.flatten() - vsf_rest.flatten()


    penalty_scale = 0
    penalty_a = np.zeros((L_B.shape[0], L_B.shape[0]))
    penalty_b = np.zeros(L_B.shape[0])

    # penalty_scale = 0
    # penalty_a = np.ones(L_B.shape[0]) * penalty_scale
    # penalty_a[boundary_mask_flatten] = 1e10
    # penalty_a = np.diag(penalty_a)
    # penalty_b = np.zeros(L_B.shape[0])

    # a = np.concatenate([L_B, L_V @ A, penalty_a], axis=0)
    # b = np.concatenate([np.zeros(L_B.shape[0]), -L_V @ C, penalty_b], axis=0)

    a = np.concatenate([L_B, L_V @ A], axis=0)
    b = np.concatenate([np.zeros(L_B.shape[0]), -L_V @ C], axis=0)


    print('least square a:', a.shape)
    print('least square b:', b.shape)

    u_B_masked = np.linalg.lstsq(a, b, rcond=None)[0]
    u_B = np.zeros(boundary_mask_flatten.shape[0])
    u_B[~boundary_mask_flatten] = u_B_masked
    u_V = A @ u_B_masked + C


    # u_B = np.linalg.lstsq(a, b, rcond=None)[0]
    # u_B_masked = u_B
    # u_V = A @ u_B + C

    print('u_B:', u_B.shape)
    print('u_V:', u_V.shape)

    print('before least square')
    print('punyo energy:', 0)
    dvsf = C
    print('vsf energy:', np.sum(dvsf @ K_V @ dvsf))
    print('total energy:', np.sum(dvsf @ K_V @ dvsf))

    print('after least square')
    print('punyo energy:', np.sum(u_B_masked @ L_B.T @ L_B @ u_B_masked))
    dvsf = u_V
    print('vsf energy:', np.sum(dvsf @ K_V @ dvsf))
    print('total energy:', np.sum(dvsf @ K_V @ dvsf) + np.sum(u_B_masked @ L_B.T @ L_B @ u_B_masked))
    print('penalty:', np.sum(penalty_scale * u_B_masked @ u_B_masked))

    # u_B_pts = u_B.reshape(-1, 3)
    # u_B_norm = np.linalg.norm(u_B_pts, axis=1)
    # sort_idx = np.argsort(u_B_norm)
    # print(sort_idx[-10:])

    # bad_pts_idx = np.array([144, 291, 350, 122, 74, 108, 145, 107, 130, 106])
    bad_pts_idx = np.array([144, 122, 130])
    u_B_masked_debug = u_B_masked.copy()
    print('debug energy 1', u_B_masked_debug @ L_B.T @ L_B @ u_B_masked_debug)
    u_B_pts = u_B.copy().reshape(-1, 3)
    u_B_pts[bad_pts_idx] = 0.0
    u_B_masked_debug = u_B_pts.flatten()[~boundary_mask_flatten]
    print('debug energy 2', u_B_masked_debug @ L_B.T @ L_B @ u_B_masked_debug)


    # TODO: visualize the deformation of punyo mesh and vsf points

    punyo_deformed_pts_o3d = o3d.geometry.PointCloud()
    punyo_deformed_pts_o3d.points = o3d.utility.Vector3dVector(punyo_rest + u_B.reshape(-1, 3))
    color = np.zeros((punyo_rest.shape[0], 3))
    color[:] = [0.0, 0.0, 1.0]
    color[bad_pts_idx] = [1.0, 0.0, 1.0]
    punyo_deformed_pts_o3d.colors = o3d.utility.Vector3dVector(color)

    punyo_rest_pts_o3d = o3d.geometry.PointCloud()
    punyo_rest_pts_o3d.points = o3d.utility.Vector3dVector(punyo_rest)
    punyo_rest_pts_o3d.paint_uniform_color([0.0, 1.0, 1.0])

    vsf_rest_pts_o3d = o3d.geometry.PointCloud()
    vsf_rest_pts_o3d.points = o3d.utility.Vector3dVector(vsf_rest)
    vsf_rest_pts_o3d.paint_uniform_color([0.0, 1.0, 0.0])

    vsf_curr_pts_o3d = o3d.geometry.PointCloud()
    vsf_curr_pts_o3d.points = o3d.utility.Vector3dVector(vsf_curr)
    vsf_curr_pts_o3d.paint_uniform_color([1.0, 0.0, 0.0])


    triangle_o3d = o3d.geometry.TriangleMesh()
    triangle_o3d.vertices = o3d.utility.Vector3dVector(punyo_rest + u_B.reshape(-1, 3))
    triangle_o3d.triangles = o3d.utility.Vector3iVector(triangles)
    triangle_o3d.compute_vertex_normals()

    boundary_pts_o3d = o3d.geometry.PointCloud()
    boundary_pts_o3d.points = o3d.utility.Vector3dVector(punyo_rest[boundary_mask == 1])
    boundary_pts_o3d.paint_uniform_color([1.0, 0.0, 0.0])


    o3d.visualization.draw_geometries([punyo_deformed_pts_o3d, vsf_rest_pts_o3d, vsf_curr_pts_o3d, punyo_rest_pts_o3d])
