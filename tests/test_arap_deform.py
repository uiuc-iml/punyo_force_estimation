import time
import numpy as np
import scipy
import matplotlib.pyplot as plt
from pathlib import Path
import open3d as o3d
import imageio
import robust_laplacian

from src.punyo_force_estimation.force_module.force_from_punyo import ForceFromPunyo
from src.punyo_force_estimation.utils import load_frames, load_data, unpack_mesh, PC_ROTATION_MATRIX, mesh_plane_z
from src.punyo_force_estimation.force_module.material_model import LinearPlaneStressModel, DebugPlaneStressModel, LinearSpringModel

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


if __name__ == "__main__":
    working_dir = "data/shoe"
    move_idx_lst = [106]
    move_direction = np.array([0.0, 0.0, 1.0])

    total_frame = 148
    rgbs = [imageio.v2.imread(f"{working_dir}/raw/punyo_color_{i}.png") for i in range(total_frame)]
    depths = [imageio.v2.imread(f"{working_dir}/raw/punyo_depth_{i}.png") for i in range(total_frame)]
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

    print('boundary_mask shape:', boundary_mask.shape)

    static_ids = np.where(boundary_mask == 1)[0]
    static_pos = points[static_ids, :]

    punyo_mesh = o3d.geometry.TriangleMesh()
    punyo_mesh.vertices = o3d.utility.Vector3dVector(points)
    punyo_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    punyo_mesh.compute_vertex_normals()
    punyo_mesh.paint_uniform_color([0.0, 0.7, 0.7])

    handle_ids = move_idx_lst
    handle_pos = [points[move_idx_lst[0], :] + np.array([-0.00, -0.00, -0.01])]
    constraint_ids = o3d.utility.IntVector(static_ids.tolist() + handle_ids)
    constraint_pos = o3d.utility.Vector3dVector(np.append(static_pos, handle_pos, axis=0))

    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        mesh_prime = punyo_mesh.deform_as_rigid_as_possible(constraint_ids,
                                                            constraint_pos,
                                                            max_iter=50)
    mesh_prime.compute_vertex_normals()
    mesh_prime.paint_uniform_color([0.7, 0.7, 0.0])

    # visualize the new mesh
    o3d.visualization.draw_geometries([mesh_prime])
    o3d.visualization.draw_geometries([mesh_prime, punyo_mesh])
