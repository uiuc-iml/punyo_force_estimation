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


if __name__ == "__main__":
    working_dir = "data/shoe"
    move_idx_lst = [106]
    move_direction = np.array([0.0, 0.0, 1.0])

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
    punyo_pcd = o3d.geometry.PointCloud()
    punyo_pcd.points = o3d.utility.Vector3dVector(punyo_rest)
    # o3d.visualization.draw_geometries_with_editing([punyo_pcd])

    boundary_mask[move_idx_lst] = 1

    boundary_mask_flatten = np.repeat(boundary_mask[..., None], 3, axis=1).reshape(-1)
    boundary_mask_flatten = boundary_mask_flatten == 1

    num_fix_pts = np.sum(boundary_mask_flatten)

    # TODO: load K_b matrix (K_v matrix is the stiffness of VSF)
    K_B = force_estimator.force_predictor.static_K.toarray()
    K_ff = K_B[~boundary_mask_flatten,:][:,~boundary_mask_flatten]
    K_fb = K_B[~boundary_mask_flatten,:][:,boundary_mask_flatten]

    print('K_B shape:', K_B.shape)
    print('K_ff shape:', K_ff.shape)
    print('K_fb shape:', K_fb.shape)

    move_init_pts = punyo_rest[move_idx_lst, :]
    free_init_pts = punyo_rest[boundary_mask == 0, :]
    print('move_init_pts shape:', move_init_pts.shape)
    print('free_init_pts shape:', free_init_pts.shape)

    punyo_deformed_pcd = o3d.geometry.PointCloud()
    punyo_deformed_pcd.points = o3d.utility.Vector3dVector(punyo_rest)

    def update_pts(move_dist):
        move_pts = move_init_pts + move_direction * move_dist
        punyo_rest[move_idx_lst, :] = move_pts

        u_b = np.zeros(num_fix_pts)
        u_b[:3*len(move_idx_lst)] = (move_direction * move_dist).flatten()
        print('u_b shape:', u_b.shape)

        u_f = np.linalg.solve(K_ff, -K_fb @ u_b)
        print('u_f shape:', u_f.shape)

        free_pts = free_init_pts + u_f.reshape(-1, 3)

        curr_pts = np.zeros(punyo_rest.shape)
        curr_pts[boundary_mask == 1] = move_pts
        curr_pts[boundary_mask == 0] = free_pts

        punyo_deformed_pcd.points = o3d.utility.Vector3dVector(curr_pts)
    
    current_move_dist = 0.0
    def create_update_move_dist(delta_dist):

        def update_move_dist(vis):
            global current_move_dist
            current_move_dist += delta_dist
            print('current_move_dist:', current_move_dist)
            update_pts(current_move_dist)

            vis.update_geometry(punyo_deformed_pcd)
            vis.poll_events()
            vis.update_renderer()
        return update_move_dist

    key_to_callback = {}
    key_to_callback[ord('A')] = create_update_move_dist(+0.001)
    key_to_callback[ord('S')] = create_update_move_dist(-0.001)

    # o3d.visualization.draw_geometries([punyo_deformed_pcd])
    o3d.visualization.draw_geometries_with_key_callbacks([punyo_deformed_pcd], key_to_callback)
