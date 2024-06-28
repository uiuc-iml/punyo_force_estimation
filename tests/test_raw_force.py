import time
import numpy as np
import scipy
import matplotlib.pyplot as plt
from pathlib import Path
import open3d as o3d
import imageio
import trimesh

import matplotlib
matplotlib.use('TkAgg')

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
    force_estimator = ForceFromPunyo(reference_rgbs, reference_pcds, reference_pressures, points, triangles, boundary, 
                                     rest_internal_force=None, material_model=LinearSpringModel(), precompile=False, verbose=True)
    
    # force_estimator = ForceFromPunyo(reference_rgbs, reference_pcds, reference_pressures, points, triangles, boundary, 
    #                                  rest_internal_force=None, precompile=False, verbose=False)


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

    # print mean, std, max, min of K_B
    print('K_B mean:', np.mean(K_B))
    print('K_B std:', np.std(K_B))
    print('K_B max:', np.max(K_B))
    print('K_B min:', np.min(K_B))

    # plt.hist(K_B.flatten(), bins=100)
    # plt.show()

    move_init_pts = punyo_rest[move_idx_lst, :]
    free_init_pts = punyo_rest[boundary_mask == 0, :]
    
    colors = np.zeros(punyo_rest.shape)
    colors[move_idx_lst, :] = [1, 0, 0]
    colors[boundary_mask == 1, :] += [0, 0, 1]
    colors[boundary_mask == 0, :] = [0, 1, 0]
    punyo_pcd.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries_with_editing([punyo_pcd])

    punyo_deformed_pcd = o3d.geometry.PointCloud()
    punyo_deformed_pcd.points = o3d.utility.Vector3dVector(punyo_rest)
    punyo_deformed_pcd.colors = o3d.utility.Vector3dVector(colors)

    punyo_deformed_mesh = o3d.geometry.TriangleMesh()
    punyo_deformed_mesh.vertices = o3d.utility.Vector3dVector(punyo_rest)
    punyo_deformed_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    punyo_deformed_mesh.compute_vertex_normals()

    start_time = time.time()
    contact_forces_lst = []
    opt_deformed_points_lst = []
    raw_deformed_points_lst = []
    for pressure, pcd, rgb in zip(pressures, pointclouds, rgbs):
        # raw_deformed_points, data = force_estimator.deformation_estimator.estimate(rgb, pcd)
        # raw_deformed_points = raw_deformed_points @ PC_ROTATION_MATRIX
        # raw_deformed_points_lst.append(raw_deformed_points)

        force_estimator.update(rgb, pcd, pressure)
       
        raw_deformed_points_lst.append(force_estimator._raw_points.copy())
        opt_deformed_points_lst.append(force_estimator.current_points.copy())
        contact_forces_lst.append(force_estimator.observed_force.copy())
    print('Time elapsed:', time.time() - start_time)

    def update_pts(deformed_points, contact_forces):
        # delta_points = deformed_points - punyo_rest

        # if contact_forces is None:
        #     force_internal = K_B @ delta_points.reshape(-1)
        #     force_internal = 0.1*force_internal.reshape(-1, 3)
        
        # current_points = punyo_rest + delta_points

        punyo_deformed_pcd.points = o3d.utility.Vector3dVector(deformed_points)
        punyo_deformed_pcd.normals = o3d.utility.Vector3dVector(contact_forces)
        punyo_deformed_mesh.vertices = o3d.utility.Vector3dVector(deformed_points)
        punyo_deformed_mesh.compute_vertex_normals()
    
    current_frame_index = 0
    force_scale = 1.0
    def create_change_frame(delta_index):

        def change_frame(vis):
            global current_frame_index
            if current_frame_index + delta_index < 0 or \
               current_frame_index + delta_index >= total_frame:
                return
            else:
                current_frame_index += delta_index
            print('current frame:', current_frame_index)
            
            # raw_deformed_points = raw_deformed_points_lst[current_frame_index]
            # update_pts(raw_deformed_points)            

            update_pts(opt_deformed_points_lst[current_frame_index], 
                       -force_scale*contact_forces_lst[current_frame_index])

            vis.update_geometry(punyo_deformed_pcd)
            vis.update_geometry(punyo_deformed_mesh)
            vis.poll_events()
            vis.update_renderer()
        return change_frame

    # o3d.visualization.draw_geometries([punyo_deformed_pcd])

    key_to_callback = {}
    key_to_callback[ord('A')] = create_change_frame(1)
    key_to_callback[ord('S')] = create_change_frame(-1)

    # Key to change force scale
    def func(vis, delta):
        global force_scale
        force_scale = force_scale + delta
        print('force_scale:', force_scale)
    
    key_to_callback[ord('Q')] = lambda vis: func(vis, 0.01)
    key_to_callback[ord('W')] = lambda vis: func(vis, -0.01)

    o3d.visualization.draw_geometries_with_key_callbacks([punyo_deformed_pcd, punyo_deformed_mesh], key_to_callback)

    # animate_callback = create_change_frame(1)
    # o3d.visualization.draw_geometries_with_animation_callback([punyo_deformed_pcd, punyo_deformed_mesh], animate_callback)