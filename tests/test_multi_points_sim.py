import time
import numpy as np
import scipy
import matplotlib.pyplot as plt
from pathlib import Path
import open3d as o3d
import imageio
import robust_laplacian
import trimesh
from trimesh import Trimesh

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

angle = np.radians(20)
kinova_punyo2ee_np = np.array([[ 0            ,-1, 0            ,  0 ],
                        [ np.sin(angle), 0,-np.cos(angle),  0.021110 ],
                        [ np.cos(angle), 0, np.sin(angle),  0.096142 ],
                        [ 0            , 0, 0.           ,  1]])

if __name__ == "__main__":
    working_dir = "data/shoe"
    move_idx_lst = [106]
    move_direction = np.array([0.0, 0.0, 1.0])

    vsf_rest_pts = np.load('/home/motion/plant-model/out_data/obj_asset_dataset/white_adidas_superstar_angle00/rest_pts.npy')
    vsf_pcd = o3d.geometry.PointCloud()
    vsf_pcd.points = o3d.utility.Vector3dVector(vsf_rest_pts)

    punyo_curr_pts = np.load('/home/motion/plant-model/out_data/small_obj_sim_deform/debug_white_adidas_superstar_angle00_trail1/seq_000/punyo_curr_pts_00057.npy')

    punyo_curr_pcd = o3d.geometry.PointCloud()
    punyo_curr_pcd.points = o3d.utility.Vector3dVector(punyo_curr_pts)

    total_frame = 58
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
    #                                  rest_internal_force=None, precompile=False, verbose=True)

    sim_out_dir = '/home/motion/plant-model/out_data/small_obj_sim_deform/debug_white_adidas_superstar_angle00_trail1/seq_000'

    vsf2punyo_trans_lst = []
    for step_idx in range(0, total_frame):
        ee2base_H = np.load(f'{sim_out_dir}/ee2base_H_{step_idx:05d}.npy')

        punyo_rot_mat = np.eye(4)
        punyo_rot_mat[:3, :3] = PC_ROTATION_MATRIX
        trans = ee2base_H @ kinova_punyo2ee_np @ punyo_rot_mat

        vsf2punyo_trans_lst.append(np.linalg.inv(trans))
    
    vsf_pcd.transform(vsf2punyo_trans_lst[0])
    punyo_rest = force_estimator.undeformed_points.numpy()

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

    init_vsf_point = punyo_rest[move_idx_lst, :]
    free_init_pts = punyo_rest[boundary_mask == 0, :]
    
    colors = np.zeros(punyo_rest.shape)
    colors[move_idx_lst, :] = [1, 0, 0]
    colors[boundary_mask == 1, :] += [0, 0, 1]
    colors[boundary_mask == 0, :] = [0, 1, 0]

    sorted_free_idx = np.sort(np.where(boundary_mask == 0)[0])

    print('free_init_pts shape:', free_init_pts.shape)

    punyo_deformed_pcd = o3d.geometry.PointCloud()
    punyo_deformed_pcd.points = o3d.utility.Vector3dVector(punyo_rest)
    punyo_deformed_pcd.colors = o3d.utility.Vector3dVector(colors)

    punyo_deformed_mesh = o3d.geometry.TriangleMesh()
    punyo_deformed_mesh.vertices = o3d.utility.Vector3dVector(punyo_rest)
    punyo_deformed_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    punyo_deformed_mesh.compute_vertex_normals()

    base_triangles = np.load('data/mesh_base_triangles.npy')
    punyo_trimesh = Trimesh(vertices=punyo_rest, faces=np.vstack([triangles, base_triangles]))

    contact_A = np.zeros((vsf_rest_pts.shape[0], np.sum(boundary_mask == 0)))

    vsf_colors = np.tile([0.0, 1.0, 0.0], (vsf_rest_pts.shape[0], 1))

    contact_bool = np.zeros(vsf_rest_pts.shape[0], dtype=bool)
    contact_mesh_pts = np.zeros(vsf_rest_pts.shape)
    contact_tri_idx = np.zeros(vsf_rest_pts.shape[0], dtype=int)
    vsf_stiffness = 10000.0
    def update_pts(trans_matrix):
        global contact_A, contact_bool, colors, vsf_colors, contact_tri_idx
        curr_vsf_pts = vsf_rest_pts @ trans_matrix[:3, :3].T + trans_matrix[:3, 3]
        vsf_pcd.points = o3d.utility.Vector3dVector(curr_vsf_pts)
        # ext_delta_pts = np.zeros_like(punyo_rest)
        # ext_delta_pts[move_idx_lst] = current_move_vector

        # curr_vsf_point = init_vsf_point + current_move_vector

        # vsf_pcd.points = o3d.utility.Vector3dVector(curr_vsf_point)

        start_time = time.time()
        curr_contact_bool = punyo_trimesh.contains(curr_vsf_pts)
        print('contact check time:', time.time() - start_time)

        vsf_colors[curr_contact_bool, :] = [1.0, 0.0, 0.0]
        vsf_colors[~curr_contact_bool, :] = [0.0, 1.0, 0.0]
        vsf_pcd.colors = o3d.utility.Vector3dVector(vsf_colors)

        # # Detect if the point is inside the mesh
        # if not contact_bool:
        #     print('INFO: Point is outside the mesh')
        #     vsf_pcd.paint_uniform_color([1, 0, 0])
        # else:
        #     print('INFO: Point is inside the mesh')
        #     vsf_pcd.paint_uniform_color([0, 1, 0])
        
        new_contact_bool = np.logical_and(curr_contact_bool, ~contact_bool)
        remove_contact_bool = np.logical_and(~curr_contact_bool, contact_bool)
        if np.any(new_contact_bool):
            new_contact_pts = curr_vsf_pts[new_contact_bool, :]
            contact_mesh_pts[new_contact_bool, :] = new_contact_pts

            # find closest triangle
            closest_tri, dist, closest_tri_idx = trimesh.proximity.closest_point(punyo_trimesh, new_contact_pts)
            triangles_nearest = punyo_trimesh.triangles[closest_tri_idx]
            contact_tri_idx[new_contact_bool] = closest_tri_idx
            barycentric = trimesh.triangles.points_to_barycentric(triangles_nearest, new_contact_pts)

            closest_vertex_idx = punyo_trimesh.faces[closest_tri_idx]
            colors[closest_vertex_idx, :] = [0.7, 0.7, 1.0]

            row_idx = np.repeat(np.arange(new_contact_pts.shape[0]), 3)
            col_idx = closest_vertex_idx.flatten()

            new_contact_A = np.zeros((new_contact_pts.shape[0], punyo_rest.shape[0]))
            new_contact_A[row_idx, col_idx] = barycentric.flatten()
            new_contact_A = new_contact_A[:, boundary_mask == 0]
            
            contact_A[new_contact_bool, :] = new_contact_A
        if np.any(remove_contact_bool):
            print('remove contact:', np.where(remove_contact_bool)[0])
            contact_A[remove_contact_bool, :] = 0.0
            remove_vertex_idx = punyo_trimesh.faces[contact_tri_idx[remove_contact_bool]]
            print('remove_vertex_idx:', remove_vertex_idx.flatten())
            colors[remove_vertex_idx.flatten(), :] = [0.0, 1.0, 0.0]

        # elif contact_bool and not curr_contact_bool:
        #     colors = init_color.copy()

        # Update contact state
        contact_bool = curr_contact_bool

        punyo_deformed_pcd.colors = o3d.utility.Vector3dVector(colors)

        if np.any(curr_contact_bool):
            expand_contact_A = np.kron(contact_A[contact_bool, :], np.eye(3))
            print('expand_contact_A shape:', expand_contact_A.shape)
            K_ff_prime = K_ff + vsf_stiffness * expand_contact_A.T @ expand_contact_A
            delta_vsf_pts = curr_vsf_pts[contact_bool, :] - contact_mesh_pts[contact_bool, :]
            print('delta_vsf_pts:', delta_vsf_pts)
            contact_effect = vsf_stiffness * expand_contact_A.T @ delta_vsf_pts.flatten()

            u_f = np.linalg.solve(K_ff_prime, contact_effect)
            print('u_f len:', np.linalg.norm(u_f))

            free_pts = free_init_pts + u_f.reshape(-1, 3)

            curr_pts = np.zeros(punyo_rest.shape)
            curr_pts[boundary_mask == 1] = punyo_rest[boundary_mask == 1]
            curr_pts[sorted_free_idx, :] = free_pts

            punyo_deformed_pcd.points = o3d.utility.Vector3dVector(curr_pts)
            punyo_deformed_mesh.vertices = o3d.utility.Vector3dVector(curr_pts)
            punyo_deformed_mesh.compute_vertex_normals()
    
    step_idx = 0
    def create_update_transform(delta_idx):

        def update_move_dist(vis):
            global step_idx
            if step_idx + delta_idx < 0 or step_idx + delta_idx >= total_frame:
                return
            else:
                step_idx += delta_idx

            print('current step:', step_idx)
            update_pts(vsf2punyo_trans_lst[step_idx])
            vis.update_geometry(punyo_deformed_pcd)
            vis.update_geometry(punyo_deformed_mesh)
            vis.update_geometry(vsf_pcd)
            vis.poll_events()
            vis.update_renderer()
        return update_move_dist

    key_to_callback = {}
    key_to_callback[ord('A')] = create_update_transform(+1)
    key_to_callback[ord('S')] = create_update_transform(-1)

    # o3d.visualization.draw_geometries([punyo_deformed_pcd])
    o3d.visualization.draw_geometries_with_key_callbacks([punyo_deformed_pcd, punyo_deformed_mesh, vsf_pcd], key_to_callback)