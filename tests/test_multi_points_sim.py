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

    rest_pts = np.load('/home/motion/plant-model/out_data/obj_asset_dataset/white_adidas_superstar_angle00/rest_pts.npy')
    vsf_pcd = o3d.geometry.PointCloud()
    vsf_pcd.points = o3d.utility.Vector3dVector(rest_pts)

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

    punyo2vsf_trans_lst = []
    # for step_idx in range(0, total_frame, 10):
    for step_idx in [total_frame-1]:
        ee2base_H = np.load(f'{sim_out_dir}/ee2base_H_{step_idx:05d}.npy')

        punyo_rest = force_estimator.undeformed_points.numpy()
        punyo_rest = punyo_rest @ PC_ROTATION_MATRIX.T
        punyo_pcd = o3d.geometry.PointCloud()
        punyo_pcd.points = o3d.utility.Vector3dVector(punyo_rest)
        punyo_pcd.transform(kinova_punyo2ee_np)
        punyo_pcd.transform(ee2base_H)

        punyo_rot_mat = np.eye(4)
        punyo_rot_mat[:3, :3] = PC_ROTATION_MATRIX
        trans = ee2base_H @ kinova_punyo2ee_np @ punyo_rot_mat

        new_punyo_pcd = o3d.geometry.PointCloud()
        new_punyo_pcd.points = o3d.utility.Vector3dVector(force_estimator.undeformed_points.numpy())
        new_punyo_pcd.transform(trans)

        punyo_curr_pcd.paint_uniform_color([1.0, 0.5, 0.0])
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([punyo_pcd, coord_frame, punyo_curr_pcd, new_punyo_pcd])

    input()

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
    punyo_pcd.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries_with_editing([punyo_pcd])

    sorted_free_idx = np.sort(np.where(boundary_mask == 0)[0])

    print('free_init_pts shape:', free_init_pts.shape)

    punyo_deformed_pcd = o3d.geometry.PointCloud()
    punyo_deformed_pcd.points = o3d.utility.Vector3dVector(punyo_rest)
    punyo_deformed_pcd.colors = o3d.utility.Vector3dVector(colors)


    init_point = punyo_rest[move_idx_lst, :]
    init_color = colors.copy()

    vsf_pcd = o3d.geometry.PointCloud()
    vsf_pcd.points = o3d.utility.Vector3dVector(init_point)

    punyo_deformed_mesh = o3d.geometry.TriangleMesh()
    punyo_deformed_mesh.vertices = o3d.utility.Vector3dVector(punyo_rest)
    punyo_deformed_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    punyo_deformed_mesh.compute_vertex_normals()

    base_triangles = np.load('data/mesh_base_triangles.npy')
    punyo_trimesh = Trimesh(vertices=punyo_rest, faces=np.vstack([triangles, base_triangles]))

    contact_A = None
    contact_bool = False
    init_contact_point = None
    vsf_stiffness = 10000.0
    def update_pts(current_move_vector):
        global contact_A, contact_bool, colors, init_contact_point
        ext_delta_pts = np.zeros_like(punyo_rest)
        ext_delta_pts[move_idx_lst] = current_move_vector

        curr_vsf_point = init_vsf_point + current_move_vector

        vsf_pcd.points = o3d.utility.Vector3dVector(curr_vsf_point)

        curr_contact_bool = punyo_trimesh.contains(curr_vsf_point)
        # Detect if the point is inside the mesh
        if not contact_bool:
            print('INFO: Point is outside the mesh')
            vsf_pcd.paint_uniform_color([1, 0, 0])
        else:
            print('INFO: Point is inside the mesh')
            vsf_pcd.paint_uniform_color([0, 1, 0])
        
        if not contact_bool and curr_contact_bool:

            init_contact_point = curr_vsf_point.copy()

            # find closest triangle
            closest_tri, dist, closest_tri_idx = trimesh.proximity.closest_point(punyo_trimesh, curr_vsf_point)
            triangles_nearest = punyo_trimesh.triangles[closest_tri_idx]
            barycentric = trimesh.triangles.points_to_barycentric(triangles_nearest, curr_vsf_point)

            closest_vertex_idx = punyo_trimesh.faces[closest_tri_idx]
            colors[closest_vertex_idx, :] = [0.7, 0.7, 1.0]

            contact_A = np.zeros(punyo_rest.shape[0])
            contact_A[closest_vertex_idx] = barycentric
            contact_A = contact_A[boundary_mask == 0]
            contact_A = np.kron(contact_A.reshape(1, -1), np.eye(3))
            print('contact_A shape:', contact_A.shape)

        elif contact_bool and not curr_contact_bool:
            colors = init_color.copy()

        # Update contact state
        contact_bool = curr_contact_bool

        punyo_deformed_pcd.colors = o3d.utility.Vector3dVector(colors)

        if curr_contact_bool:
            K_ff_prime = K_ff + vsf_stiffness * contact_A.T @ contact_A
            delta_vsf_pts = curr_vsf_point - init_contact_point
            print('delta_vsf_pts:', delta_vsf_pts)
            contact_effect = vsf_stiffness * contact_A.T @ delta_vsf_pts.flatten()

            u_f = np.linalg.solve(K_ff_prime, contact_effect)
            print('u_f len:', np.linalg.norm(u_f))

            free_pts = free_init_pts + u_f.reshape(-1, 3)

            curr_pts = np.zeros(punyo_rest.shape)
            curr_pts[boundary_mask == 1] = punyo_rest[boundary_mask == 1]
            curr_pts[sorted_free_idx, :] = free_pts

            punyo_deformed_pcd.points = o3d.utility.Vector3dVector(curr_pts)
            punyo_deformed_mesh.vertices = o3d.utility.Vector3dVector(curr_pts)
            punyo_deformed_mesh.compute_vertex_normals()
    
    current_move_vector = np.zeros(3)
    def create_update_move_vector(delta_vector):

        def update_move_dist(vis):
            global current_move_vector
            current_move_vector += delta_vector
            print('current_move_vector:', current_move_vector)
            update_pts(current_move_vector)

            vis.update_geometry(punyo_deformed_pcd)
            vis.update_geometry(punyo_deformed_mesh)
            vis.update_geometry(vsf_pcd)
            vis.poll_events()
            vis.update_renderer()
        return update_move_dist

    key_to_callback = {}
    key_to_callback[ord('A')] = create_update_move_vector(np.array([0.001, 0.0, 0.0]))
    key_to_callback[ord('D')] = create_update_move_vector(np.array([-0.001, 0.0, 0.0]))
    key_to_callback[ord('W')] = create_update_move_vector(np.array([0.0, 0.001, 0.0]))
    key_to_callback[ord('S')] = create_update_move_vector(np.array([0.0, -0.001, 0.0]))
    key_to_callback[ord('P')] = create_update_move_vector(np.array([0.0, 0.0, 0.001]))
    key_to_callback[ord('L')] = create_update_move_vector(np.array([0.0, 0.0, -0.001]))

    # o3d.visualization.draw_geometries([punyo_deformed_pcd])
    o3d.visualization.draw_geometries_with_key_callbacks([punyo_deformed_pcd, punyo_deformed_mesh, vsf_pcd], key_to_callback)