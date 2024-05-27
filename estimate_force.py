import os
import json
import open3d as o3d
import numpy as np
import imageio
from src.punyo_force_estimation.force_module.force_from_punyo import ForceFromPunyo
from src.punyo_force_estimation.utils import load_frames, load_data, unpack_mesh

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
    import argparse
    import pathlib

    parser = argparse.ArgumentParser(description="Run the force estimator on a data sequence. Save and visualize output.")
    parser.add_argument('working_dir', type=pathlib.Path, help="Folder containing data (bunch of numpy arrays of rgb, pcd, pressure data of the bubble not in contact with anything)")
    parser.add_argument('--ref_dir', type=pathlib.Path, default="data/ref_data", help="Folder containing reference mesh")

    args = parser.parse_args()
    working_dir = args.working_dir
    ref_dir = args.ref_dir

    total_frame = 148
    rgbs = [imageio.imread(f"{working_dir}/raw/punyo_color_{i}.png") for i in range(total_frame)]
    depths = [imageio.imread(f"{working_dir}/raw/punyo_depth_{i}.png") for i in range(total_frame)]
    pressure_scale = 100
    pressures = np.load(f"{working_dir}/raw/pressure.npy") * pressure_scale
    rgbs_o3d = [o3d.geometry.Image(rgbs[i]) for i in range(total_frame)]
    depths_o3d = [o3d.geometry.Image(depths[i]) for i in range(total_frame)]
    pointclouds = [rgbd_to_pc(rgbs_o3d[i], depths_o3d[i])[0] for i in range(total_frame)]

    reference_frames = [0, 1, 2, 3, 4]
    reference_rgbs = [rgbs[i] for i in reference_frames]
    reference_pcds = [pointclouds[i] for i in reference_frames]
    reference_pressures = [pressures[i] for i in reference_frames]

    points, triangles, boundary, boundary_mask = unpack_mesh(f"{ref_dir}/equalized.vtk")
    force_estimator = ForceFromPunyo(reference_rgbs, reference_pcds, reference_pressures, points, triangles, boundary, 
                                     rest_internal_force=None, precompile=True, verbose=True)
    
    START_FRAME = 5
    END_FRAME = total_frame
    os.makedirs(f"{working_dir}/result", exist_ok=True)
    for i in range(START_FRAME, END_FRAME):
        pressure, pcd, rgb = pressures[i], pointclouds[i], rgbs[i]
        force_estimator.update(rgb, pcd, pressure)

        contact_forces = force_estimator.observed_force
        displaced_points = force_estimator.current_points

        np.save(f"{working_dir}/result/force_{i}.npy", contact_forces)
        np.save(f"{working_dir}/result/points_{i}.npy", displaced_points)

        print(f"Frame {i} done.")