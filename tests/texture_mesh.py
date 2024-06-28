import os
import json
import open3d as o3d
import numpy as np
import imageio
import xatlas
import torch
import nvdiffrast.torch as dr
from punyo_force_estimation.force_module.force_from_punyo import ForceFromPunyo
from punyo_force_estimation.utils import load_frames, load_data, unpack_mesh
import util

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

def UV_mapping(points, triangles):
    atlas = xatlas.Atlas()
    atlas.add_mesh(points, triangles)
    chart_options = xatlas.ChartOptions()
    chart_options.max_iterations = 0 # disable merge_chart for faster unwrap...
    pack_options = xatlas.PackOptions()
    # pack_options.blockAlign = True
    # pack_options.bruteForce = False
    # pack_options.create_image = True
    atlas.generate(chart_options=chart_options, pack_options=pack_options)
    # show_image(atlas.chart_image)
    vmapping, indices, uvs = atlas[0] # [N], [M, 3], [N, 2]
    return indices, uvs

def transform_pos(mtx, pos):
    t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx, np.ndarray) else mtx
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
    return torch.matmul(posw, t_mtx.t())[None, ...]

def render(glctx, mtx, pos, pos_idx, uv, uv_idx, tex, resolution, enable_mip, max_mip_level):
    pos_clip = transform_pos(mtx, pos)
    rast_out, rast_out_db = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[resolution, resolution])

    if enable_mip:
        texc, texd = dr.interpolate(uv[None, ...], rast_out, uv_idx, rast_db=rast_out_db, diff_attrs='all')
        color = dr.texture(tex[None, ...], texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=max_mip_level)
    else:
        texc, _ = dr.interpolate(uv[None, ...], rast_out, uv_idx)
        color = dr.texture(tex[None, ...], texc, filter_mode='linear')

    color = color * torch.clamp(rast_out[..., -1:], 0, 1) # Mask out background.
    return color

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
    # downsample pointclouds
    pointclouds = [pointclouds[i][::10, :] for i in range(total_frame)]

    reference_frames = [0, 1, 2, 3, 4]
    reference_rgbs = [rgbs[i] for i in reference_frames]
    reference_pcds = [pointclouds[i] for i in reference_frames]
    reference_pressures = [pressures[i] for i in reference_frames]

    points, triangles, boundary, boundary_mask = unpack_mesh(f"{ref_dir}/equalized.vtk")
    force_estimator = ForceFromPunyo(reference_rgbs, reference_pcds, reference_pressures, points, triangles, boundary, 
                                     rest_internal_force=None, precompile=False, verbose=True)
    
    rest_pts = force_estimator.undeformed_points.numpy()
    indices, uvs = UV_mapping(rest_pts, triangles)

    f = 385.263 / 640

    project_matrix = util.projection(f, n=0.01, f=500.0)
    mvp = project_matrix
    tex_opt = torch.full((600,600,3), 0.2, device='cuda', requires_grad=True)
    vtx_pos = torch.from_numpy(rest_pts).float().cuda()
    pos_idx = torch.from_numpy(triangles).int().cuda()
    uvs = torch.from_numpy(uvs).float().cuda()
    uv_idx = torch.from_numpy(indices).int().cuda()

    vtx_pos[:,2] *= -1


    glctx = dr.RasterizeCudaContext()

    lr_base = 1e-2
    lr_ramp = 0.1
    max_iter = 1000

    optimizer = torch.optim.Adam([tex_opt], lr=lr_base)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: lr_ramp**(float(x)/float(max_iter)))

    for i in range(max_iter):
        optimizer.zero_grad()
        color = render(glctx, mvp, vtx_pos, pos_idx, uvs, uv_idx, tex_opt, 600, False, 0)
        color = color[0]
        # visualize color as image and show it
        print((color != 0).sum())
        o3d.visualization.draw_geometries([o3d.geometry.Image((color.cpu().detach().numpy()*255).astype(np.uint8))])
        import sys
        sys.exit()
        loss = -torch.mean(color)
        loss.backward()
        optimizer.step()
        scheduler.step()

        
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