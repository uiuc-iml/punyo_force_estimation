import json

import cv2
import numpy as np
import scipy as sp
import meshio
import meshio
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()
import open3d as o3d

from .pipeline import *

from .force_module.deformation_estimator import DeformationEstimation
from .force_module.force_predictor import CoupledForcePrediction
from .force_module.material_model import LinearTriangleModel, LinearPlaneStressModel, CorotatedPlaneStressModel, triangle_optimal_rotation
from .force_module.force_from_punyo import ForceFromPunyo

from .vis_utils.cv_player import FramePlayer
from .vis_utils.cv_utils import visualize_flow_arrows
from .utils import *

dat = np.load("inflating_sequence/pressure.0.npy")
atmospheric_pressure = dat[1] * 100

import os

method = "L1"
if method == "L1":
    # from optimization. For L1 norm.
    # FINAL OPT RUN
    k_force, k_disp, E, contact_threshold = (0.332213216, 537591.774, 855.995785, 2000)

    # random testing
    k_force, k_disp, E, contact_threshold = [2.741e-01, 2.212e+05, 8.333e+03, 2.00000000e+03]
    #k_force, k_disp, E, contact_threshold = (0.332213216, 537591.774, 855.995785, 2000)
    #k_force, k_disp, E, contact_threshold = [3.46097857e-01, 5.33857716e+05, 7.36718848e+02, 2.00000000e+03]
    #k_force, k_disp, E, contact_threshold = [4.24e-01, 2.718e+05, 4.127e+03, 2.00000000e+03]

if method == "L2":
    # from optimization. For L2 norm.
    k_force, k_disp, E, contact_threshold = [3.29251786e-03, 6.49870548e+06, 6.91101237e+02, 2.00000000e+03]

#working_dir = os.path.expanduser("~/Documents/punyo_unit_test/data_0/expand")   # poke with robot center
#working_dir = os.path.expanduser("~/Documents/punyo_unit_test/data_1/expand")   # poke with robot left
#working_dir = os.path.expanduser("~/Documents/punyo_unit_test/data_2/expand")   # poke with robot right
working_dir = os.path.expanduser("~/Documents/punyo_unit_test/data_4/expand")   # poke with robot and slide indenter left
#working_dir = os.path.expanduser("~/Documents/punyo_unit_test/data_5/expand")   # poke with robot offset side
#working_dir = os.path.expanduser("~/Documents/punyo_unit_test/data_6/expand")   # poke with robot center sharp
#working_dir = os.path.expanduser("~/Documents/punyo_unit_test/fun/expand")   # poke with non convex
#working_dir = os.path.expanduser("~/Documents/punyo_unit_test/push_harder2/expand")   # poke very deep
#working_dir = os.path.expanduser("~/remote/punyo_collected_data/poker.STL/16/expand")
#working_dir = os.path.expanduser("~/remote/punyo_collected_data/poker_twins.STL/28/expand")
#working_dir = os.path.expanduser("~/Documents/punyo_video_data/poker.STL/0/expand")
reference_frames = [0, 1, 2, 3, 4]
#reference_frames = list(range(0, 30, 6))
#idx_offset = 30
#idx_offset = 50
idx_offset = 65
#idx_offset = 80
#idx_offset = 90
#n_frames = 286 - idx_offset
n_frames = 255 - idx_offset
robot_dat = json.load(open(f"{working_dir}/../data.json"))

# TODO HACK: semi-inflated mesh as the start. to have less warping in the mesh
points, triangles, boundary, boundary_mask = unpack_mesh("ref_data/equalized.vtk")
#points, triangles, boundary, boundary_mask = unpack_mesh(f"base_meshes/4.vtk")
#points, triangles, boundary, boundary_mask = unpack_mesh(f"base_meshes/3.vtk")
#points, triangles, boundary, boundary_mask = unpack_mesh(f"base_meshes/0.vtk")

reference_rgbs, reference_pcds, reference_pressures = load_frames(working_dir, reference_frames)

rest_internal_force = np.load("ref_data/triangle_force.npy")
rest_internal_force_total = np.zeros_like(points)
for tri_idx in range(len(rest_internal_force)):
    i, j, k = triangles[tri_idx]
    tri_force = rest_internal_force[tri_idx]
    rest_internal_force_total[i] += tri_force[0, :]
    rest_internal_force_total[j] += tri_force[1, :]
    rest_internal_force_total[k] += tri_force[2, :]

punyo_model = ForceFromPunyo(reference_rgbs, reference_pcds, reference_pressures, points, triangles, boundary, rest_internal_force,
                force_penalty=k_force, displacement_penalty=k_disp, optimization_params={'nu': 0.5, 'E': E}, precompile=True, verbose=True, method=method)
                #force_penalty=k_force, displacement_penalty=k_disp, optimization_params={'nu': 0.5, 'E': E}, precompile=False, verbose=True, method=method)
undeformed_points = punyo_model.undeformed_points

undeformed_pcd = pointcloud(undeformed_points)
o3d_mesh = o3d.geometry.TriangleMesh()
o3d_mesh.vertices = o3d.utility.Vector3dVector(undeformed_points)
o3d_mesh.triangles = o3d.utility.Vector3iVector(triangles)
o3d_mesh.compute_vertex_normals()


def get_frame(idx):

    frame = cached_frames[idx]
    if frame is not None:
        return frame
    if idx > 0:
        prev_frame = get_frame(idx-1)
        punyo_model.current_pressure = prev_frame['pressure']
        punyo_model.current_points = prev_frame['corrected_points']
    else:
        punyo_model.current_pressure = punyo_model.reference_pressure
        punyo_model.current_points = punyo_model.undeformed_points

    raw_data = get_data(idx)
    print(f"Computing frame {idx}")
    t0 = time.time()
    punyo_model.update(raw_data['rgb'], raw_data['pcd'], raw_data['raw_pressure'])
    t1 = time.time()

    flow_vector, flow_vector2, mesh_flow_2d, _obs = punyo_model.flow_data

    print(np.sum(punyo_model.observed_force * (1 - punyo_model.boundary_mask).reshape(-1, 1), axis=0))

    #max_flow = np.max(np.linalg.norm(flow_vector2, axis=2))
    #flow_mask = np.linalg.norm(flow_vector2, axis=2) > (max_flow/2)
    flow_mask = np.linalg.norm(flow_vector2, axis=2) > 6
    flow_mask_img = np.array(flow_mask, dtype=np.uint8) * 255
    frame = {
            "pressure": punyo_model.current_pressure,
            "rgb": raw_data['rgb'],
            "flow_mask_img": flow_mask_img,
            "pressure_forces": punyo_model._pressure_forces,
            "observed_force": punyo_model.observed_force,
            "internal_force": punyo_model._f_internal,
            "deformed_points": punyo_model._deformed_points,
            "corrected_points": punyo_model.current_points,
            "flows": flow_vector2,
            "point_areas": punyo_model._areas,
            "observed": punyo_model._observed,
            "compute_time": t1-t0
    }
    if robot_dat is not None:
        robot_cfg = raw_data['robot_cfg']
        motion_inst.set_joint_config('left_limb', robot_cfg, {})
        motion_inst._loop()
        motion_inst._loop()

        punyo_transform = punyo_link.getTransform()
        full_transform = se3.mul(punyo_transform, punyo_pcd_transform)

        klampt_mesh.setVertices(punyo_model._raw_points @ PC_ROTATION_MATRIX.T)
        mesh_obj.geometry().setTriangleMesh(klampt_mesh)
        mesh_obj.setTransform(*full_transform)
        contacts = mesh_obj.geometry().contacts(poker_obj.geometry(), 0.0005, 0.0005)

        point_in_frame = PC_ROTATION_MATRIX.T @ se3.apply(
            se3.inv(punyo_pcd_transform),
            se3.apply(se3.inv(punyo_transform), POKER_TIP)
        )

        frame["robot_force"] = so3.apply(so3.inv(punyo_transform[0]), raw_data['robot_force'])
        frame["robot_cfg"] = robot_cfg
        frame["contact_tris"] = set(contacts.elems1)
        frame["timestamp"] = raw_data["timestamp"]
        frame["poker_tip"] = point_in_frame
    cached_frames[idx] = frame
    return frame

def plot_variations(fignum, idx):
    fig = plt.figure(fignum)
    plt.clf()
    point_xs = undeformed_points[:, 0]
    point_ys = undeformed_points[:, 1]

    raw_points = []
    for i in range(idx+1):
        arr = np.array(filtered_data[i]["raw_points"])
        for v in boundary:
            arr[v, :] = undeformed_points[v, :]
        raw_points.append(arr)

    # for now, z stdev.
    point_zs_avg = np.zeros(undeformed_points.shape[0])
    for i in range(idx+1):
        point_zs_avg += raw_points[i][:, 2]
    point_zs_avg /= (idx+1)

    point_zs_std = np.zeros(undeformed_points.shape[0])
    for i in range(idx+1):
        point_zs_std += (raw_points[i][:, 2] - point_zs_avg)**2
    point_zs_std = np.sqrt(point_zs_std / (idx+1))

    ax = fig.subplots(subplot_kw={"projection": "3d", "proj_type": "ortho"})
    surf = ax.plot_trisurf(point_xs, point_ys, point_zs_std, cmap=plt.cm.coolwarm)

ONCE = True
def display_frame(vis, frame, idx):
    print(f"Showing frame {idx}")
    pressure = frame["pressure"]
    rgb = frame["rgb"]
    flow_mask_img = frame["flow_mask_img"]
    pressure_forces = frame["pressure_forces"]
    observed_force = frame["observed_force"]
    deformed_points = frame["deformed_points"]
    corrected_points = frame["corrected_points"]
    point_areas = frame["point_areas"]
    flows = frame["flows"]
    observed_flag = frame["observed"]

    est_normals = estimate_tri_normals(corrected_points, triangles)

    fig = plt.figure(0)
    plt.clf()
    plt.imshow(flow_mask_img)
    contacts = np.zeros((len(corrected_points), 3))
    gt_contact = np.zeros(len(corrected_points))
    if robot_dat is not None:
        for tri_idx in frame['contact_tris']:
            for n in triangles[tri_idx]:
                gt_contact[n] = 1
                #contacts[n] = [0, 1, 0]
    #contacts[observed_flag > 0] += [0, 0, 1]
    proj_mesh = project_pointcloud(np.array(undeformed_points) @ PC_ROTATION_MATRIX.T)
    plt.scatter(proj_mesh[:, 0], proj_mesh[:, 1], marker='.', c=contacts)

    fig = plt.figure(1)
    plt.clf()
    a1, a2 = fig.subplots(2)
    im = a1.imshow(flows[:, :, 0])
    fig.colorbar(im, ax=a1)
    im = a2.imshow(flows[:, :, 1])
    fig.colorbar(im, ax=a2)

    #plot_variations(2, idx)

    # Grab all total pressures and total observed forces.
    pressure_totals = np.empty((idx+1, 3))
    external_totals = np.empty((idx+1, 3))
    if robot_dat is not None:
        times = np.empty(idx+1)
        robot_totals = np.empty((idx+1, 3))
        poker_tips = []
        poker_tip_i = []
    else:
        times = np.array(list(range(idx+1)))
    center_mask = 1-boundary_mask
    compute_time_total = 0
    for i in range(idx+1):
        pressure_force_prev = cached_frames[i]["pressure_forces"]
        observed_force_prev = cached_frames[i]["observed_force"]
        pressure_totals[i] = -np.sum(pressure_force_prev * center_mask.reshape((-1, 1)), axis=0)
        external_totals[i] = np.sum(observed_force_prev * center_mask.reshape((-1, 1)), axis=0)
        if robot_dat is not None:
            if len(poker_tips):
                poker_tip_i.append([len(poker_tips) - 1, len(poker_tips)])
            poker_tips.append(cached_frames[i]["poker_tip"])
            robot_force = cached_frames[i]["robot_force"]
            times[i] = cached_frames[i]["timestamp"]
            robot_totals[i][2] = robot_force[1] * FT_Z_SCALING
            robot_totals[i][1] = robot_force[0] * FT_Y_SCALING
            robot_totals[i][0] = robot_force[2] * FT_X_SCALING

        if i > 0:
            compute_time_total += cached_frames[i]["compute_time"]
            print(cached_frames[i]["compute_time"])

    if idx > 0:
        print("Average computation time:", compute_time_total / idx)
    fig = plt.figure(3)
    plt.clf()
    plt.plot(times, pressure_totals[:, 2], label="total pressure force z")
    #plt.plot(times, external_totals[:, 0], label="total external force x")
    #plt.plot(times, external_totals[:, 1], label="total external force y")
    plt.plot(times, external_totals[:, 2], label="total external force z")
    #if robot_dat is not None:
    #    plt.plot(times, robot_totals[:, 0], label="measured external force x")
    #    plt.plot(times, robot_totals[:, 1], label="measured external force y")
    #    plt.plot(times, robot_totals[:, 2], label="measured external force z")
    plt.legend()

    fig = plt.figure(4)
    point_xs = corrected_points[:, 0] - 0.04
    point_ys = corrected_points[:, 1]
    pressures = observed_force * ((1 - boundary_mask) / point_areas).reshape(-1, 1)
    point_zs = -np.sum(pressures * est_normals, axis=1)
    point_zs[point_zs < 0] = 0

    thresh = max(2*np.mean(point_zs), 2000)
    est_contact = point_zs > thresh

    plt.clf()
    ax = plt.gca()
    surf = ax.tricontourf(point_xs, point_ys, point_zs, cmap="Reds", levels=[0, 800, 1600, 2400, 3200, 4000, 4800, 5600])
    s2 = ax.tricontourf(point_xs, point_ys, point_zs, cmap="Blues", alpha=0.6, levels=[thresh, max(thresh+1, np.max(point_zs))])
    s1 = ax.tricontourf(point_xs, point_ys, gt_contact, cmap="Greens", alpha=0.6, levels=[1, 2])

    #ax.scatter(points[:, 0], points[:, 1], marker='+')
    #contacts_disp = np.vstack([est_contact, gt_contact, np.zeros((len(corrected_points)))]).T
    #ax.scatter(corrected_points[:, 0], corrected_points[:, 1], marker='.', c=contacts_disp)
    a2, _ = s2.legend_elements()
    #ax.legend(a2, ["Est. Contact"])
    ax.set_aspect('equal')
    #cbar = plt.colorbar(surf)
    #cbar.set_label("Contact Pressure (Pa)")
    #plt.title("Contact Patch Estimation")
    #plt.xlabel("X")
    #plt.ylabel("Y")
    ax.axis("off")
    fig.tight_layout()

    fig = plt.figure(5)
    point_zs_0 = observed_force[:, 0] * (1 - boundary_mask) / point_areas
    point_zs_1 = observed_force[:, 1] * (1 - boundary_mask) / point_areas
    point_zs_2 = observed_force[:, 2] * (1 - boundary_mask) / point_areas
    plt.clf()
    a0, a1, a2 = fig.subplots(3)
    s0 = a0.tricontourf(point_xs, point_ys, point_zs_0)
    plt.colorbar(s0, ax=a0)
    s1 = a1.tricontourf(point_xs, point_ys, point_zs_1)
    plt.colorbar(s1, ax=a1)
    s2 = a2.tricontourf(point_xs, point_ys, point_zs_2)
    plt.colorbar(s2, ax=a2)
    a0.set_aspect('equal')
    a1.set_aspect('equal')
    a2.set_aspect('equal')
    a0.invert_yaxis()
    a1.invert_yaxis()
    a2.invert_yaxis()

    plt.show()
    plt.pause(0.05)

    undeformed_pcd = pointcloud(undeformed_points)
    undeformed_pcd.paint_uniform_color([0, 0, 1]) # blue undeformed pcd

    corrected_pcd = pointcloud(corrected_points)
    corrected_pcd.paint_uniform_color([0, 1, 0]) # green corrected pcd

    deformed_pcd = pointcloud(deformed_points)

    #f_plot = observed_force# + rest_internal_force_total
    f_plot = pressure_forces
    #corrected_pcd.normals = o3d.utility.Vector3dVector(f_plot / point_areas.reshape((-1, 1)) / 1000)

    #test_vec = np.zeros_like(points)
    #tri = 183
    #deformed_pcd.normals = o3d.utility.Vector3dVector(pressure_forces / point_areas.reshape((-1, 1)) / 1000)

    deformed_pcd.paint_uniform_color([1, 0, 0]) # red deformed pcd

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(corrected_points)
    # NOTE: double side triangles
    double_triangles = np.append(triangles, triangles[:, ::-1], axis=0)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(double_triangles)
    o3d_mesh.compute_vertex_normals()
    #mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)

    from vis_utils.cv_utils import create_arrow_lst
    vectors = (f_plot / (point_areas).reshape((-1, 1))) * center_mask.reshape((-1, 1))
    vectors /= np.max(np.linalg.norm(vectors, axis=1))*70
    tmp = corrected_points - vectors
    arrows = create_arrow_lst(tmp, corrected_points, scale=1)

    vis.clear_geometries()
    vis.add_geometry(undeformed_pcd, reset_bounding_box=False)
    vis.add_geometry(corrected_pcd, reset_bounding_box=False)
    vis.add_geometry(deformed_pcd, reset_bounding_box=False)
    vis.add_geometry(o3d_mesh, reset_bounding_box=False)
    if robot_dat is not None:
        #trajectory = pointcloud(poker_tips)
        if len(poker_tip_i):
            from vis_utils.cv_utils import create_graph_lines
            trajectory = create_graph_lines(poker_tips, poker_tip_i)
            trajectory.paint_uniform_color([0, 0, 0])
            vis.add_geometry(trajectory, reset_bounding_box=False)

    #vis.add_geometry(mesh_frame, reset_bounding_box=False)
    for arrow in arrows:
        vis.add_geometry(arrow, reset_bounding_box=False)
    vis.capture_screen_image(f"out4/{idx:03d}.png", do_render=True)

    import pickle
    pickle_data = {
        "trajectory": poker_tips,
        "triangles": triangles,
        "undeformed": undeformed_points,
        "deformed": corrected_points,
        "pressures": vectors
    }
    pickle.dump(pickle_data, open("pickle.dat", 'wb'))


observed_data = [None]*n_frames
def get_data(idx):
    print(f"loading frame {idx+idx_offset}")
    if observed_data[idx]:
        return observed_data[idx]

    current_pressure, current_pcd, current_rgb = load_data(working_dir, idx+idx_offset)
    dat = {
        "raw_pressure": current_pressure,
        "rgb": current_rgb,
        "pcd": current_pcd,
    }

    if robot_dat is not None:
        timestamp, pressure, cfg, force = robot_dat[idx+idx_offset]
        dat['robot_force'] = force
        dat['robot_cfg'] = cfg
        dat['timestamp'] = timestamp

    observed_data[idx] = dat
    return dat

cached_frames = [None]*n_frames
once = True
def frame_callback(vis, prev_idx, idx):
    global once
    frame = get_frame(idx)
    if prev_idx != idx or once:
        display_frame(vis, frame, idx)
        once = False
    plt.pause(0.05)
    return frame["rgb"]

###################################################
# SIMULATION SETUP

if robot_dat is not None:
    import os
    import sys
    sys.path.append(os.path.expanduser("~/TRINA"))
    import klampt
    from klampt import WorldModel, vis
    from klampt.math import so3, se3, vectorops as vo
    from calibration.jcp_hardware_calibration_08_17_2023.calibration_common import *
    from Motion.motion import Motion

    from utils import load_data
    from force_module.force_from_punyo import ForceFromPunyo

    angle = np.radians(25)
    rot_M = np.array([
        [np.cos(angle), 0, -np.sin(angle)],
        [0, 1, 0],
        [np.sin(angle), 0, np.cos(angle)],
    ])
    punyo_pcd_transform = se3.from_homogeneous(
            [[ 0            , 1, 0            ,  0 ],
             [-np.sin(angle), 0, np.cos(angle), -0.01611 ],
             [ np.cos(angle), 0, np.sin(angle),  0.090142 ],
             [ 0            , 0, 0.           ,  1]]
        )

    def start_motion(setup_server=False):
        ALL_PARTS = ['left_limb', 'right_limb', 'left_gripper', 'right_gripper', 'head', 'torso', 'base', 'eyes']
        world = WorldModel()
        world.loadElement(os.path.expanduser("~/TRINA/Models/robots/Ebola_punyo.urdf"))
        motion_inst = Motion(mode='Kinematic', components=['left_limb', 'right_limb'], world=world,
                             ALL_PARTS=ALL_PARTS, log_fname='motion.log', feature_flag=0)

        motion_inst._startup()

#        import signal
#        def sigint_handler(signum, frame):
#            motion_inst.shutdown()
#            print("Quicktest exiting")
#            raise KeyboardInterrupt()
#        signal.signal(signal.SIGINT, sigint_handler)
        return world, motion_inst

    world, motion_inst = start_motion()
    points, triangles, boundary, boundary_mask = unpack_mesh("ref_data/equalized.vtk")
    klampt_mesh = klampt.TriangleMesh()
    klampt_mesh.setVertices(points @ rot_M.T)
    indices = list(triangles)
    for i in range(1, len(boundary) - 1):
        indices.append([0, i, i+1])
    klampt_mesh.setIndices(np.array(indices, dtype=np.int32))

    world, motion_inst = start_motion()
    punyo_link = motion_inst.robot_model.link("punyo:base_link")
    pcd_obj = world.makeRigidObject("punyo_pcd")
    pcd_obj.appearance().setColor(1, 0, 0, 1)
    mesh_obj = world.makeRigidObject("punyo_mesh")
    mesh_obj.geometry().setTriangleMesh(klampt_mesh)

    poker_obj = world.loadRigidObject(os.path.expanduser("~/TRINA/Models/objects/poker.STL"))
    #poker_obj = world.loadRigidObject(os.path.expanduser("~/TRINA/Models/objects/poker_sharp.STL"))
    poker_obj.setTransform([0, -1, 0, 1, 0, 0, 0, 0, 1], vo.add(POKER_TIP, [-0.09, 0, 0]))

# SIMULATION SETUP
###################################################

vis = o3d.visualization.Visualizer()
vis.create_window()
opt = vis.get_render_option()
opt.point_show_normal = True
vis.add_geometry(undeformed_pcd)
vis.add_geometry(o3d_mesh)
#vis.add_geometry(pointcloud(deformation_estimator.filter_pc(reference_pcd)))

player = FramePlayer("deform_est", initial_idx=0, n_frames=n_frames, frame_callback=frame_callback, o3d_vis=vis, spinrate=5)
player.show()
input("enter to exit...")
player.hide()
