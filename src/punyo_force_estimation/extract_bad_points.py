import os

import numpy as np
import matplotlib.pyplot as plt

from .force_module.deformation_estimator import DeformationEstimation
from .utils import *

def get_point_errors(working_dir, undeformed_mesh_points, deformation_estimator, still_frames):
    all_deviations = np.zeros([len(still_frames), len(undeformed_mesh_points)])

    for i, frame in enumerate(still_frames):
        print("loading frame", i)
        pressure, pcd, rgb = load_data(working_dir, frame)
        raw_deformed_points, data = deformation_estimator.estimate(rgb, pcd)
        raw_deformed_points = raw_deformed_points @ PC_ROTATION_MATRIX

        observed_flag = data[3]
        deviations = np.linalg.norm(raw_deformed_points - undeformed_mesh_points, axis=1)
        all_deviations[i, :] = deviations + observed_flag

    return all_deviations


if __name__ == "__main__":
    working_dir = os.path.expanduser("~/Documents/punyo_unit_test/data_1/expand")   # poke with robot left
    #reference_frames = [0, 1, 2, 3, 4]
    reference_frames = list(range(0, 30, 6))
    still_frames = list(range(0, 80))

    points, triangles, boundary, boundary_mask = unpack_mesh(f"base_meshes/4.vtk")

    reference_rgbs, reference_pcds, reference_pressures = load_frames(working_dir, reference_frames)
    deformation_estimator, undeformed_mesh_points, reference_pressure = get_reference_mesh(
                reference_rgbs, reference_pcds, reference_pressures,
                points, triangles, boundary, smooth_iters=10
            )
    all_deviations = get_point_errors(working_dir, undeformed_mesh_points, deformation_estimator, still_frames)

    point_xs = undeformed_mesh_points[:, 0]
    point_ys = undeformed_mesh_points[:, 1]
    point_zs_0 = np.mean(all_deviations, axis=0)
    point_zs_1 = np.std(all_deviations, axis=0)

    print("saving data")
    point_zs_0.dump("point_err_mean.npy")
    point_zs_1.dump("point_stdevs.npy")

    fig = plt.figure()
    a0, a1 = fig.subplots(2)
    s0 = a0.tricontourf(point_xs, point_ys, point_zs_0)
    plt.colorbar(s0, ax=a0)
    s1 = a1.tricontourf(point_xs, point_ys, point_zs_1)
    plt.colorbar(s1, ax=a1)
    a0.set_aspect('equal')
    a1.set_aspect('equal')
    a0.invert_yaxis()
    a1.invert_yaxis()
    plt.show()
