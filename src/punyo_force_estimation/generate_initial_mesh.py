import numpy as np
import scipy.interpolate
import pygmsh
import meshio

from .utils import *
from .ellipse_mesh import x_bounds, y_bounds, axis_aligned_ellipse
from .guess_force import predict_internal_force
from .extract_bad_points import get_point_errors

def spread_points(points, ideal_xy):
    return scipy.interpolate.griddata(
            points[:, :2],
            points,
            ideal_xy
    )

def gen_flat_mesh(n=64):
    with pygmsh.geo.Geometry() as geom:
        outer_ellipse = axis_aligned_ellipse(x_bounds, y_bounds, n_ellipse_points=n)
        ellipse = np.hstack([outer_ellipse, np.ones([len(outer_ellipse), 1]) * mesh_plane_z])
        geom.add_polygon(ellipse, mesh_size=0.05)
        mesh = geom.generate_mesh()
    return mesh

ATMOSPHERIC_PRESSURE = 101325   # Pascals
# dynamic load
# dat = np.load("inflating_sequence/pressure.0.npy")
# ATMOSPHERIC_PRESSURE = dat[1] * 100


if __name__ == "__main__":
    import os
    import argparse
    import pathlib

    parser = argparse.ArgumentParser(description="generate initial mesh for no force config")
    parser.add_argument('-n', '--num_points', type=int, default=64)
    parser.add_argument('-p', '--working_dir', type=pathlib.Path, default="~/Documents/punyo_unit_test/data_0/expand")
    args = parser.parse_args()
    n_points = args.num_points
    working_dir = os.path.expanduser(args.working_dir)

    reference_frames = [0, 1, 2, 3, 4]
    still_frames = list(range(0, 80, 1))

    mesh = gen_flat_mesh(n_points)
    os.makedirs("ref_data", exist_ok=True)
    mesh.write("ref_data/flat.vtk")
    points, triangles, boundary, boundary_mask = unpack_mesh("ref_data/flat.vtk")

    print("Mesh stats:", len(points), "points,", len(triangles), "triangles,", len(boundary), "boundary nodes")

    reference_rgbs, reference_pcds, reference_pressures = load_frames(working_dir, reference_frames)
    deformation_estimator, undeformed_mesh_points, reference_pressure = get_reference_mesh(
                reference_rgbs, reference_pcds, reference_pressures,
                points, triangles, boundary, smooth_iters=10
            )

    mesh.points = undeformed_mesh_points
    mesh.write("ref_data/domed_unequal.vtk")

    undeformed_mesh_points = spread_points(undeformed_mesh_points, points[:, :2])
    for n in boundary:
        undeformed_mesh_points[n] = points[n]
    mesh.points = undeformed_mesh_points
    mesh.write("ref_data/equalized.vtk")

    total_pressure_delta = reference_pressure - ATMOSPHERIC_PRESSURE
    print("Pressure delta: ", total_pressure_delta)

    forces, internal_forces, all_tri_forces = predict_internal_force(undeformed_mesh_points, triangles, boundary, total_pressure_delta)

    print("saving forces...")
    np.save("ref_data/triangle_force.npy", all_tri_forces)


    deformation_estimator, undeformed_mesh_points, reference_pressure = get_reference_mesh(
                reference_rgbs, reference_pcds, reference_pressures,
                undeformed_mesh_points, triangles, boundary, smooth_iters=0
            )
    all_deviations = get_point_errors(working_dir, undeformed_mesh_points, deformation_estimator, still_frames)

    point_zs_0 = np.mean(all_deviations, axis=0)
    point_zs_1 = np.std(all_deviations, axis=0)

    print("saving error data")
    point_zs_0.dump("ref_data/point_err_mean.npy")
    point_zs_1.dump("ref_data/point_stdevs.npy")
    deformation_estimator, undeformed_mesh_points, reference_pressure = get_reference_mesh(
                reference_rgbs, reference_pcds, reference_pressures,
                undeformed_mesh_points, triangles, boundary, smooth_iters=0
            )
