""" Generate a mesh corresponding to the bubble without any forces applied to it.

Start from a flat mesh, and project it up along the camera direction to the bubble.
Then "fix" the XY coordinates by
"""
import numpy as np
import scipy.interpolate
import pygmsh
import meshio

from punyo_force_estimation.utils import *
from punyo_force_estimation.ellipse_mesh import x_bounds, y_bounds, axis_aligned_ellipse
from punyo_force_estimation.guess_force import predict_internal_force
from punyo_force_estimation.extract_bad_points import get_point_errors

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

def gen_ref_data(n_points, working_dir, out_dir, reference_frames=[0,1,2,3,4], still_frames=list(range(0, 30, 1))):
    mesh = gen_flat_mesh(n_points)
    os.makedirs(out_dir, exist_ok=True)
    mesh.write(f"{out_dir}/flat.vtk")
    points, triangles, boundary, boundary_mask = unpack_mesh(f"{out_dir}/flat.vtk")

    print("Mesh stats:", len(points), "points,", len(triangles), "triangles,", len(boundary), "boundary nodes")

    reference_rgbs, reference_pcds, reference_pressures = load_frames(working_dir, reference_frames)
    deformation_estimator, undeformed_mesh_points, reference_pressure = get_reference_mesh(
                reference_rgbs, reference_pcds, reference_pressures,
                points, triangles, boundary, smooth_iters=10
            )

    mesh.points = undeformed_mesh_points
    mesh.write(f"{out_dir}/domed_unequal.vtk")

    undeformed_mesh_points = spread_points(undeformed_mesh_points, points[:, :2])
    for n in boundary:
        undeformed_mesh_points[n] = points[n]
    mesh.points = undeformed_mesh_points
    mesh.write(f"{out_dir}/equalized.vtk")

    total_pressure_delta = reference_pressure - ATMOSPHERIC_PRESSURE
    print("Pressure delta: ", total_pressure_delta)

    forces, internal_forces, all_tri_forces = predict_internal_force(undeformed_mesh_points, triangles, boundary, total_pressure_delta)

    print("saving forces...")
    np.save(f"{out_dir}/triangle_force.npy", all_tri_forces)


    deformation_estimator, undeformed_mesh_points, reference_pressure = get_reference_mesh(
                reference_rgbs, reference_pcds, reference_pressures,
                undeformed_mesh_points, triangles, boundary, smooth_iters=0
            )
    all_deviations = get_point_errors(working_dir, undeformed_mesh_points, deformation_estimator, still_frames)

    point_zs_0 = np.mean(all_deviations, axis=0)
    point_zs_1 = np.std(all_deviations, axis=0)

    print("saving error data")
    point_zs_0.dump(f"{out_dir}/point_err_mean.npy")
    point_zs_1.dump(f"{out_dir}/point_stdevs.npy")

if __name__ == "__main__":
    import os
    import argparse
    import pathlib

    parser = argparse.ArgumentParser(description="generate initial mesh for no force config")
    parser.add_argument('working_dir', type=pathlib.Path, help="Folder containing data (bunch of numpy arrays of rgb, pcd, pressure data of the bubble not in contact with anything)")
    parser.add_argument('-n', '--num_points', type=int, default=64, help="Number of points on the perimeter of the mesh")
    parser.add_argument('-o', '--out_dir', type=pathlib.Path, default="ref_data", help="Folder to dump output files in. Will be created recursively if it does not exist.")
    args = parser.parse_args()
    n_points = args.num_points
    working_dir = os.path.expanduser(args.working_dir)
    out_dir = os.path.expanduser(args.out_dir)

    gen_ref_data(n_points, working_dir, out_dir)
