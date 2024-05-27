import os
import json
import numpy as np
from src.punyo_force_estimation.force_module.force_from_punyo import ForceFromPunyo
from src.punyo_force_estimation.utils import load_frames, load_data, unpack_mesh

if __name__ == "__main__":
    import argparse
    import pathlib

    parser = argparse.ArgumentParser(description="Run the force estimator on a data sequence. Save and visualize output.")
    parser.add_argument('working_dir', type=pathlib.Path, help="Folder containing data (bunch of numpy arrays of rgb, pcd, pressure data of the bubble not in contact with anything)")
    parser.add_argument('--ref_dir', type=pathlib.Path, default="ref_data", help="Folder containing reference mesh")

    args = parser.parse_args()
    working_dir = args.working_dir
    ref_dir = args.ref_dir

    reference_frames = [0, 1, 2, 3, 4]
    robot_dat = json.load(open(f"{working_dir}/data.json"))
    reference_rgbs, reference_pcds, reference_pressures = load_frames(f"{working_dir}/expand", reference_frames)

    points, triangles, boundary, boundary_mask = unpack_mesh(f"{ref_dir}/equalized.vtk")
    force_estimator = ForceFromPunyo(reference_rgbs, reference_pcds, reference_pressures, points, triangles, boundary, 
                                     rest_internal_force=None, precompile=False, verbose=True)

    START_FRAME = 30
    END_FRAME = len(robot_dat)
    os.makedirs(f"{working_dir}/result", exist_ok=True)
    for i in range(START_FRAME, END_FRAME):
        pressure, pcd, rgb = load_data(f"{working_dir}/expand", i)
        force_estimator.update(rgb, pcd, pressure)

        contact_forces = force_estimator.observed_force
        displaced_points = force_estimator.current_points

        np.save(f"{working_dir}/result/force_{i}.npy", contact_forces)
        np.save(f"{working_dir}/result/points_{i}.npy", displaced_points)

        print(f"Frame {i} done.")