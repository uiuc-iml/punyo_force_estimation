"""Expand the "single file" data format taken during data collection
into a bunch of indivitual files (because the other code likes it this way)
"""

import json

import numpy as np

def explode_data(working_dir):
    robot_dat = json.load(open(f"{working_dir}/data.json"))
    pcds = np.load(f"{working_dir}/pcd.npy", allow_pickle=True)
    rgbs = np.load(f"{working_dir}/color.npy", allow_pickle=True)

    os.makedirs(f"{working_dir}/expand", exist_ok=True)

    for i in range(len(robot_dat)):
        time, pressure_hPa, joint_cfg, force = robot_dat[i]
        pointcloud = np.asanyarray(pcds[i]).view(np.float32).reshape(-1, 3)
        rgb = rgbs[i]

        np.save(f"{working_dir}/expand/pressure_{i}.npy", pressure_hPa)
        np.save(f"{working_dir}/expand/pc_{i}.npy", pointcloud)
        np.save(f"{working_dir}/expand/rgb_{i}.npy", rgb)

if __name__ == "__main__":
    import argparse
    import pathlib
    import os

    parser = argparse.ArgumentParser(description="generate initial mesh for no force config")
    parser.add_argument('working_dir', type=pathlib.Path)
    args = parser.parse_args()
    working_dir = os.path.expanduser(args.working_dir)
    explode_data(working_dir)
