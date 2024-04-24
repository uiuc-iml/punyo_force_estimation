"""Plot pressure over the recorded trajectory.

Useful for determining which parts are in contact.
"""

import json
import matplotlib.pyplot as plt

def plot_pressure(working_dir):
    robot_dat = json.load(open(f"{working_dir}/data.json"))

    pressures = []
    for i in range(len(robot_dat)):
        time, pressure_hPa, joint_cfg, force = robot_dat[i]
        pressures.append(pressure_hPa*100)
    
    plt.figure()
    plt.plot(pressures)
    plt.show()

if __name__ == "__main__":
    import argparse
    import pathlib
    import os

    parser = argparse.ArgumentParser(description="generate initial mesh for no force config")
    parser.add_argument('working_dir', type=pathlib.Path)
    args = parser.parse_args()
    working_dir = os.path.expanduser(args.working_dir)
    plot_pressure(working_dir)

