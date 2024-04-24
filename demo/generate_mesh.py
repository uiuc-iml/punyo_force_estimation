import os
import argparse
import pathlib

from punyo_force_estimation import generate_initial_mesh

parser = argparse.ArgumentParser(description="generate initial mesh for no force config")
parser.add_argument('-i', '--working_dir', type=pathlib.Path, default="0/expand", help="Folder containing data (bunch of numpy arrays of rgb, pcd, pressure data of the bubble not in contact with anything)")
parser.add_argument('-n', '--num_points', type=int, default=64, help="Number of points on the perimeter of the mesh")
parser.add_argument('-o', '--out_dir', type=pathlib.Path, default="ref_data", help="Folder to dump output files in. Will be created recursively if it does not exist.")
args = parser.parse_args()
n_points = args.num_points
working_dir = os.path.expanduser(args.working_dir)
out_dir = os.path.expanduser(args.out_dir)

generate_initial_mesh.gen_ref_data(n_points, working_dir, out_dir)
