import numpy as np
import open3d as o3d
import meshio
import scipy.sparse

import os

from .force_module.deformation_estimator import DeformationEstimation
from .utils import *

def predict_internal_force(points, triangles, boundary, total_pressure_delta):
    K_FORCE_INPLANE = 1
    K_TRI_FORCE_BALANCE = 1
    K_TRI_MOMENT_BALANCE = 100
    K_TRI_FORCE_REGULARIZE = 0.02
    K_NODE_FORCE = 1000000

    boundary_mask = np.zeros(len(points), dtype=np.uint8)
    for n in boundary:
        boundary_mask[n] = 1

    forces = np.zeros((len(points), 3))
    tri_normals = np.zeros((len(triangles), 3))
    for tri_idx, (i, j, k) in enumerate(triangles):
        a = points[j] - points[i]
        b = points[k] - points[i]
        area_v = np.cross(a, b) / 2
        area = np.linalg.norm(area_v)
        normal_force = total_pressure_delta * area_v / 3
        forces[i, :] += normal_force
        forces[j, :] += normal_force
        forces[k, :] += normal_force
        tri_normals[tri_idx, :] = area_v / area

    total_force = np.sum(forces, axis=0)
    distances = np.zeros(len(boundary))
    for i, [n] in enumerate(boundary):
        delta = points[n] - points[boundary[i-1][0]]
        distances[i-1] = np.linalg.norm(delta)
    total_distance = np.sum(distances)
    for i, [n] in enumerate(boundary):
        dist_fraction = (distances[i-1] + distances[i])/(2*total_distance)
        forces[n] -= dist_fraction * total_force

    areas = np.zeros(len(points))
    for tri_idx, (i, j, k) in enumerate(triangles):
        a = points[j] - points[i]
        b = points[k] - points[i]
        area_v = np.cross(a, b) / 2
        area = np.linalg.norm(area_v)
        areas[i] += area/3
        areas[j] += area/3
        areas[k] += area/3

    rhs = (-forces*K_NODE_FORCE*areas.reshape(-1, 1)).flatten().tolist()
    r_idx_lst = []
    c_idx_lst = []
    data = []

    forces_len = len(rhs)
    for tri_idx, triangle in enumerate(triangles):
        # Set up equilibrium equations for each triangle.
        i, j, k = triangle
        base_row = forces_len + tri_idx*18

        r1 = points[j] - points[i]
        r2 = points[k] - points[i]

        # cross product matrices.
        r1_cmat = np.array([[     0, -r1[2],  r1[1]],
                            [ r1[2],      0, -r1[0]],
                            [-r1[1],  r1[0],      0]])
        r2_cmat = np.array([[     0, -r2[2],  r2[1]],
                            [ r2[2],      0, -r2[0]],
                            [-r2[1],  r2[0],      0]])

        for x in range(3):
            # force should be in plane (x: the point in question)
            r_idx_lst.extend([base_row + x] * 3)
            c_idx_lst.extend([9*tri_idx+0+x*3, 9*tri_idx+1+x*3, 9*tri_idx+2+x*3])
            data.extend(tri_normals[tri_idx, :] * K_FORCE_INPLANE)
            rhs.append(0)

            # sum of force should be zero (x: the direction in question)
            r_idx_lst.extend([base_row + 3 + x] * 3)
            c_idx_lst.extend([9*tri_idx+x+0, 9*tri_idx+x+3, 9*tri_idx+x+6])
            data.extend([K_TRI_FORCE_BALANCE]*3)
            rhs.append(0)

            # sum of moment should be zero (x: the direction in question)
            # ignoring f0
            # r1xf1 + r2xf2 = 0
            r_idx_lst.extend([base_row + 6 + x] * 6)
            c_idx_lst.extend([9*tri_idx+3, 9*tri_idx+4, 9*tri_idx+5])
            c_idx_lst.extend([9*tri_idx+6, 9*tri_idx+7, 9*tri_idx+8])
            data.extend(K_TRI_MOMENT_BALANCE*r1_cmat[x])
            data.extend(K_TRI_MOMENT_BALANCE*r2_cmat[x])
            rhs.append(0)

            # Forces overall should be low (x: the direction in question)
            r_idx_lst.extend([base_row + 9 + 3*x, base_row + 9 + 3*x + 1, base_row + 9 + 3*x + 2])
            c_idx_lst.extend([9*tri_idx+x+0, 9*tri_idx+x+3, 9*tri_idx+x+6])
            data.extend([K_TRI_FORCE_REGULARIZE]*3)
            rhs.extend([0, 0, 0])

            # Add terms to node force balance. (x: the direction in question)
            for sub_idx, y in enumerate(triangle):
                if boundary_mask[y] and x != 2:
                    continue
                    #r_idx_lst.append(forces_len + 9*len(triangles) + x)
                    #c_idx_lst.append(9*tri_idx+x+3*sub_idx)
                    #data.append(1)
                #    continue
                r_idx_lst.append(3*y+x)
                c_idx_lst.append(9*tri_idx+x+3*sub_idx)
                data.append(K_NODE_FORCE*areas[y])
    #print(len(data), len(r_idx_lst), len(c_idx_lst))
    K = scipy.sparse.coo_matrix((data, (r_idx_lst, c_idx_lst)), shape=(forces_len + 18*len(triangles), 9*len(triangles)))

    pred_internal_force, istop, itn, *_ = scipy.sparse.linalg.lsqr(K, np.array(rhs))

    internal_forces = np.zeros_like(forces)
    all_tri_forces = np.zeros((len(triangles), 3, 3))
    for tri_idx, (o, p, q) in enumerate(triangles):
        adjust_idx = tri_idx * 9
        tri_forces = pred_internal_force[adjust_idx:adjust_idx+9].reshape((3, 3))
        internal_forces[o, :] += tri_forces[0, :]
        internal_forces[p, :] += tri_forces[1, :]
        internal_forces[q, :] += tri_forces[2, :]
        all_tri_forces[tri_idx, :, :] = tri_forces

    return forces, internal_forces, all_tri_forces

if __name__ == "__main__":
    dat = np.load("inflating_sequence/pressure.0.npy")
    atmospheric_pressure = dat[1] * 100
    #working_dir = os.path.expanduser("~/Documents/punyo_1688505384")   # Weights
    #reference_frames = [41, 42, 43, 44, 45, 46, 47] # early frames have weight on them
    #reference_frames = [70, 71, 72, 73, 74, 75] # early frames have weight on them
    #reference_frames = [70] # early frames have weight on them

    #working_dir = os.path.expanduser("~/Documents/punyo_1691767741")   # Weights aligned
    #reference_frames = [96, 97, 98, 99]
    working_dir = os.path.expanduser("~/Documents/punyo_unit_test/data_0/expand")   # poke with robot
    reference_frames = [0, 1, 2, 3, 4]

    #pc_rotation, mesh_plane_z = np.load("pc_transform.npy")
    #mesh = meshio.read("base_meshes/0.vtk")
    mesh = meshio.read("base_meshes/4.vtk")
    points, triangles, boundary, boundary_mask = unpack_mesh("base_meshes/4.vtk")

    reference_rgbs, reference_pcds, reference_pressures = load_frames(working_dir, reference_frames)
    deformation_estimator, undeformed_mesh_points, reference_pressure = get_reference_mesh(
                reference_rgbs, reference_pcds, reference_pressures,
                points, triangles, boundary, smooth_iters=10
            )
    print("ref pressure:", reference_pressure)
    points = undeformed_mesh_points
    pcd = pointcloud(points)
    mesh.points = points
    mesh.write("smoothed.vtk")

    o3d_mesh = make_o3d_mesh(points, triangles)

    total_pressure_delta = reference_pressure - atmospheric_pressure
    print(total_pressure_delta)
    print(atmospheric_pressure)

    forces, internal_forces, all_tri_forces = predict_internal_force(points, triangles, boundary, total_pressure_delta)
    #moment_errors = []
    #for tri_idx, triangle in enumerate(triangles):
    #    i, j, k = triangle
    #    base_row = forces_len + tri_idx*18
    #
    #    r1 = points[j] - points[i]
    #    r2 = points[k] - points[i]
    #
    #    # cross product matrices.
    #    r1_cmat = np.array([[     0, -r1[2],  r1[1]],
    #                        [ r1[2],      0, -r1[0]],
    #                        [-r1[1],  r1[0],      0]])
    #    r2_cmat = np.array([[     0, -r2[2],  r2[1]],
    #                        [ r2[2],      0, -r2[0]],
    #                        [-r2[1],  r2[0],      0]])
    #    f1_vec = pred_internal_force[9*tri_idx+3:9*tri_idx+6]
    #    f2_vec = pred_internal_force[9*tri_idx+6:9*tri_idx+9]
    #    moment_errors.extend(r1_cmat @ f1_vec + r2_cmat @ f2_vec)
    #print(np.linalg.norm(moment_errors))

    boundary_force = np.zeros(3)
    for n in boundary:
        boundary_force += internal_forces[n, :].flatten() + forces[n, :].flatten()

    total_force = np.sum(forces, axis=0)
    print("total pressure force:", total_force)
    print(boundary_force)
    #print(np.sum(forces, axis=0))
    print("error force:", np.sum((internal_forces + forces) * (1 - boundary_mask).reshape(-1, 1), axis=0))

    print("saving forces...")
    np.save("triangle_force.npy", all_tri_forces)

    pcd.normals = o3d.utility.Vector3dVector(forces*10)
    o3d.visualization.draw_geometries([
            pcd,
            o3d_mesh
        ], point_show_normal=True)
    pcd.normals = o3d.utility.Vector3dVector(internal_forces*10)
    o3d.visualization.draw_geometries([
            pcd,
            o3d_mesh
        ], point_show_normal=True)
    pcd.normals = o3d.utility.Vector3dVector((internal_forces + forces)*10)
    o3d.visualization.draw_geometries([
            pcd,
            o3d_mesh
        ], point_show_normal=True)

    import matplotlib.pyplot as plt
    plt.figure()
    ax = plt.gca()
    ax.tricontourf(points[:, 0], points[:, 1], np.linalg.norm(internal_forces + forces, axis=1))
    plt.show()

    for tri_idx, (o, p, q) in enumerate(triangles):
        #if tri_idx < 180:
        #    continue
        adjust_idx = tri_idx * 9
        elem_force = all_tri_forces[tri_idx, :, :]
        print(f"element {tri_idx} force:", elem_force)
        print(forces[o], forces[p], forces[q])
        r10 = points[p] - points[o]
        r20 = points[q] - points[o]
        r21 = points[q] - points[p]

        # cross product matrices.
        r10_cmat = np.array([[      0, -r10[2],  r10[1]],
                             [ r10[2],       0, -r10[0]],
                             [-r10[1],  r10[0],       0]])
        r20_cmat = np.array([[      0, -r20[2],  r20[1]],
                             [ r20[2],       0, -r20[0]],
                             [-r20[1],  r20[0],       0]])
        r21_cmat = np.array([[      0, -r21[2],  r21[1]],
                             [ r21[2],       0, -r21[0]],
                             [-r21[1],  r21[0],       0]])
        print("force balance:", np.sum(elem_force, axis=0))
        print("moment balance0:", r10_cmat @ elem_force[1, :] + r20_cmat @ elem_force[2, :])
        print("moment balance terms:", r10_cmat @ elem_force[1, :], r20_cmat @ elem_force[2, :])
        print("moment balance1:", r21_cmat @ elem_force[2, :] - r10_cmat @ elem_force[0, :])

        tri_normals = np.zeros_like(points)
        tri_normals[o] = elem_force[0]
        tri_normals[p] = elem_force[1]
        tri_normals[q] = elem_force[2]
        #print(tri_normals)

        pcd.normals = o3d.utility.Vector3dVector(tri_normals*10)

        o3d.visualization.draw_geometries([
                pcd,
                o3d_mesh
            ], point_show_normal=True)
