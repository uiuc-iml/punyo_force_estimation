import time

import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import meshio
import open3d as o3d

from ..pipeline import *
from ..ellipse_mesh import outer_ellipse

pc_rotation = 25*(np.pi/180)
CAMERA_R = np.array([
    [np.cos(pc_rotation), 0, -np.sin(pc_rotation)],
    [0, 1, 0],
    [np.sin(pc_rotation), 0, np.cos(pc_rotation)],
])
mesh_plane_z = 0.054

z_tol = 0.000
max_bubble_height = 0.045
downsample_skip = 1

class DeformationEstimation:
    def __init__(self, base_rgb, base_pc, pc_rot_M, mesh_plane_z, tracked_points, fixed_points=None):
        self.mesh_plane_z = mesh_plane_z
        self.camera_rot = pc_rot_M
        self.set_base_frame(base_rgb, base_pc)

        self.mesh_points = tracked_points @ pc_rot_M.T

        # This is some fudge to create a "boundary" on the mesh, for interpolation purposes...
        #ground_ellipse_rotated = np.hstack([outer_ellipse * 1.15, np.ones([len(outer_ellipse), 1]) * mesh_plane_z])
        ground_ellipse_rotated = np.hstack([outer_ellipse, np.ones([len(outer_ellipse), 1]) * mesh_plane_z])
        #ground_ellipse_rotated[:, 0] -= 0.0025

        self.proj_mesh = project_pointcloud(self.mesh_points)
        outer_ellipse_3D = ground_ellipse_rotated @ pc_rot_M.T
        
        # plus indicators for OOB
        self.outer_ellipse_3D_flagged = np.hstack([outer_ellipse_3D, np.ones((len(outer_ellipse_3D), 1))])
        self.proj_outer_ellipse = project_pointcloud(outer_ellipse_3D)

        self.fixed_points = fixed_points
        self.dt = (0, 0)    # project, flow


    def filter_pc(self, pc):
        rotated_pc = pc[:, :3] @ self.camera_rot
        pc_filter = (
            (rotated_pc[:, 2] < self.mesh_plane_z + max_bubble_height + z_tol)
            * (rotated_pc[:, 2] > self.mesh_plane_z - z_tol)
        )
        pc = pc[pc_filter, :]
        o3d_pc = o3d.geometry.PointCloud()
        o3d_pc.points = o3d.utility.Vector3dVector(pc[:, :3])
        #c1, ind = o3d_pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.2)
        c1, ind = o3d_pc.remove_statistical_outlier(nb_neighbors=200, std_ratio=1.2)
        return np.asarray(c1.points)

    def set_base_frame(self, base_rgb, base_pc):
        """set the "home"/reference rgb image and pointcloud.
        """
        self.base_rgb = base_rgb
        self.base_gray = cv2.cvtColor(self.base_rgb, cv2.COLOR_BGR2GRAY)[::downsample_skip, ::downsample_skip]

        self.base_pc = self.filter_pc(base_pc)
        self.base_proj_pc = project_pointcloud(self.base_pc)

    def estimate(self, rgb, pc, tracked_points=None):
        """estimate deformation of the mesh from a given observed rgb image and observed pointcloud.

        Parameter           type            desc
        ---------------------------------------------
        rgb:                wxhx3 array     new frame RGB image.
        pc:                 nx3 array       new frame pointcloud image.
        tracked_points:     nx3 array       points to apply estimated motion to.

        Return
        ---------------------------------------------
        pair (estimated_points, data), where data is misc. internals for plotting
        """
        t0 = time.time()

        if tracked_points is None:
            proj_mesh = self.proj_mesh
        else:
            proj_mesh = project_pointcloud(tracked_points @ self.camera_rot.T)

        PYR_SCALE = 0.5
        CPE_DEPTH_OFFSET = 5
        FLOW_WINDOW_SIZE = 50
        #FLOW_WINDOW_SIZE = 25
        FLOW_LEVELS = 30
        #FLOW_LEVELS = 10
        current_gray = cv2.cvtColor(rgb[::downsample_skip, ::downsample_skip], cv2.COLOR_BGR2GRAY)
        flow_vector = cv2.calcOpticalFlowFarneback(
            self.base_gray,
            current_gray,
            np.zeros(self.base_gray.shape),
            PYR_SCALE,
            FLOW_LEVELS,
            FLOW_WINDOW_SIZE,
            iterations=5,
            poly_n=7,
            poly_sigma=1.5,
            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
        )
        t1 = time.time()

        current_pc_filtered = self.filter_pc(pc)
        current_proj_pc = project_pointcloud(current_pc_filtered)

        #c, c_inv = get_confidence_filter(self.base_gray)
        #flow_vector2 = refine_optical_flow(self.base_gray, flow_vector, c, c_inv)
        flow_vector2 = flow_vector

        current_pc_flagged = np.hstack([current_pc_filtered, np.zeros((len(current_pc_filtered), 1))])

        xs = np.array(list(range(0, 480, downsample_skip))) + 0.5
        ys = np.array(list(range(0, 640, downsample_skip))) + 0.5
        mesh_displaced, mesh_flow_2d = pointcloud_flow(
            proj_mesh,
            np.vstack([current_pc_flagged, self.outer_ellipse_3D_flagged]),
            np.vstack([current_proj_pc, self.proj_outer_ellipse]),
            flow_vector2,
            xs=xs, ys=ys
        )

        if self.fixed_points is not None:
            for n in self.fixed_points:
                mesh_displaced[n] = np.array([*(self.mesh_points[n, :][0]), 0])
        t2 = time.time()

        print("Deformation estimate time:", t2-t0)
        self.dt = (t1 - t0, t2 - t1)

        return mesh_displaced[:, :3], (flow_vector, flow_vector2, mesh_flow_2d, mesh_displaced[:, 3])

