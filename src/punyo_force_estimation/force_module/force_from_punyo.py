import time

import numpy as np
import torch

from .deformation_estimator import DeformationEstimation
from .force_predictor import CoupledForcePrediction
#from force_module.force_predictor_custom import CoupledForcePrediction
from .material_model import CorotatedPlaneStressModel, triangle_optimal_rotation
from ..utils import *

# TODO: come up with a better way to (dynamically?) determine the uncertainty
#k_force = 0.0003
#k_force = 0.00003
#k_force = 0.00001

#k_force = 0.01 * (MESH_SCALE**2)   # quad form
DEFAULT_K_FORCE = 0.25                     # (l1 of l2)^2
#k_force = 0.01  # l1 norm norm2s
#k_disp = 40000
#k_disp = 500 / (MESH_SCALE**2)   # quad form
#k_disp = 500 / (MESH_SCALE**2)   # quad form
DEFAULT_K_DISP = 720000   # l1 of l2
#k_disp = 100 / (MESH_SCALE**2)   # l1 of l2
#k_disp = 50   # l1 norm norm2s

NU_DEFAULT = 0.5
#E = 400
#E = 800
#E_DEFAULT = 1400    # l1 of l2
E_DEFAULT = 1200    # l1 of l2
#E = 1200   # quad form
#E = 2000
#E = 5000
#E_DEFAULT = 2400

# Optimization result
k_force = 0.34601
k_disp = 534094
E_DEFAULT = 737.1

#bad_points = np.load("ref_data/point_err_mean.npy", allow_pickle=True) > 0.0005

class ForceFromPunyo:

    """
    Construct a force predictor for the punyo sensor.

    Params:
    ---------------------
    reference_rgbs:         list of rgb images to compute reference mesh
    reference_pcds:         list of point clouds to compute reference mesh
    reference_pressures:    list of rgb images to compute reference pressure
    points:                 mesh points. nx3 array (float)
    triangles:              mesh connectivity: kx3 array (int), indices into points array.
    boundary:               list of vertices on the mesh boundary. will be fixed. bx3 array
    rest_internal_force:    pre-calibrated internal tension forces within each triangular element. kx3x3 array
                                where row i of the 3z3 element corresponds to 3d force at node i of triangle k.
    material_model:         which material model to use. Currently will break on things other than the default
    force_penalty:          One of:
                                None: use defaults.
                                length n array: penalty weights for forces at each node individually in QP.
                                scalar: penalty weight for force at every node in QP (with some adjustment at boundary).
    displacement_penalty:   One of:
                                None: use defaults.
                                length n array: penalty weights for displacements at each node individually in QP.
                                scalar: penalty weight for displacement at every node in QP (with some adjustment at boundary).
    optimization_params:    One of:
                                None: use defaults.
                                Dict of:
                                    nu: material poisson ratio
                                    E:  material young's modulus
    precompile:             bool, whether the force predictor should precompile cvxpy problems. Faster for many evaluations (~10).
    method:                 str, "L1", or "L2". L1 norm is more accurate but slower than L2.
    """
    def __init__(self, reference_rgbs, reference_pcds, reference_pressures, points, triangles, boundary,
                 rest_internal_force=None, material_model=CorotatedPlaneStressModel(),
                 force_penalty=None, displacement_penalty=None, optimization_params = {'nu': NU_DEFAULT, "E": E_DEFAULT},
                 precompile=True, verbose=False, method="L1"):

        self.method = method
        self.verbose = verbose

        boundary_mask = np.zeros(len(points), dtype=np.uint8)
        for n in boundary:
            boundary_mask[n] = 1
        self.triangles = triangles
        self.boundary = boundary
        self.boundary_mask = boundary_mask
        self.material_model = material_model
        self.deformation_estimator, undeformed_mesh_points, self.reference_pressure = get_reference_mesh(
                    reference_rgbs, reference_pcds, reference_pressures,
                    points, triangles, boundary, smooth_iters=10)

        self.undeformed_points = torch.tensor(undeformed_mesh_points).double()

        if rest_internal_force is None:
            rest_internal_force = np.zeros((len(triangles), 3, 3))
        self.rest_internal_force = rest_internal_force
        
        self.precompile = precompile
        self.force_predictor = None
        self.cache_E = None
        self.update_opt_params(force_penalty, displacement_penalty, optimization_params)

        self.current_pressure = self.reference_pressure
        self.current_points = self.undeformed_points
        self.observed_force = np.zeros_like(self.current_points)
        self.dt = (0, 0, 0) # deform, preproc, solve

    def _get_force_penalty_const(self, const):
        mesh_scale = len(self.undeformed_points) / len(self.boundary)
        if self.method == "L1":
            return (
                    np.array(1 - self.boundary_mask, dtype=np.float64) * const
                    + np.array(self.boundary_mask, dtype=np.float64) * (const / mesh_scale)
                )
        if self.method == "L2":
            return (
                    np.array(1 - self.boundary_mask, dtype=np.float64) * const * len(self.undeformed_points)
                    + np.array(self.boundary_mask, dtype=np.float64) * (const * mesh_scale)
                )

    def _get_disp_penalty_const(self, const):
        mesh_scale = len(self.undeformed_points) / len(self.boundary)
        displacement_penalty = (
                np.array(1 - self.boundary_mask, dtype=np.float64) * const / (mesh_scale**2)
                + np.array(self.boundary_mask, dtype=np.float64) * const / (mesh_scale**2) * 1000000
            )
        #displacement_penalty[bad_points > 0] /= 4
        return displacement_penalty

    def update_opt_params(self, force_penalty, displacement_penalty, optimization_params):

        if force_penalty is None:
            force_penalty = self._get_force_penalty_const(DEFAULT_K_FORCE)
        else:
            try:
                iter(force_penalty)
            except:
                force_penalty = self._get_force_penalty_const(force_penalty)
        if displacement_penalty is None:
            displacement_penalty = self._get_disp_penalty_const(DEFAULT_K_DISP)
        else:
            try:
                iter(displacement_penalty)
            except:
                displacement_penalty = self._get_disp_penalty_const(displacement_penalty)
        self.force_penalty = force_penalty
        self.displacement_penalty = displacement_penalty

        if optimization_params['E'] == self.cache_E:
            self.force_predictor.update_penalties(force_penalty, displacement_penalty)
        else:
            self.cache_E = optimization_params['E']
            params = []
            for n, (i, j, k) in enumerate(self.triangles):
                params.append([optimization_params['E'], optimization_params['nu'], self.rest_internal_force[n]])
            self.force_predictor = CoupledForcePrediction(self.triangles, params,
                                                          self.material_model, self.undeformed_points, self.boundary,
                                                          force_penalty, displacement_penalty,
                                                          precompile=self.precompile, verbose=self.verbose, method=self.method)


    def update(self, rgb, pcd, pressure, alpha=1):
        t0 = time.time()
        filt_pressure = pressure*alpha + self.current_pressure*(1-alpha)

        raw_deformed_points, data = self.deformation_estimator.estimate(rgb, pcd)
        raw_deformed_points = raw_deformed_points @ PC_ROTATION_MATRIX
        self._raw_points = raw_deformed_points

        raw_deformed_points = torch.tensor(raw_deformed_points).double()
        #raw_deformed_points = torch.tensor(smooth_mesh_taubin(raw_deformed_points, self.triangles, self.boundary, num_iters=1)).double()
        deformed_points = raw_deformed_points*alpha + self.current_points*(1-alpha)

        t1 = time.time()

        pressure_change = filt_pressure - self.reference_pressure
        pressure_forces = np.zeros((len(deformed_points), 3))

        #shape_forces = np.zeros((len(deformed_points), 3))

        areas = np.zeros(len(deformed_points))
        for tri_idx, (i, j, k) in enumerate(self.triangles):
            a = deformed_points[j] - deformed_points[i]
            b = deformed_points[k] - deformed_points[i]
            area_v = np.cross(a, b) / 2
            area = np.linalg.norm(area_v)
            areas[i] += area/3
            areas[j] += area/3
            areas[k] += area/3
            normal_force = pressure_change * area_v / 3
            pressure_forces[i, :] += normal_force
            pressure_forces[j, :] += normal_force
            pressure_forces[k, :] += normal_force


#            tri_forces = self.rest_internal_force[tri_idx]
#            R = triangle_optimal_rotation(self.undeformed_points[i], self.undeformed_points[j], self.undeformed_points[k], deformed_points[i], deformed_points[j], deformed_points[k]).T
#            force_change = np.array((R @ tri_forces.T).T - tri_forces)
#            #if tri_idx == 183:
#            #    HACK_internal_force = force_change
#            #    print(force_change)
#            shape_forces[i, :] += force_change[0, :]
#            shape_forces[j, :] += force_change[1, :]
#            shape_forces[k, :] += force_change[2, :]
        
        t2 = time.time()

        #self._shape_forces = shape_forces

        observed_flag = data[3]
        #displacement_penalty = np.copy(self.displacement_penalty)
        #_obs[_obs > 0.9] = 0.9
        #obs_mask = np.logical_or(observed_flag, bad_points)
#        displacement_penalty[obs_mask > 0] /= 4
#        force_penalty = np.copy(self.force_penalty)
#        force_penalty[obs_mask > 0] *= 1
#
#        observed_force, corrected_points, f_internal, penalties = self.force_predictor.estimate(
#            deformed_points, pressure_forces, force_penalty, displacement_penalty
#        )
        observed_force, corrected_points, f_internal, penalties = self.force_predictor.estimate(deformed_points, pressure_forces)

        for v in self.boundary:
            corrected_points[v, :] = self.undeformed_points[v, :]

        self.current_pressure = filt_pressure
        self.current_points = corrected_points
        self.observed_force = observed_force
        self.flow_data = data

        # debug/vis use
        self._pressure_forces = pressure_forces
        self._deformed_points = deformed_points
        self._f_internal = f_internal
        self._areas = areas
        #self._observed = obs_mask

        t3 = time.time()
        self.dt = (t1-t0, t2-t1, t3-t2)
