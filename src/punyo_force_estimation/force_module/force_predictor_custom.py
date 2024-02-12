import time
import torch
import scipy.sparse
from scipy.sparse import coo_matrix
import cvxpy as cp
import numpy as np
np.set_printoptions(linewidth=260)

import socplib.socp

from .material_model import BaseMaterialModel

def idx_pts2mat(pts_idx):
    mat_idx = torch.tensor([[3*i, 3*i+1, 3*i+2] for i in pts_idx]).reshape(-1)
    return torch.cartesian_prod(mat_idx, mat_idx)

def assemble_K_mat(element_lst, material_lst, material_model: BaseMaterialModel,
                   rest_points, curr_points, sp_type='coo'):
    num_pts = curr_points.shape[0]

    if isinstance(rest_points, np.ndarray):
        rest_points = torch.tensor(rest_points).double()
    if isinstance(curr_points, np.ndarray):
        curr_points = torch.tensor(curr_points).double()

    r_idx_lst = []
    c_idx_lst = []
    data = []
    for e, m in zip(element_lst, material_lst):

        p = rest_points[e, :].flatten()
        u = curr_points[e, :].flatten() - p

        #m = torch.tensor(m)

        #print(e)
        K = material_model.element_stiffness(p, u, m)
        #print(K)
        #input()

        rc_idx = idx_pts2mat(e)
        r_idx_lst.extend(rc_idx[:, 0])
        c_idx_lst.extend(rc_idx[:, 1])
        data.extend(K.reshape(-1))

    sp_K_mat = coo_matrix((data, (r_idx_lst, c_idx_lst)), shape=(3*num_pts, 3*num_pts))

    if sp_type == "coo":
        return sp_K_mat
    elif sp_type == "csr":
        return sp_K_mat.tocsr()
    elif sp_type == "csc":
        return sp_K_mat.tocsc()

def build_coo(vec, eps):
    data = []
    rc = []
    for i, v in enumerate(vec):
        rc.extend(list(range(3*i, 3*i+3)))
        if v == 0:
            data.extend([eps, eps, eps])
        else:
            data.extend([v, v, v])
    return np.array([rc, rc, data]).T

class CoupledForcePrediction:
    def __init__(self, elements, element_params, material_model: BaseMaterialModel,
                 rest_points, boundary_idx, force_penalty, displacement_penalty, precompile=True, verbose=False,
                 method="L1"):
        self.method = method
        self.material_model = material_model
        self.elements = elements
        self.element_params = element_params
        self.rest_points = np.array(rest_points)
        #print(self.rest_points)

        n = len(rest_points)
        self._n = n
        self._K = cp.Parameter((3*n, 3*n), PSD=True)
        self._x0 = np.array(self.rest_points.flatten())
        self._x1 = cp.Parameter(3*n)
        self._u = cp.Variable(3*n)
        self._u.value = np.zeros(3*n)
        self.force_penalty = force_penalty
        self._known_force = cp.Parameter((n, 3))
        self._W_u = cp.Parameter((3*n, 3*n), symmetric=True)
        self._boundary_idx = boundary_idx
        self._boundary_mask = np.zeros(3*len(self.rest_points))

        self.constraints = []
        self.constraints_A = []
        for i in self._boundary_idx:
            i = int(i)
            self._boundary_mask[3*i:3*(i+1)] = 1
            self.constraints.append(self._u[3*i:3*(i+1)] == 0)
            self.constraints_A.append([3*i, 3*i, 1])
            self.constraints_A.append([3*i+1, 3*i+1, 1])
            self.constraints_A.append([3*i+2, 3*i+2, 1])
        self.static_K = assemble_K_mat(self.elements, self.element_params, self.material_model,
                              self.rest_points, self.rest_points, sp_type="csr")
        self.K_slices = []
        for i in range(self._n):
            k_slice = self.static_K[3*i:3*i+3, :].tocoo()
            list_fmt = np.array([k_slice.row, k_slice.col, k_slice.data]).T.tolist()
            self.K_slices.append(list_fmt)

        self.static_W_u = build_coo(displacement_penalty, eps=1e-7)
        self.update_penalties(force_penalty, displacement_penalty)
        self.precompile = precompile
        self.verbose = verbose
        self.dt = 0

    def update_penalties(self, force_penalty, displacement_penalty):
        f_ext = -(self.static_K @ self._u + self._known_force.T.flatten())

        if self.method == "L1":
            f_ext_v = cp.reshape(f_ext, (3, self._n))
            f_norms = cp.multiply(cp.norm(f_ext_v, 2, axis=0), force_penalty)
            self.f_norm_penalty = cp.norm1(f_norms)

        #self.u_penalty = cp.quad_form((self._x0 + self._u) - self._x1, static_W_u, assume_PSD=True)

        #self.cp_objective = cp.Minimize(self.f_norm_penalty + self.u_penalty)
        #self.cp_problem = cp.Problem(self.cp_objective, self.constraints)

    def assemble_K_matrix(self, observed_points, compute_problem=True):
        self.constraints = []
        for i in self._boundary_idx:
            i = int(i)
            self._boundary_mask[3*i:3*(i+1)] = 1
            self.constraints.append(self._u[3*i:3*(i+1)] == 0)
        #self._boundary_mask = self._boundary_mask.reshape(
        static_K = assemble_K_mat(self.elements, self.element_params, self.material_model,
                              self.rest_points, observed_points)

        f_ext = -(static_K @ self._u + self._known_force.T.flatten())

        if self.method == "L1":
            f_ext_v = cp.reshape(f_ext, (3, self._n))
            f_norms = cp.multiply(cp.norm(f_ext_v, 2, axis=0), force_penalty)
            self.f_norm_penalty = cp.norm1(f_norms)

        self.cp_objective = cp.Minimize(self.f_norm_penalty + self.u_penalty)
        self.cp_problem = cp.Problem(self.cp_objective, self.constraints)
        return static_K

    def estimate(self, observed_points: np.ndarray, known_force: np.ndarray, reassemble_K=False, verbose=False) -> tuple:
        t0 = time.time()

        precompile = self.precompile
        if reassemble_K:
            precompile = False
            sp_K = self.assemble_K_matrix(observed_points)
        else:
            sp_K = self.static_K

        x1 = np.array(observed_points.flatten())
        dx = self._x0 - x1;

        print("lib code")
        socplib.socp.init_solver()
        L_trans = np.sqrt(self.static_W_u).tolist()
        c = self.static_W_u[:, -1] * dx
        socplib.socp.set_lin_term(c)
        socplib.socp.add_linear_constraint(self.constraints_A, np.zeros(len(self.constraints_A)))
        socplib.socp.add_quad_term(L_trans, len(dx))
        for i in range(self._n):
            socplib.socp.add_l2_term(self.K_slices[i], known_force[i], self.force_penalty[i])
        print("added l2")
        socplib.socp.finalize_solver()

        result = socplib.socp.socp_solve()
        print("solve done")

        u_opt:np.ndarray = np.array(result[2])
        print(u_opt)
        socplib.socp.dealloc_solver()
        print("solver dealloc done")

        t1 = time.time()
        for [node] in self._boundary_idx:
            u_opt[3*node:3*(node+1)] = 0

        corrected_points = self.rest_points + u_opt.reshape((-1, 3))
        #corrected_points = observed_points + u_opt.reshape((-1, 3))
        f_internal = sp_K @ u_opt
        observed_force = -(f_internal + known_force.flatten()).reshape((-1, 3))
        resid = 99#(self.f_norm_penalty.value, self.u_penalty.value)
        #print(f_total.value)
        self.dt = t1-t0
        return (observed_force, corrected_points, f_internal.reshape((-1, 3)), resid)
