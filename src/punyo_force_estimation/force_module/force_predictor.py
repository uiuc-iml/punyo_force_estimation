import time
import torch
import scipy.sparse
from scipy.sparse import coo_matrix
import cvxpy as cp
import numpy as np
np.set_printoptions(linewidth=260)

try:
    from .material_model import BaseMaterialModel
except:
    from material_model import BaseMaterialModel

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

    #k_tmp = sp_K_mat.todense()
    #from PIL import Image
    #Image.fromarray(k_tmp != 0).save("K.png")
    

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
    return coo_matrix((data, (rc, rc)))

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
        #self._f_internal = cp.Parameter(3*n)
        self._x0 = cp.Parameter(3*n)
        self._x0.value = np.array(self.rest_points.flatten())
        self._x1 = cp.Parameter(3*n)
        self._u = cp.Variable(3*n)
        self._u.value = np.zeros(3*n)
        self.force_penalty = force_penalty
        self._known_force = cp.Parameter((n, 3))
        #self._W_f = cp.Parameter((3*n, 3*n), symmetric=True)
        self._W_u = cp.Parameter((3*n, 3*n), symmetric=True)
        self._boundary_idx = boundary_idx
        self._boundary_mask = np.zeros(3*len(self.rest_points))

        self.constraints = []
        for i in self._boundary_idx:
            i = int(i)
            self._boundary_mask[3*i:3*(i+1)] = 1
            self.constraints.append(self._u[3*i:3*(i+1)] == 0)
        #self._boundary_mask = self._boundary_mask.reshape(
        self.static_K = assemble_K_mat(self.elements, self.element_params, self.material_model,
                              self.rest_points, self.rest_points)
        self.update_penalties(force_penalty, displacement_penalty)
        self.precompile = precompile
        self.verbose = verbose
        self.dt = 0

    def update_penalties(self, force_penalty, displacement_penalty):
        f_ext = -(-self.static_K @ self._u + self._known_force.T.flatten())

        static_W_u = build_coo(displacement_penalty, eps=1e-7)

        if self.method == "L1":
            f_ext_v = cp.reshape(f_ext, (3, self._n))
            f_norms = cp.multiply(cp.norm(f_ext_v, 2, axis=0), force_penalty)
            self.f_norm_penalty = cp.norm1(f_norms)
        if self.method == "L2":
            static_W_f = build_coo(force_penalty, eps=1e-7)
            self.f_norm_penalty = cp.quad_form(f_ext, static_W_f, assume_PSD=True)

        #self.f_norm_penalty = cp.norm(cp.vstack(f_norms), 1) * 0.95 + cp.norm(self._W_f @ f_ext, 2) * 0.05
        #self.f_norm_penalty = cp.norm1(cp.vstack(f_norms))
        #self.f_norm_penalty = cp.norm(cp.vstack(f_norms), 1) + cp.quad_form(f_ext, self._W_f, assume_PSD=True)*0.1
        #self.u_penalty = cp.quad_form((self._x0 + self._u) - self._x1, self._W_u, assume_PSD=True)

        self.u_penalty = cp.quad_form((self._x0 + self._u) - self._x1, static_W_u, assume_PSD=True)

        self.cp_objective = cp.Minimize(self.f_norm_penalty + self.u_penalty)
        self.cp_problem = cp.Problem(self.cp_objective, self.constraints)

    def assemble_K_matrix(self, observed_points, compute_problem=True):
        self.constraints = []
        for i in self._boundary_idx:
            i = int(i)
            self._boundary_mask[3*i:3*(i+1)] = 1
            self.constraints.append(self._u[3*i:3*(i+1)] == 0)
        #self._boundary_mask = self._boundary_mask.reshape(
        static_K = assemble_K_mat(self.elements, self.element_params, self.material_model,
                              self.rest_points, observed_points)

        f_ext = -(-static_K @ self._u + self._known_force.T.flatten())

        if self.method == "L1":
            f_ext_v = cp.reshape(f_ext, (3, self._n))
            f_norms = cp.multiply(cp.norm(f_ext_v, 2, axis=0), self.force_penalty)
            self.f_norm_penalty = cp.norm1(f_norms)
        if self.method == "L2":
            static_W_f = build_coo(self.force_penalty, eps=1e-7)
            self.f_norm_penalty = cp.quad_form(f_ext, static_W_f, assume_PSD=True)

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
        self._x1.value = np.array(observed_points.flatten())

        # NOTE: testing code.
        # No regularization, direct solve using F=KX
        #corrected_points = observed_points.numpy()
        #u_opt = (corrected_points - self.rest_points).reshape(-1)
        #f_internal = sp_K @ u_opt
        #observed_force = -(f_internal + known_force.flatten()).reshape((-1, 3))
        #resid = (self.f_norm_penalty.value, self.u_penalty.value)
        #print(f_total.value)
        #return (observed_force, corrected_points, f_internal.reshape((-1, 3)), resid)

        #area_calc = observed_points
        #areas = np.zeros(len(area_calc))
        #for (i, j, k) in self.elements:
        #    a = area_calc[j] - area_calc[i]
        #    b = area_calc[k] - area_calc[i]
        #    area_v = np.cross(a, b) / 2
        #    area = np.linalg.norm(area_v)
        #    areas[i] += area/3
        #    areas[j] += area/3
        #    areas[k] += area/3
        #print(max(areas))

        #f_penalty_normalized = np.mean(areas) * force_penalty / areas
        #print(f_penalty_normalized)
        #self._W_f.value = build_coo(f_penalty_normalized, eps=1e-5)
        #self._W_f.value = build_coo(force_penalty, eps=1e-5)

        #data = []
        #rows = []
        #cols = []
        #for i, v in enumerate(displacement_penalty):
        #    point = observed_points[i]  # Referenced to camera coordinates.
        #    normal_dir = point / np.linalg.norm(point)
        #    A = np.zeros((3, 3))
        #    A[0, :] = normal_dir
        #    A = (A.T @ A) + (np.eye(3) * 0.001)

        #    for j in range(3):
        #        for k in range(3):
        #            rows.append(3*i+j)
        #            cols.append(3*i+k)
        #            data.append(displacement_penalty[i] * A[j][k])
        #W_u = coo_matrix((data, (rows, cols)))

        #self._W_u.value = build_coo(displacement_penalty, eps=1e-7)
        #self._force_penalty.value = force_penalty

        self._known_force.value = known_force

        #res = problem.solve(ignore_dpp=True, verbose=True)

        # NOTE: ignore_dpp is faster for ~20 solves or less. Otherwise we want precompilation.
        #   For running timing tests, remove this flag. Initial compilation can take ~100sec;
        #   afterwards solves take much less time!
        #
        # Approximate (total) timing: 1.5-5sec without compilation
        #   SCS Solver time: ~0.5sec

        ################# TESTING #################
        if self.precompile:
            res = self.cp_problem.solve(solver="SCS", warm_start=True, verbose=self.verbose, eps=1e-4, use_indirect=True)
        else:
            res = self.cp_problem.solve(ignore_dpp=True, solver="SCS", warm_start=False, verbose=self.verbose, eps=1e-4)
        # self._u.value = observed_points.flatten().numpy() - self.rest_points.flatten()
        ############### END TESTING ###############

        u_opt:np.ndarray = self._u.value

        t1 = time.time()
        # if verbose or self.verbose:
        #     print("min objective:", res, " time:", t1-t0) 
        #     print("solver status:", self.cp_problem.status)
        #     print("force cost:", self.f_norm_penalty.value)

        #     print("displacement cost:", self.u_penalty.value)
        for [node] in self._boundary_idx:
            u_opt[3*node:3*(node+1)] = 0

        corrected_points = self.rest_points + u_opt.reshape((-1, 3))
        #corrected_points = observed_points + u_opt.reshape((-1, 3))
        f_internal = -sp_K @ u_opt
        observed_force = -(f_internal + known_force.flatten()).reshape((-1, 3))
        resid = (self.f_norm_penalty.value, self.u_penalty.value)
        #print(f_total.value)
        self.dt = t1-t0
        return (observed_force, corrected_points, f_internal.reshape((-1, 3)), resid)
