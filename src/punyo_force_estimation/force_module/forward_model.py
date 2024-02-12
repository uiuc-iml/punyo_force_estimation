import time
import torch
import scipy.sparse
from scipy.sparse import coo_matrix
import cvxpy as cp
import numpy as np
np.set_printoptions(linewidth=260)

from .material_model import BaseMaterialModel, CorotatedPlaneStressModel
from ..utils import estimate_tri_normals

class PVConstantModel(BaseMaterialModel):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.dp_dv = 0

    def set_PV(self, P, V):
        # PV = C
        # P = C/V
        # dp = -C/(V^2) dV
        #    = -P/V
        self.dp_dv = -P/V

    def element_forces(self, p, u, material):
        f = self.base_model.element_forces(p, u, material)
        o1, p1, q1 = p.split(3, dim=0)
        do, dp, dq = u.split(3, dim=0)

        a = p1 - o1
        b = q1 - o1
        nA = torch.linalg.cross(a, b).detach()
        A = torch.norm(nA)
        kHat = nA / A

        ox = torch.dot(do, kHat)
        px = torch.dot(dp, kHat)
        qx = torch.dot(dq, kHat)

        # A is twice area.
        prop_factor = self.dp_dv * (A * A / 4) / 3
        Fx = prop_factor * ox * kHat
        Fy = prop_factor * oy * kHat
        Fz = prop_factor * oz * kHat

        return f

    def element_stiffness(self, p, u, material):
        r = self.base_model.element_stiffness(p, u, material)

        o1, p1, q1 = p.split(3, dim=0)
        do, dp, dq = u.split(3, dim=0)

        o2 = o1 + do
        p2 = p1 + dp
        q2 = q1 + dq

        a = p1 - o1
        b = q1 - o1
        nA = torch.linalg.cross(a, b).detach()
        A = torch.norm(nA)
        kHat = nA / A

        # A is twice area.
        prop_factor = self.dp_dv * (A * A / 4) / 3

        res = torch.zeros((9, 9)).double()
        
        # vector v. (point on triangle)
        for v in range(3):
            for out_dim in range(3):
                for in_dim in range(3):
                    res[v*3 + out_dim, v*3 + in_dim] = prop_factor * kHat[out_dim] * kHat[in_dim]

        return r + res
        #return r


def mesh_volume(vertices, triangles, z_floor):
    """
    NOTE: Assumes the mesh is aligned with +Z as upwards and the bottom on even Z level...
    """

    total_v = 0
    for (i, j, k) in triangles:
        a = vertices[j] - vertices[i]
        b = vertices[k] - vertices[i]
        a[2] = 0
        b[2] = 0

        # overcounting by a factor of 2.
        area = np.linalg.norm(np.cross(a, b))

        z_avg = (vertices[i][2] + vertices[j][2] + vertices[k][2]) / 3

        total_v += area*(z_avg - z_floor)

    return total_v / 2;


def idx_pts2mat(pts_idx):
    mat_idx = torch.tensor([[3*i, 3*i+1, 3*i+2] for i in pts_idx]).reshape(-1)
    return torch.cartesian_prod(mat_idx, mat_idx)

def assemble_K_mat(element_lst, material_lst, material_model: BaseMaterialModel,
                   rest_points, curr_points, boundary, boundary_mask, sp_type='coo'):
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
        for (r_idx, c_idx, entry) in zip(rc_idx[:, 0], rc_idx[:, 1], K.reshape(-1)):
            if boundary_mask[r_idx] or boundary_mask[c_idx]:
                continue
            r_idx_lst.append(r_idx)
            c_idx_lst.append(c_idx)
            data.append(entry)
        #r_idx_lst.extend(rc_idx[:, 0])
        #c_idx_lst.extend(rc_idx[:, 1])
        #data.extend(K.reshape(-1))

    for i in boundary:
        i = int(i)
        r_idx_lst.extend(range(3*i, 3*i+3))
        c_idx_lst.extend(range(3*i, 3*i+3))
        data.extend([1, 1, 1])

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



class ForwardModel:
    def __init__(self, elements, element_params, material_model: BaseMaterialModel,
                 rest_points, boundary_idx, p0, verbose=False):
        self.material_model = PVConstantModel(material_model)
        #self.material_model = material_model
        self.p0 = p0;

        z_floor = np.inf
        for point in rest_points:
            if point[2] < z_floor:
                z_floor = point[2]
        self.z_floor = z_floor
        self.v0 = mesh_volume(rest_points, elements, z_floor)

        self.elements = elements
        self.element_params = element_params
        self.rest_points = np.array(rest_points)
        #print(self.rest_points)

        n = len(rest_points)
        self._n = n
        self._boundary_idx = boundary_idx
        self._boundary_mask = np.zeros(3*len(self.rest_points))

        for i in self._boundary_idx:
            i = int(i)
            self._boundary_mask[3*i:3*(i+1)] = 1

        self.verbose = verbose
        self.dt = 0

    def sim_forward(self, forces, gt_pressure):

        pv = self.v0 * self.p0
        self.material_model.set_PV(self.p0, self.v0)
        total_ext_force = np.sum(forces, axis=0)

        forces_flat = forces.reshape(-1)

        points = self.rest_points
        boundary = self._boundary_idx
        distances = np.zeros(len(boundary))
        for i, [n] in enumerate(boundary):
            delta = points[n] - points[boundary[i-1][0]]
            distances[i-1] = np.linalg.norm(delta)
        total_distance = np.sum(distances)

        #alpha = 0.9
        alpha = 0
        print(self.p0, self.v0, alpha)
        pred_points = self.rest_points

        area_vectors = np.zeros_like(forces)
        for tri_idx, (i, j, k) in enumerate(self.elements):
            a = points[j] - points[i]
            b = points[k] - points[i]
            area_v = np.cross(a, b) / 2
            area_v = area_v / 3
            area_vectors[i, :] += area_v
            area_vectors[j, :] += area_v
            area_vectors[k, :] += area_v
        normals = estimate_tri_normals(self.rest_points, self.elements)

        #pressure_forces = np.zeros_like(forces)
        dp = gt_pressure - self.p0
        pressure_forces = area_vectors * dp
        for i in self._boundary_idx:
            pressure_forces[int(i)] = 0

        #for i in range(10):
        for i in range(1):
            total_pressure_force = np.sum(pressure_forces, axis=0)

            total_force = total_pressure_force + total_ext_force

            rim_forces = np.zeros_like(forces)
            for i, [n] in enumerate(boundary):
                dist_fraction = (distances[i-1] + distances[i])/(2*total_distance)
                rim_forces[n] -= dist_fraction * total_force

            pressure_forces_flat = pressure_forces.reshape(-1)
            rim_forces_flat = rim_forces.reshape(-1)
            K = assemble_K_mat(self.elements, self.element_params, self.material_model,
                               self.rest_points, self.rest_points, self._boundary_idx, self._boundary_mask).tocsr()
            #u = scipy.sparse.linalg.spsolve(K, -(forces_flat + pressure_forces_flat + rim_forces_flat));
            u = scipy.sparse.linalg.spsolve(K, -(forces_flat + pressure_forces_flat));

            for i in self._boundary_idx:
                i = int(i)
                u[3*i:3*i+3] = 0

            pred_points = self.rest_points + u.reshape(-1, 3)*(1-alpha)

            new_v = mesh_volume(pred_points, self.elements, self.z_floor);
            new_p = pv / new_v
            self.material_model.set_PV(new_p, new_v)

            pressure_change = new_p - self.p0
            #pressure_forces = area_vectors * pressure_change
            for i in self._boundary_idx:
                pressure_forces[int(i)] = 0

            alpha *= 0.9
            print(new_p, new_v, alpha, gt_pressure)

        K = assemble_K_mat(self.elements, self.element_params, self.material_model.base_model,
                           self.rest_points, self.rest_points, self._boundary_idx, self._boundary_mask).todense()
        K_diag = np.zeros(K.shape)
        for i in range(0, K.shape[0], 3):
            K_diag[i:i+3, i:i+3] = K[i:i+3, i:i+3]
        return pred_points, pressure_forces, new_p, (K_diag @ normals.reshape(-1)).reshape(-1, 3)
