import torch
from torch.func import jacrev

class BaseMaterialModel:
    def __init__(self):
        super().__init__()

    def element_forces(self, p, u, material):
        """Compute the forces given the displacement and material parameters."""
        raise NotImplementedError
    
    def element_stiffness(self, p, u, material):
        """Compute the stiffness matrix given the displacement and material parameters."""
        raise NotImplementedError

    def element_stiffness_autodiff(self, p, u, material):
        """Compute the stiffness matrix given the displacement and material parameters."""
        return jacrev(self.element_forces, argnums=1)(p, u, material)

class LinearPlaneStressModel(BaseMaterialModel):
    def __init__(self):
        super().__init__()


    def element_forces(self, p, u, material):
        """Compute the forces given the displacement and material parameters."""
        E, nu = material
        stress_c1 = E / (1 - nu*nu)
        stress_c2 = E / (2 * (1 + nu))

        o1, p1, q1 = p.split(3, dim=0)
        do, dp, dq = u.split(3, dim=0)
        o2 = o1 + do
        p2 = p1 + dp
        q2 = q1 + dq

        a_undeformed = p1 - o1
        b_undeformed = q1 - o1
        a_deformed = p2 - o2
        b_deformed = q2 - o2

        nA = torch.linalg.cross(a_deformed, b_deformed).detach()
        #nA = torch.linalg.cross(a_undeformed, b_undeformed)
        A = torch.norm(nA)
        kHat = nA / A

        s = b_deformed - a_deformed
        #s = b_undeformed - a_undeformed
        iHat = s / torch.norm(s)
        jHat = torch.linalg.cross(kHat, iHat)
        R_local = torch.stack([iHat, jHat], dim=1).detach()

        u_mat = torch.stack([a_deformed - a_undeformed, b_deformed - b_undeformed], dim=0)
        u_local = u_mat @ R_local

        T = R_local.T @ torch.stack([a_deformed, b_deformed], dim=1).detach()
        #T = R_local.T @ torch.stack([a_undeformed, b_undeformed], dim=1)
        #res = torch.linalg.solve(T.T, torch.hstack([u_local, torch.tensor([[-1], [-1]])]))
        T_inv_T = torch.linalg.inv(T.T)
        res = T_inv_T @ torch.hstack([u_local, torch.tensor([[-1, 1, 0], [-1, 0, 1]])])
        ui_i = res[0, 0]
        uj_j = res[1, 1]
        ui_j = res[1, 0]
        uj_i = res[0, 1]

        stress_local = torch.zeros((2, 2)).double()
        stress_local[0, 0] = stress_c1*(ui_i + nu*uj_j)
        stress_local[0, 1] = stress_c2*(ui_j+uj_i)
        stress_local[1, 0] = stress_c2*(ui_j+uj_i)
        stress_local[1, 1] = stress_c1*(uj_j + nu*ui_i)

        forces = (R_local @ (stress_local @ res[:, 2:]) * A/2).T.flatten()
#        if boundary_mask[0] and boundary_mask[1]:
#            #out_normal = torch.cross(a_deformed, kHat) / 2  # has length/2 factor rolled in
#            out_normal = torch.cross(a_undeformed, kHat) / 2  # has length/2 factor rolled in
#            stress_transformed = R_local @ stress_local @ R_local.T
#            forces[0:3] += stress_transformed @ out_normal
#            forces[3:6] += stress_transformed @ out_normal
#        if boundary_mask[0] and boundary_mask[2]:
#            #out_normal = torch.cross(kHat, b_deformed) / 2  # has length/2 factor rolled in
#            out_normal = torch.cross(kHat, b_undeformed) / 2  # has length/2 factor rolled in
#            stress_transformed = R_local @ stress_local @ R_local.T
#            forces[0:3] += stress_transformed @ out_normal
#            forces[6:9] += stress_transformed @ out_normal
#        if boundary_mask[1] and boundary_mask[2]:
#            #out_normal = torch.cross(kHat, b_deformed) / 2  # has length/2 factor rolled in
#            out_normal = torch.cross(kHat, s) / 2  # has length/2 factor rolled in
#            stress_transformed = R_local @ stress_local @ R_local.T
#            forces[3:6] += stress_transformed @ out_normal
#            forces[6:9] += stress_transformed @ out_normal
        return forces

    def element_stiffness(self, p, u, material):
        r = self.element_stiffness_autodiff(p, u, material)
        return -r   # TODO hack

def triangle_optimal_rotation(x1, x2, x3, y1, y2, y3):
    x_centroid = (x1 + x2 + x3) / 3
    y_centroid = (y1 + y2 + y3) / 3
    A = (torch.stack([x1, x2, x3], dim=1) - x_centroid).T
    B = (torch.stack([y1, y2, y3], dim=1) - y_centroid).T
    u, s, vh = torch.linalg.svd(A @ B.T)
    # convert the output from a (possibly only) orthogonal matrix to rotation matrix
    s_fudge = torch.diag(torch.tensor([1, 1, torch.linalg.det(u @ vh)]))
    R = (u @ s_fudge @ vh)
    return R

class CorotatedPlaneStressModel(BaseMaterialModel):
    def __init__(self):
        super().__init__()

    def element_forces(self, p, u, material):
        """Compute the forces given the displacement and material parameters."""
        E, nu, tri_forces = material
        tri_forces = torch.tensor(tri_forces)
        stress_c1 = E / (1 - nu*nu)
        stress_c2 = E / (2 * (1 + nu))

        o1, p1, q1 = p.split(3, dim=0)
        do, dp, dq = u.split(3, dim=0)
        o2 = o1 + do
        p2 = p1 + dp
        q2 = q1 + dq

        a = p2 - o2
        b = q2 - o2
        nA = torch.linalg.cross(a, b).detach()
        A = torch.norm(nA)
        kHat = nA / A


        R_grad = triangle_optimal_rotation(o1, p1, q1, o2, p2, q2).T
        force_change = (R_grad @ tri_forces.T).T - tri_forces
        R = R_grad.detach()

        a_undeformed = p1 - o1
        b_undeformed = q1 - o1
        a_deformed = p2 - o2
        b_deformed = q2 - o2

        s = b_deformed - a_deformed
        iHat = s / torch.norm(s)
        jHat = torch.linalg.cross(kHat, iHat)
        R_local = torch.stack([iHat, jHat], dim=1).detach()

        u_mat = torch.stack([a_deformed - (R @ a_undeformed), b_deformed - (R @ b_undeformed)], dim=0)
        u_local = u_mat @ R_local

        T = R_local.T @ torch.stack([a_deformed, b_deformed], dim=1).detach()
        #T = R_local.T @ torch.stack([a_undeformed, b_undeformed], dim=1)
        T_inv_T = torch.linalg.inv(T.T)
        res = T_inv_T @ torch.hstack([u_local, torch.tensor([[-1, 1, 0], [-1, 0, 1]])])
        ui_i = res[0, 0]
        uj_j = res[1, 1]
        ui_j = res[1, 0]
        uj_i = res[0, 1]

        stress_local = torch.zeros((2, 2)).double()
        stress_local[0, 0] = stress_c1*(ui_i + nu*uj_j)
        stress_local[0, 1] = stress_c2*(ui_j+uj_i)
        stress_local[1, 0] = stress_c2*(ui_j+uj_i)
        stress_local[1, 1] = stress_c1*(uj_j + nu*ui_i)

        #forces = ((R_local @ (stress_local @ res[:, 2:]) * A/2).T - force_change).flatten()
        forces = ((R_local @ (stress_local @ res[:, 2:]) * A/2).T).flatten()
        #forces = (-force_change).flatten()
        return forces

    def element_stiffness(self, p, u, material):
        r = -self.element_stiffness_autodiff(p, u, material)
        return r


def tri_K_2d(p:torch.Tensor, m:torch.Tensor):
    E, nu = m[:2]

    x1, y1, x2, y2, x3, y3 = p

    J = torch.tensor([[x1-x3, y1-y3], [x2-x3, y2-y3]])
    det_J = torch.det(J)

    D = torch.tensor([[1, nu, 0],[nu, 1, 0], [0, 0, (1-nu)/2]])
    D *= E/(1-nu*nu)

    B = torch.tensor([[y2-y3,     0, y3-y1,     0, y1-y2,     0],
                      [    0, x3-x2,     0, x1-x3,     0, x2-x1],
                      [x3-x2, y2-y3, x1-x3, y3-y1, x2-x1, y1-y2]])
    B /= det_J

    Ae = 0.5*torch.abs(det_J)

    return Ae*(B.T @ D @ B)

def tri_local_coord(p1, p2, p3):
    n = torch.cross(p3-p1, p2-p1)
    n_unit = (n / torch.norm(n)).detach()
    e1 = ((p3-p1)/torch.norm(p3-p1)).detach()
    e2 = torch.cross(n_unit, e1).detach()
    return e1, e2, n_unit

def proj2d(p: torch.Tensor):
    p1, p2, p3 = p.split(3, dim=0)
    e1, e2, _ = tri_local_coord(p1, p2, p3)

    proj_p2 = torch.zeros(2)
    proj_p2[0] = torch.dot(p2-p1, e1)
    proj_p2[1] = torch.dot(p2-p1, e2)

    proj_p3 = torch.zeros(2)
    proj_p3[0] = torch.dot(p3-p1, e1)
    proj_p3[1] = torch.dot(p3-p1, e2)

    return proj_p2, proj_p3


def triangle_3d(p:torch.Tensor, u:torch.Tensor, m:torch.Tensor):
    proj_p2, proj_p3 = proj2d(p)
    proj_q2, proj_q3 = proj2d(p+u)

    proj_p = torch.cat([torch.zeros(2), proj_p2, proj_p3])
    K = tri_K_2d(proj_p, m)
    
    u2d = torch.cat([torch.zeros(2), proj_q2 - proj_p2, proj_q3 - proj_p3])
    f2d = -K @ u2d

    q1, q2, q3 = (p+u).split(3, dim=0)
    e1q, e2q, _ = tri_local_coord(q1, q2, q3)

    f1 = e1q*f2d[0] + e2q*f2d[1]
    f2 = e1q*f2d[2] + e2q*f2d[3]
    f3 = e1q*f2d[4] + e2q*f2d[5]

    return f1, f2, f3


class LinearTriangleModel(BaseMaterialModel):
    def __init__(self):
        super().__init__()

        self.p_dim = 9
        self.u_dim = 9
        self.m_dim = 2
        self.f_dim = 9
        self.geom_type = "triangle"

    def element_forces(self, p, u, material):
        f1, f2, f3 = triangle_3d(p, u, material)
        return torch.cat([f1, f2, f3])

    def element_stiffness(self, p, u, material):
        return self.element_stiffness_autodiff(p, u, material)
