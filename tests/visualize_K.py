from src.punyo_force_estimation.force_module.material_model import CorotatedPlaneStressModel

import numpy as np
import torch
import open3d as o3d
from torch.func import jacrev

cps = CorotatedPlaneStressModel()

p = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float64)
u = torch.tensor([0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
m = (1.0, 0.3, np.array([0.0, 0.0, 0.0], dtype=np.float64))

f = cps.element_forces(p, u, m)
K = cps.element_stiffness(p, u, m)

print(cps.element_forces(p, u, m))
print(cps.element_forces(p, 2*u, m))


for i in range(3):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(p.numpy().reshape(-1, 3))
    pcd.normals = o3d.utility.Vector3dVector(f.numpy().reshape(-1, 3) * 10)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(p.numpy().reshape(-1, 3))
    mesh.triangles = o3d.utility.Vector3iVector(np.array([[0, 1, 2]]))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.5, 0.5, 0.5])
    o3d.visualization.draw_geometries([pcd, mesh], point_show_normal=True)