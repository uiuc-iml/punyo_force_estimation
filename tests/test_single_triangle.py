from punyo_force_estimation.force_module.material_model import CorotatedPlaneStressModel

import numpy as np
import torch
import open3d as o3d

if __name__ == "__main__":
    cps = CorotatedPlaneStressModel()

    p = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float64)

    move_idx = 0
    move_step = 0.01
    normal_scale = 10.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(p.numpy().reshape(-1, 3))

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(p.numpy().reshape(-1, 3))
    mesh.triangles = o3d.utility.Vector3iVector(np.array([[0, 1, 2]]))

    def create_update_pcd_function(vec):
        def update_pcd(vis):
            current_point = np.array(pcd.points)
            current_point[move_idx] += move_step*vec
            pcd.points = o3d.utility.Vector3dVector(current_point)
            mesh.vertices = o3d.utility.Vector3dVector(current_point)        

            u = torch.tensor(current_point.reshape(-1), dtype=torch.float64) - p
            f = cps.element_forces(p, u, (1.0, 0.3, np.array([1.0, 0.0, 0.0], dtype=np.float64)))
            pcd.normals = o3d.utility.Vector3dVector(f.numpy().reshape(-1, 3) * normal_scale)
            vis.update_geometry(pcd)
            vis.update_geometry(mesh)
            vis.poll_events()
            vis.update_renderer
        return update_pcd

    key_to_callback = {}
    key_lst = ['Q', 'W', 'E', 'A', 'S', 'D']
    vec_lst = [ np.array([1.0, 0.0, 0.0], dtype=np.float64), np.array([0.0, 1.0, 0.0], dtype=np.float64), np.array([0.0, 0.0, 1.0], dtype=np.float64),
                np.array([-1.0, 0.0, 0.0], dtype=np.float64), np.array([0.0, -1.0, 0.0], dtype=np.float64), np.array([0.0, 0.0, -1.0], dtype=np.float64) ]
    for key, vec in zip(key_lst, vec_lst):  
        key_to_callback[ord(key)] = create_update_pcd_function(vec)
    

    # Interactive visualize pcd and normals
    o3d.visualization.draw_geometries_with_key_callbacks([pcd, mesh], key_to_callback)
