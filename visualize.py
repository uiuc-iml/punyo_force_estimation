import numpy as np
import open3d as o3d

START_FRAME = 5
END_FRAME = 150

working_dir = "data/shoe"

for i in range(START_FRAME, END_FRAME, 5):
    contact_forces = np.load(f"{working_dir}/result/force_{i}.npy") * 20
    displaced_points = np.load(f"{working_dir}/result/points_{i}.npy")
    
    # visualize the force vectors with arrows
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(displaced_points)
    pcd.normals = o3d.utility.Vector3dVector(contact_forces)
    # draw pcd with normals
    o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    

# pc_good = np.load("tmp/pc_good.npy")
# pc_bad = np.load("tmp/pc_bad.npy")
# # visualize the point clouds
# pcd_good = o3d.geometry.PointCloud()
# pcd_good.points = o3d.utility.Vector3dVector(pc_good)
# pcd_bad = o3d.geometry.PointCloud()
# pcd_bad.points = o3d.utility.Vector3dVector(pc_bad)

# pcd_bad.paint_uniform_color([0, 0, 1])
# pcd_good.paint_uniform_color([1, 0, 0])

# # draw pcd with normals
# o3d.visualization.draw_geometries([pcd_good, pcd_bad])