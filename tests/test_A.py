import numpy as np
import open3d as o3d
import trimesh

vertices = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [10, 10, 0], [5, 5, 5]])
triangles = np.array([[0, 1, 4], [1, 3, 4], [0, 4, 2], [3, 2, 4]])

edge01 = vertices[1] - vertices[0]
edge04 = vertices[4] - vertices[0]
edge13 = vertices[3] - vertices[1]
edge14 = vertices[4] - vertices[1]

points = [edge01 * 0.15 + edge04 * 0.4 + vertices[0], edge13 * 0.4 + edge14 * 0.35 + vertices[1]]
points = np.array(points)

# plot the mesh with open3d
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(vertices)
mesh.triangles = o3d.utility.Vector3iVector(triangles)
mesh.compute_vertex_normals()

pts = o3d.geometry.PointCloud()
pts.points = o3d.utility.Vector3dVector(points)
pts.paint_uniform_color([1.0, 0.0, 0.0])

o3d.visualization.draw_geometries([mesh, pts])

def compute_A(vertices, triangles, points):
    tri_mesh = trimesh.Trimesh(vertices, triangles)
    # find closest triangle
    closest_tri, dist, closest_tri_idx = trimesh.proximity.closest_point(tri_mesh, points)
    triangles_nearest = tri_mesh.triangles[closest_tri_idx]
    barycentric = trimesh.triangles.points_to_barycentric(triangles_nearest, points)
    print(barycentric)

    vertices_flatten = vertices.flatten()
    points_flatten = points.flatten()
    A = np.zeros((points_flatten.shape[0], vertices_flatten.shape[0]))
    point_idx = np.arange(points_flatten.shape[0] // 3)
    for pt_idx, bary, tri_idx in zip(point_idx, barycentric, closest_tri_idx):
        tri = triangles[tri_idx]
        for i, tri_v in enumerate(tri):
            A[pt_idx*3:pt_idx*3+3, tri_v*3:tri_v*3+3] = np.eye(3) * bary[i]
    return A

A = compute_A(vertices, triangles, points)
dvertices = np.array([[1, 2, -1], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
dvertices_flatten = dvertices.flatten()
dpoints_flatten = A @ dvertices_flatten
dpoints = dpoints_flatten.reshape(-1, 3)

# plot the mesh with open3d
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(vertices + dvertices)
mesh.triangles = o3d.utility.Vector3iVector(triangles)
mesh.compute_vertex_normals()

pts = o3d.geometry.PointCloud()
pts.points = o3d.utility.Vector3dVector(points + dpoints)
pts.paint_uniform_color([1.0, 0.0, 0.0])

o3d.visualization.draw_geometries([mesh, pts])

# compute barycentric coordinates with trimesh
tri_mesh = trimesh.Trimesh(vertices + dvertices, triangles)
# find closest triangle
closest_tri, dist, closest_tri_idx = trimesh.proximity.closest_point(tri_mesh, points + dpoints)
triangles_nearest = tri_mesh.triangles[closest_tri_idx]
barycentric = trimesh.triangles.points_to_barycentric(triangles_nearest, points + dpoints)
print(barycentric)