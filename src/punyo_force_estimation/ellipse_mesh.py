import numpy as np
import pygmsh

# CAD measurement (center)
#x_bounds = (-0.0283, 0.0917)

# CAD measurement (left camera)
#x_bounds = (-0.0201, 0.0999)

# Fudge + CAD measurement (left camera + fudge)
#x_bounds = (-0.0228, 0.0972)
x_bounds = (-0.017, 0.0975)
y_bounds = (-0.04, 0.04)
#y_bounds = (-0.038, 0.038)

# WRONG old fudge
#x_bounds = (-0.013, 0.087)
#y_bounds = (-0.033, 0.033)
#y_bounds = (-0.0345, 0.0345)
#x_bounds = (-0.021, 0.099)
#y_bounds = (-0.04, 0.04)

#def axis_aligned_ellipse(x_bounds, y_bounds, n_ellipse_points = 128):
#def axis_aligned_ellipse(x_bounds, y_bounds, n_ellipse_points = 64):
def axis_aligned_ellipse(x_bounds, y_bounds, n_ellipse_points = 48):
    x0 = (x_bounds[1] + x_bounds[0]) / 2
    y0 = (y_bounds[1] + y_bounds[0]) / 2
    ellipse = []
    x_size = (x_bounds[1] - x_bounds[0]) / 2
    y_size = (y_bounds[1] - y_bounds[0]) / 2
    for i in range(n_ellipse_points):
        angle = i/n_ellipse_points * 2*np.pi
        ellipse.append([x0 + np.cos(angle) * x_size, y0 + np.sin(angle) * y_size])
    ellipse = np.array(ellipse)
    return ellipse

outer_ellipse = axis_aligned_ellipse(x_bounds, y_bounds)

if __name__ == "__main__":
    with pygmsh.geo.Geometry() as geom:
        ellipse = np.hstack([outer_ellipse, np.ones([len(outer_ellipse), 1]) * 0.0540])
        geom.add_polygon(ellipse, mesh_size=0.05)
        mesh = geom.generate_mesh()
        # Old fudge
        #mesh.points[:, 2] = 0.0543

        # CAD measurement (center)
        #mesh.points[:, 2] = 0.05618

        # CAD measurement (left)
        #mesh.points[:, 2] = 0.05238

        # CAD measurement (left) + fudge
        #mesh.points[:, 2] = 0.0540

        points = np.asarray(mesh.points)
        triangles = mesh.cells_dict['triangle']
        print(len(points), len(triangles), len(mesh.cells_dict['vertex']))

    mesh.write("base_meshes/0.vtk")
