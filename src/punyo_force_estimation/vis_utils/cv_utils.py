# Punyo Soft-Bubble Sensor - Copyright 2023 Toyota Research Institute. All rights reserved.

import cv2
import numpy as np
import scipy as sp
import open3d as o3d

def forces_to_rgb(points, forces, x_range, y_range, pix_size):
    x_mid = (x_range[0] + x_range[1]) / 2
    y_mid = (y_range[0] + y_range[1]) / 2
    x_half = x_mid - x_range[0]
    y_half = y_mid - y_range[0]
    x_halfrange = int(np.ceil(x_half / pix_size))
    y_halfrange = int(np.ceil(y_half / pix_size))

    x_values = np.linspace(-1, 1, 2*x_halfrange+1) * x_half + x_mid
    y_values = -np.linspace(-1, 1, 2*y_halfrange+1) * y_half + y_mid

    xs, ys = np.meshgrid(x_values, y_values)
    print(f"generate image, size={xs.shape}")
    print(np.min(y_values), np.max(y_values))
    print(np.min(points[:, 1]), np.max(points[:, 1]))

    xy_flat = np.vstack([xs.flatten(), ys.flatten()]).T
    print(xy_flat.shape)

    forces = -np.copy(forces)
    force_intensities = np.linalg.norm(forces, axis=1)
    max_force_val = 4000
    #max_force_val = np.max(force_intensities)
    forces[force_intensities > max_force_val] *= (max_force_val/force_intensities[force_intensities > max_force_val]).reshape(-1, 1)
    #pixel_scaling = (force_intensities / max_force_val) * 255
    force_rgb = np.hstack([((forces * 0.5 / max_force_val) + [0.5, 0.5, 0.5]), np.ones([len(forces), 1])]) * 255
    interp_xy = points[:, :2]
    rgb_flat = sp.interpolate.griddata(interp_xy, force_rgb, xy_flat, fill_value=0)
    return np.array(rgb_flat.reshape((*xs.shape, 4)), dtype=np.uint8)

def visualize_flow_arrows(flow, img, scale=2, step=60):
    """ Given a flow, add an overlay of force vectors to the image. """
    h, w = img.shape[0], img.shape[1]
    flag = False
    color = (20, 255, 255)  # BGR

    arrows_img = np.zeros_like(img)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Add the arrows, skipping every *step* pixels
    for i in range(0, mag.shape[0], step):
        for j in range(0, mag.shape[1], step):
            mags = scale * mag[i, j]
            if int(mags):
                ndx = min(i + int(mags * np.sin(ang[i, j])), h)
                ndy = min(j + int(mags * np.cos(ang[i, j])), w)
                pt1 = (j, i)
                pt2 = (max(ndy, 0), max(ndx, 0))
                arrows_img = cv2.arrowedLine(
                    arrows_img,
                    pt1,
                    pt2,
                    color,
                    6,
                    tipLength=0.25,
                )
                flag = True
    if flag:
        if len(img.shape) == 3:
            # Just want to overlay the arrows
            img2gray = cv2.cvtColor(arrows_img, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            masked_source = cv2.bitwise_and(img, img, mask=mask_inv)
            img = cv2.addWeighted(arrows_img, 1.0, masked_source, 1.0, 0)
        else:
            img = cv2.add(img, arrows_img)

    return img


def force_arrow_visualize(img, f, centroid, scale=40):
    h, w = img.shape[0], img.shape[1]

    if centroid is None:
        center = (int(w / 2.0), int(h / 2.0))
    else:
        center = (centroid[0], centroid[1])

    shear_tip = np.around(np.array([center[0] + f[0] * scale, center[1] + f[1] * scale])).astype(int)

    img = cv2.arrowedLine(
        img,
        pt1=center,
        pt2=tuple(shear_tip),
        color=(250, 250, 250),
        thickness=2,
        tipLength=0.5,
    )
    normal_tip = np.around(
        np.array([center[0], center[1] + f[2] * scale])
    ).astype(int)

    img = cv2.arrowedLine(
        img,
        pt1=center,
        pt2=tuple(normal_tip),
        color=(50, 255, 50),
        thickness=2,
        tipLength=0.5,
    )
    return img


def merge_cylinder_segments(cylinders):
     vertices_list = [np.asarray(mesh.vertices) for mesh in cylinders]
     triangles_list = [np.asarray(mesh.triangles) for mesh in cylinders]
     triangles_offset = np.cumsum([v.shape[0] for v in vertices_list])
     triangles_offset = np.insert(triangles_offset, 0, 0)[:-1]
    
     vertices = np.vstack(vertices_list)
     triangles = np.vstack([triangle + offset for triangle, offset in zip(triangles_list, triangles_offset)])
    
     merged_mesh = o3d.geometry.TriangleMesh(o3d.open3d.utility.Vector3dVector(vertices), 
                                             o3d.open3d.utility.Vector3iVector(triangles))
     return merged_mesh

def create_graph_lines(points, edges, scale=0.001):
    """ Visualize line set """
    #line_set = o3d.geometry.LineSet()
    #line_set.points = o3d.utility.Vector3dVector(points)
    #line_set.lines = o3d.utility.Vector2iVector(edges)
    #return line_set
    segments = []
    for line in edges:
        segments.append(create_vector_cylinder(points[line[0]], points[line[1]], scale=scale))
    return merge_cylinder_segments(segments)

def calculate_zy_rotation_for_arrow(vec):
    gamma = np.arctan2(vec[1], vec[0])
    Rz = np.array([
                    [np.cos(gamma), -np.sin(gamma), 0],
                    [np.sin(gamma), np.cos(gamma), 0],
                    [0, 0, 1]
                ])
    vec = Rz.T @ vec
    beta = np.arctan2(vec[0], vec[2])
    Ry = np.array([
                    [np.cos(beta), 0, np.sin(beta)],
                    [0, 1, 0],
                    [-np.sin(beta), 0, np.cos(beta)]
                ])
    return Rz, Ry

def create_vector_cylinder(end, origin=np.array([0, 0, 0]), scale=1, color=[0.707, 0.707, 0.0]):
    assert(not np.all(end == origin))
    vec = end - origin
    size = np.sqrt(np.sum(vec**2))
    mid = (end + origin) / 2
    Rz, Ry = calculate_zy_rotation_for_arrow(vec)
    mesh = o3d.geometry.TriangleMesh.create_cylinder(
        radius=scale,
        height=size)
    mesh.rotate(Ry, center=np.array([0, 0, 0]))
    mesh.rotate(Rz, center=np.array([0, 0, 0]))
    mesh.translate(mid)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return mesh

def create_vector_arrow(end, origin=np.array([0, 0, 0]), scale=1, color=[0.707, 0.707, 0.0]):
    assert(not np.all(end == origin))
    vec = end - origin
    size = np.sqrt(np.sum(vec**2))
    Rz, Ry = calculate_zy_rotation_for_arrow(vec)
    mesh = o3d.geometry.TriangleMesh.create_arrow(cone_radius=size/17.5 * scale,
        cone_height=size*0.2 * scale,
        cylinder_radius=size/30 * scale,
        cylinder_height=size*(1 - 0.2*scale))
    mesh.rotate(Ry, center=np.array([0, 0, 0]))
    mesh.rotate(Rz, center=np.array([0, 0, 0]))
    mesh.translate(origin)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return mesh

def create_arrow_lst(p1_ary, p2_ary, **args):
    arrow_lst = []
    for p1, p2 in zip(p1_ary, p2_ary):
        if np.linalg.norm(p2-p1) > 0.001:
            arrow_lst.append(create_vector_arrow(p2, origin=p1, **args))
    return arrow_lst
