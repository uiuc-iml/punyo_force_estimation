import cv2
import numpy as np
import scipy.interpolate

#sx = (640/2) / np.tan(np.radians(87/2))
#sy = (480/2) / np.tan(np.radians(58/2))
#sx = (820/2) / np.tan(np.radians(87/2))
#sy = (470/2) / np.tan(np.radians(58/2))
#image_center = [640/2, 480/2]
#image_center = [630/2, 490/2]

# these are grabbed from ros, topic: /color/camera_info
# K matrix is 2x3 transform:
# fx 0  cx
# 0  fy cy

# the sensor on the table
#sx = 429.51
#sy = 428.93
#image_center = [313.77, 247.67]

# the sensor on the kinova
#sx = 430.101
#sy = 429.507
#image_center = [312.768, 238.591]

# calibrated with TRINA calibration script
#sx = 388.877
#sy = 388.298
#image_center = [318.651, 234.539]

# the depth sensor on the kinova
sx = 385.263
sy = 385.263
image_center = [307.943, 241.596]

def get_confidence_filter(source, epsilon=0.001, blur_shape=(71, 71)):
    laplacian = np.abs(cv2.Laplacian(source, cv2.CV_32F, ksize=3)) + epsilon
    weights = 1 / cv2.GaussianBlur(laplacian, blur_shape, 0)
    return laplacian, weights

def unproject_pointcloud(proj_pc, proj_pc2, pc2):
    return scipy.interpolate.griddata(
        proj_pc2,
        pc2,
        proj_pc
    )

def pointcloud_flow(proj_pc, pc2, proj_pc2, flow, xs=None, ys=None):
    if xs is None:
        xs = np.array(list(range(480))) + 0.5
    if ys is None:
        ys = np.array(list(range(640))) + 0.5
    flow_2d = scipy.interpolate.interpn(
        (xs, ys),
        flow,
        proj_pc[:, ::-1],
        method='linear',
        bounds_error=False,
        fill_value=(0,0)
    )
    displaced_proj_pc = proj_pc + flow_2d
    end_points = unproject_pointcloud(displaced_proj_pc, proj_pc2, pc2)
    return end_points, flow_2d

def refine_optical_flow(source, flow, confidence, inv_conf, blur_shape=(71, 71)):
    """Refine the optical flow calculation by selectively blurring to fill in
    low confidence regions.

    TODO: do some stats

    Parameters:
    --------------------
    source:     ndarray         The first image of the optical flow pair (nxm)
    flow:       ndarray         The optical flow vector field (nxmxd, usually d=2)
    confidence: ndarray         Confidence in each reading
    inv_conf:   ndarray         "Inverse confidence" (blurred 1/x)
    blur_shape: pair(int, int)  Shape of the blur kernel. Larger is more aggressive blurring
    """
    ret = np.empty(flow.shape)
    for i in range(flow.shape[-1]):
        ret[:, :, i] = cv2.GaussianBlur(confidence * flow[:, :, i], blur_shape, 0) * inv_conf
    return ret

def project_pointcloud(pc):
    proj_pc = pc[:, :2] / (pc[:, 2].reshape(-1, 1))
    proj_pc *= [[sx, sy]]
    proj_pc += image_center
    return proj_pc
