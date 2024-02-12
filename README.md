# Punyo bubble force estimation

Use linear FE (plane stress assumption) to predict forces from punyo sensor rgb and depth, and pressure readings.

To capture RGB images, you can use ROS or the realsense API.

To capture pressure data you'll need TRI's repository for the punyo sensor to run the micro-ros thingy.


## Dependencies

```
pip install numpy open3d cvxpy torch meshio scipy opencv-python pygmsh
```

## Installing it

In root dir:
```
python -m build

# or maybe I forgot to bump the version in README, but its the .whl file
pip install dist/punyo_force_estimation-0.0.1-py3-none-any.whl
```

## Using it

TODO how to generate intitial mesh?

```
from punyo_force_estimation.force_module.force_from_punyo import ForceFromPunyo

# Set up estimator with:
#   - List of reference RGB images
#   - List of reference point clouds
#   - List of reference pressure readings
#   - Mesh points
#   - Mesh triangles (connectivity)
#   - Mesh boundary (list of vertex indices)
force_estimator = ForceFromPunyo(ref_rgbs, ref_pcds, ref_pressures, points, triangles, boundary, rest_internal_force=None, precompile=True, verbose=True)

# Throw in observed rgb image, point cloud (from depth image), and presure (in pascals, default measure is in hectopascals)
force_estimator.update(frame['punyo_color'], pcd_points, frame['pressure'] * 100)
contact_forces = force_estimator.observed_force
displaced_points = force_estimator.current_points
```
