import numpy as np
from mayavi import mlab

from medial import compute_medial_axis
from surface import BSpline
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import directed_hausdorff

fig = mlab.figure(size=(800, 800))


def generate_initial_control_points(
    points, axis, percentile=5, smoothing=0, nu=18, nv=18
):
    cpts = []
    ts = np.linspace(0, 1, nv)

    for t in ts:
        q = np.array(axis(t))

        normal = np.array(axis(t, der=1))
        normal = normal / np.linalg.norm(normal)

        distance = np.abs(np.tensordot(points - q, normal, axes=(1, 0)))
        mask = distance < np.percentile(distance, percentile)
        filtered = points[mask]

        # fmt: off
        spline, *_ = splprep(filtered.T, task=-1, s=smoothing, per=0, t=np.linspace(0, 1, 4 + nu))
        mlab.plot3d(*splev(np.linspace(0, 1, nu), spline), tube_radius=0.005)
        # fmt: on

        cpts.append(np.array(spline[1]).T)

    return np.array(cpts)


def hausdorff(mesh, points):
    hausdorff_pm, *_ = directed_hausdorff(points, mesh.reshape((mesh.shape[0], -1)).T)
    hausdorff_mp, *_ = directed_hausdorff(points, mesh.reshape((mesh.shape[0], -1)).T)

    return max(hausdorff_mp, hausdorff_pm)


if __name__ == "__main__":
    p0 = np.array([0.0, 0.0, -1.0])
    p1 = np.array([0.0, 0.0, 1.0])

    with open("./points.p3d") as f:
        points = list(
            map(lambda line: list(map(float, line.rstrip().split(","))), f.readlines())
        )

        points = np.array(points)

    axis = compute_medial_axis(points, p0, p1)
    cpts = generate_initial_control_points(points, axis, smoothing=5)
    surface = BSpline(cpts)

    # Visualize point cloud and medial axis
    mlab.plot3d(*axis(np.linspace(0, 1, 64)), tube_radius=0.005)
    mlab.points3d(*points.T, color=(0.0, 0.0, 1.0), scale_factor=0.005)
    mlab.points3d(*np.vstack((p0, p1)).T, color=(1.0, 0.0, 0.0), scale_factor=0.025)

    # Visualize constructed surface
    mesh, _ = surface.generate_mesh(128)
    mlab.mesh(*mesh, color=(0.6, 0.65, 0.8))

    mlab.show()
