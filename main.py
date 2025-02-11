import numpy as np
from mayavi import mlab

from medial import compute_medial_axis
from surface import BSpline
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import directed_hausdorff

from qpsolvers import solve_qp

fig = mlab.figure(size=(800, 800))


def generate_initial_control_points(
    point_cloud, axis, percentile=5, smoothing=0, nu=18, nv=18, plot=False
):
    cpts = []
    ts = np.linspace(0, 1, nv)
    effective_mask = np.ones(point_cloud.shape[0], dtype=bool)

    for t in ts:
        q = np.array(axis(t))

        normal = np.array(axis(t, der=1))
        normal = normal / np.linalg.norm(normal)

        distance = np.abs(np.tensordot(point_cloud - q, normal, axes=(1, 0)))
        mask = distance < np.percentile(distance, percentile)
        filtered = point_cloud[mask]

        # fmt: off
        spline, *_ = splprep(filtered.T, task=-1, s=smoothing, per=0, t=np.linspace(0, 1, 4 + nu))
        if plot:
            mlab.plot3d(*splev(np.linspace(0, 1, nu), spline), tube_radius=0.005)
        # fmt: on

        cpts.append(np.array(spline[1]).T)

        if t == 0 or t == 1:
            effective_mask[mask] = False

    return np.array(cpts), point_cloud[effective_mask]


def hausdorff(surface, point_cloud):
    hausdorff_pm, *_ = directed_hausdorff(
        point_cloud, surface.mesh.reshape((surface.mesh.shape[0], -1)).T
    )
    hausdorff_mp, *_ = directed_hausdorff(
        point_cloud, surface.mesh.reshape((surface.mesh.shape[0], -1)).T
    )

    return max(hausdorff_mp, hausdorff_pm)


def optimize(point_cloud, surface):
    nu, nv, D = surface.cpts.shape
    cpts = surface.cpts[:].reshape((-1, D))

    G = []
    h = []

    for p in point_cloud:
        uv = surface.closest_point(p)

        if uv is None:
            continue

        J = surface.jacobian(uv)

        if J is None:
            continue

        w = surface.f(uv)
        n = surface.normal(uv)

        G.append(n.T @ J)
        h.append(np.dot(p - w, n))

    G = np.array(G)
    h = np.array(h)

    P = 2.0 * np.eye(3 * cpts.shape[0])
    q = np.zeros((3 * cpts.shape[0],))

    x = solve_qp(P, q, -G, -h, solver="cvxopt")
    cpts += x.reshape((nu * nv, D))
    cpts = cpts.reshape((nv, nu, D))
    return BSpline(cpts)


if __name__ == "__main__":
    p0 = np.array([0.0, 0.0, -1.0])
    p1 = np.array([0.0, 0.0, 1.0])

    with open("./points.p3d") as f:
        processor = lambda line: list(map(float, line.rstrip().split(",")))
        point_cloud = list(map(processor, f.readlines()))

        point_cloud = np.array(point_cloud)

    axis = compute_medial_axis(point_cloud, p0, p1)
    cpts, effective_cloud = generate_initial_control_points(
        point_cloud, axis, smoothing=5
    )
    surface = BSpline(cpts)

    # # Visualize point cloud and medial axis
    # mlab.plot3d(*axis(np.linspace(0, 1, 64)), tube_radius=0.005)
    mlab.points3d(*point_cloud.T, color=(0.0, 0.0, 1.0), scale_factor=0.005)
    # mlab.points3d(*np.vstack((p0, p1)).T, color=(1.0, 0.0, 0.0), scale_factor=0.025)

    # Optimize
    print(f"hausdorff(0): {hausdorff(surface, effective_cloud)}")
    mlab.mesh(*surface.mesh, color=(0.6, 0.65, 0.8))

    for i in range(1, 6):
        new_surface = optimize(effective_cloud, surface)
        print(f"hausdorff({i}): {hausdorff(new_surface, effective_cloud)}")
        surface = new_surface

    # Visualize constructed surface
    mlab.mesh(*new_surface.mesh, color=(0.8, 0.65, 0.6))

    mlab.show()
