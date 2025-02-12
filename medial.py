from scipy.spatial import Delaunay, cKDTree
from scipy import interpolate
import networkx as nx
import numpy as np


def compute_medial_axis(points, p0, p1, r_thresh=np.inf):
    qhull = Delaunay(points)
    xyz_centers, edge_lst = compute_voronoi_vertices_and_edges(qhull, r_thresh)

    kdt = cKDTree(xyz_centers)
    _, idx0 = kdt.query(p0)
    _, idx1 = kdt.query(p1)

    # compute shortest weighted path
    edge_lengths = [
        np.linalg.norm(xyz_centers[e[0], :] - xyz_centers[e[1], :]) for e in edge_lst
    ]
    g = nx.Graph(
        (i, j, {"weight": dist}) for (i, j), dist in zip(edge_lst, edge_lengths)
    )
    path_s = nx.shortest_path(g, source=idx0, target=idx1, weight="weight")

    spline, *_ = interpolate.splprep(xyz_centers[np.array(path_s)].T)
    return lambda t, der=0: interpolate.splev(t, spline, der=der)


# https://stackoverflow.com/a/65777239
def compute_delaunay_tetra_circumcenters(dt):
    """
    Compute the centers of the circumscribing circle of each tetrahedron in the Delaunay triangulation.
    :param dt: the Delaunay triangulation
    :return: array of xyz points
    """
    simp_pts = dt.points[dt.simplices]
    # (n, 4, 3) array of tetrahedra points where simp_pts[i, j, :] holds the j'th 3D point (of four) of the i'th tetrahedron
    assert simp_pts.shape[1] == 4 and simp_pts.shape[2] == 3

    # finding the circumcenter (x, y, z) of a simplex defined by four points:
    # (x-x0)**2 + (y-y0)**2 + (z-z0)**2 = (x-x1)**2 + (y-y1)**2 + (z-z1)**2
    # (x-x0)**2 + (y-y0)**2 + (z-z0)**2 = (x-x2)**2 + (y-y2)**2 + (z-z2)**2
    # (x-x0)**2 + (y-y0)**2 + (z-z0)**2 = (x-x3)**2 + (y-y3)**2 + (z-z3)**2
    # becomes three linear equations (squares are canceled):
    # 2(x1-x0)*x + 2(y1-y0)*y + 2(z1-z0)*y = (x1**2 + y1**2 + z1**2) - (x0**2 + y0**2 + z0**2)
    # 2(x2-x0)*x + 2(y2-y0)*y + 2(z2-z0)*y = (x2**2 + y2**2 + z2**2) - (x0**2 + y0**2 + z0**2)
    # 2(x3-x0)*x + 2(y3-y0)*y + 2(z3-z0)*y = (x3**2 + y3**2 + z3**2) - (x0**2 + y0**2 + z0**2)

    # building the 3x3 matrix of the linear equations
    a = 2 * (simp_pts[:, 1, 0] - simp_pts[:, 0, 0])
    b = 2 * (simp_pts[:, 1, 1] - simp_pts[:, 0, 1])
    c = 2 * (simp_pts[:, 1, 2] - simp_pts[:, 0, 2])
    d = 2 * (simp_pts[:, 2, 0] - simp_pts[:, 0, 0])
    e = 2 * (simp_pts[:, 2, 1] - simp_pts[:, 0, 1])
    f = 2 * (simp_pts[:, 2, 2] - simp_pts[:, 0, 2])
    g = 2 * (simp_pts[:, 3, 0] - simp_pts[:, 0, 0])
    h = 2 * (simp_pts[:, 3, 1] - simp_pts[:, 0, 1])
    i = 2 * (simp_pts[:, 3, 2] - simp_pts[:, 0, 2])

    v1 = (simp_pts[:, 1, 0] ** 2 + simp_pts[:, 1, 1] ** 2 + simp_pts[:, 1, 2] ** 2) - (
        simp_pts[:, 0, 0] ** 2 + simp_pts[:, 0, 1] ** 2 + simp_pts[:, 0, 2] ** 2
    )
    v2 = (simp_pts[:, 2, 0] ** 2 + simp_pts[:, 2, 1] ** 2 + simp_pts[:, 2, 2] ** 2) - (
        simp_pts[:, 0, 0] ** 2 + simp_pts[:, 0, 1] ** 2 + simp_pts[:, 0, 2] ** 2
    )
    v3 = (simp_pts[:, 3, 0] ** 2 + simp_pts[:, 3, 1] ** 2 + simp_pts[:, 3, 2] ** 2) - (
        simp_pts[:, 0, 0] ** 2 + simp_pts[:, 0, 1] ** 2 + simp_pts[:, 0, 2] ** 2
    )

    # solve a 3x3 system by inversion (see https://en.wikipedia.org/wiki/Invertible_matrix#Inversion_of_3_%C3%97_3_matrices)
    A = e * i - f * h
    B = -(d * i - f * g)
    C = d * h - e * g
    D = -(b * i - c * h)
    E = a * i - c * g
    F = -(a * h - b * g)
    G = b * f - c * e
    H = -(a * f - c * d)
    I = a * e - b * d

    det = a * A + b * B + c * C

    # multiplying inv*[v1, v2, v3] to get solution point (x, y, z)
    x = (A * v1 + D * v2 + G * v3) / det
    y = (B * v1 + E * v2 + H * v3) / det
    z = (C * v1 + F * v2 + I * v3) / det

    return (np.vstack((x, y, z))).T


# https://stackoverflow.com/a/65777239
def compute_voronoi_vertices_and_edges(dt, r_thresh=np.inf):
    """
    Compute (finite) Voronoi edges and vertices of a set of points.
    :param points: input points.
    :param r_thresh: radius value for filtering out vertices corresponding to
    Delaunay tetrahedrons with large radii of circumscribing sphere (alpha-shape condition).
    :return: array of xyz Voronoi vertex points and an edge list.
    """
    xyz_centers = compute_delaunay_tetra_circumcenters(dt)

    # filtering tetrahedrons that have radius > thresh
    simp_pts_0 = dt.points[dt.simplices[:, 0]]
    radii = np.linalg.norm(xyz_centers - simp_pts_0, axis=1)
    is_in = radii < r_thresh

    # build an edge list from (filtered) tetrahedrons neighbor relations
    edge_lst = []
    for i in range(len(dt.neighbors)):
        if not is_in[i]:
            continue  # i is an outside tetra
        for j in dt.neighbors[i]:
            if j != -1 and is_in[j]:
                edge_lst.append((i, j))

    return xyz_centers, edge_lst


# edges = []
# for tet, center in zip(qhull.simplices, circumcenters):
#     skip = False
#     for pidx in tet:
#         if (skip := (np.linalg.norm(points[pidx] - center) > 1.)):
#             break

#     if skip:
#         continue

#     edges += list(combinations(tet, 2))

# src = mlab.pipeline.scalar_scatter(*points.T, np.ones(points.shape[0]))
# src.mlab_source.dataset.lines = np.array(edges[:])
# src.update()

# lines = mlab.pipeline.stripper(src)
# mlab.pipeline.surface(src, color=(0., 1., 0.), line_width=1, opacity=0.4)
