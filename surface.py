import numpy as np


class Surface:
    eps = 1e-5

    def __init__(self):
        pass

    def normal(self, u, v):
        du = (np.array(self.f(u + self.eps, v)) - np.array(self.f(u - self.eps, v))) / (
            2.0 * self.eps
        )
        dv = (np.array(self.f(u, v + self.eps)) - np.array(self.f(u, v - self.eps))) / (
            2.0 * self.eps
        )

        normal = np.cross(du, dv, axis=0)
        normal /= np.linalg.norm(normal, axis=0)

        return normal

    def generate_mesh(self, N=32):
        parametric_domain = np.meshgrid(
            np.linspace(self.u_min, self.u_max, N),
            np.linspace(self.v_min, self.v_max, N),
        )

        return self.f(*parametric_domain)


class BSpline(Surface):
    u_min = 0
    u_max = 0
    v_min = 2
    v_max = 2

    # fmt: off
    basis = (
        1 / 6 * np.array([
            [-1.0,  3.0, -3.0,  1.0],
            [ 3.0, -6.0,  3.0,  0.0],
            [-3.0,  0.0,  3.0,  0.0],
            [ 1.0,  4.0,  1.0,  0.0],
        ])
    )
    # fmt: on

    def __init__(self, points):
        self.points = points
        self.v_max = len(points) - 1
        self.u_max = len(points[0])

    def get_uv(self, u, v):
        while u < 0.0:
            u += self.u_max

        iu = int(u)
        iv = int(v)

        if iv == int(self.v_max):
            iv -= 1

        if iv == 1:
            iv += 1

        u -= iu
        v -= iv

        iu %= int(self.u_max)

        return (u, v, iu, iv)

    def f(self, u, v):
        u, v, iu, iv = self.get_uv(u, v)

        U = np.array([u**3, u**2, u, 1.0]) @ self.basis
        V = np.array([v**3, v**2, v, 1.0]) @ self.basis

        iu += int(self.u_max)

        # fmt: off
        P = np.array([
            V @ np.array([
                    self.points[iv - 2][(iu - 2) % int(self.u_max)],
                    self.points[iv - 1][(iu - 2) % int(self.u_max)],
                    self.points[iv - 0][(iu - 2) % int(self.u_max)],
                    self.points[iv + 1][(iu - 2) % int(self.u_max)],
            ]),
            V @ np.array([
                    self.points[iv - 2][(iu - 1) % int(self.u_max)],
                    self.points[iv - 1][(iu - 1) % int(self.u_max)],
                    self.points[iv - 0][(iu - 1) % int(self.u_max)],
                    self.points[iv + 1][(iu - 1) % int(self.u_max)],
            ]),
            V @ np.array([
                    self.points[iv - 2][(iu + 0) % int(self.u_max)],
                    self.points[iv - 1][(iu + 0) % int(self.u_max)],
                    self.points[iv - 0][(iu + 0) % int(self.u_max)],
                    self.points[iv + 1][(iu + 0) % int(self.u_max)],
            ]),
            V @ np.array([
                    self.points[iv - 2][(iu + 1) % int(self.u_max)],
                    self.points[iv - 1][(iu + 1) % int(self.u_max)],
                    self.points[iv - 0][(iu + 1) % int(self.u_max)],
                    self.points[iv + 1][(iu + 1) % int(self.u_max)],
            ]),
        ])
        # fmt: on

        return U @ P

    def generate_mesh(self, N=32):
        parametric_space = np.array(
            np.meshgrid(
                np.linspace(self.u_min, self.u_max, N),
                np.linspace(self.v_min, self.v_max, N),
            )
        )

        mesh = np.zeros((3, N, N))
        normals = np.zeros((3, N, N))

        for i in range(N):
            for j in range(N):
                mesh[:, i, j] = self.f(*parametric_space[:, i, j])
                normals[:, i, j] = self.normal(*parametric_space[:, i, j])

        return mesh, normals
