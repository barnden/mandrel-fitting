import numpy as np
from scipy.spatial import cKDTree


class Surface:
    eps = 1e-5

    def __init__(self):
        self.mesh, _ = self.generate_mesh(128)
        grid3d = []
        grid2d = []

        parametric_domain = np.array(
            np.meshgrid(
                np.linspace(self.u_min, self.u_max, 256),
                np.linspace(self.v_min, self.v_max, 256),
            )
        )

        for i in range(256):
            for j in range(256):
                p = parametric_domain[:, i, j]
                w = self.f(p)

                grid3d.append(w)
                grid2d.append(p)

        self.grid3d = np.array(grid3d)
        self.grid2d = np.array(grid2d)
        self.kd = cKDTree(self.grid3d)

    def f(self, p):
        return np.zeros((3,))

    def f_u(self, p):
        delta = np.array([self.eps, 0.0])

        return (self.f(p + delta) - self.f(p - delta)) / (2.0 * self.eps)

    def f_v(self, p):
        delta = np.array([0.0, self.eps])

        return (self.f(p + delta) - self.f(p - delta)) / (2.0 * self.eps)

    def f_uu(self, p):
        delta = np.array([2.0 * self.eps, 0.0])

        return (self.f(p + delta) + self.f(p - delta) - 2.0 * self.f(p)) / (
            4.0 * self.eps * self.eps
        )

    def f_vv(self, p):
        delta = np.array([0.0, 2.0 * self.eps])

        return (self.f(p + delta) + self.f(p - delta) - 2.0 * self.f(p)) / (
            4.0 * self.eps * self.eps
        )

    def f_uv(self, p):
        # fmt: off
        return (
            self.f(p + np.array([ self.eps,  self.eps]))
          + self.f(p + np.array([-self.eps, -self.eps]))
          - self.f(p + np.array([-self.eps,  self.eps]))
          - self.f(p + np.array([ self.eps, -self.eps]))
        ) / (4. * self.eps * self.eps)
        # fmt: on

    def normal(self, p):
        u, v = p

        du = (
            np.array(self.f((u + self.eps, v))) - np.array(self.f((u - self.eps, v)))
        ) / (2.0 * self.eps)
        dv = (
            np.array(self.f((u, v + self.eps))) - np.array(self.f((u, v - self.eps)))
        ) / (2.0 * self.eps)

        normal = np.cross(du, dv, axis=0)
        normal /= np.linalg.norm(normal, axis=0)

        return normal

    def generate_mesh(self, N=32):
        parametric_domain = np.array(
            np.meshgrid(
                np.linspace(self.u_min, self.u_max, N),
                np.linspace(self.v_min, self.v_max, N),
            )
        )

        mesh = np.zeros((3, N, N))

        for i in range(N):
            for j in range(N):
                mesh[:, i, j] = self.f(parametric_domain[:, i, j])

        return mesh, parametric_domain

    def closest_point(self, p, guess=None):
        if guess is None:
            _, idx = self.kd.query(p)
            guess = self.grid2d[idx]

        xk = guess
        for _ in range(1000):
            f_ = self.f(xk)

            if f_ is None:
                return None

            fp = f_ - p
            fu = self.f_u(xk)
            fv = self.f_v(xk)
            fuu = self.f_uu(xk)
            fuv = self.f_uv(xk)
            fvv = self.f_vv(xk)

            # fmt: off
            J = np.array([np.dot(fu, fp), np.dot(fv, fp)])
            H = np.array([
                [np.dot(fuu, fp) + np.dot(fu, fu), np.dot(fuv, fp) + np.dot(fu, fv)],
                [np.dot(fuv, fp) + np.dot(fu, fv), np.dot(fvv, fp) + np.dot(fv, fv)],
            ])
            # fmt: on

            dx = np.linalg.inv(H) @ J

            if np.linalg.norm(dx) < 1e-5:
                break

            xk -= dx

        return xk


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
        self.cpts = points
        self.nu = len(points[0])
        self.nv = len(points)

        self.u_max = self.nu
        self.v_max = self.nv - 1
        super().__init__()

    def get_uv(self, u, v):
        while u < 0.0:
            u += self.nu

        iu = int(u)
        iv = int(v)

        if iv == self.nv - 1:
            iv -= 1

        if iv == 1:
            iv += 1

        u -= iu
        v -= iv

        iu %= self.nu

        return (u, v, iu, iv)

    def f(self, p):
        u, v, iu, iv = self.get_uv(*p)

        if iv < 2 or iv >= self.nv:
            return None

        U = np.array([u**3, u**2, u, 1.0]) @ self.basis
        V = np.array([v**3, v**2, v, 1.0]) @ self.basis

        iu += int(self.u_max)

        # fmt: off
        P = np.array([
            V @ np.array([
                    self.cpts[iv - 2][(iu - 2) % self.nu],
                    self.cpts[iv - 1][(iu - 2) % self.nu],
                    self.cpts[iv - 0][(iu - 2) % self.nu],
                    self.cpts[iv + 1][(iu - 2) % self.nu],
            ]),
            V @ np.array([
                    self.cpts[iv - 2][(iu - 1) % self.nu],
                    self.cpts[iv - 1][(iu - 1) % self.nu],
                    self.cpts[iv - 0][(iu - 1) % self.nu],
                    self.cpts[iv + 1][(iu - 1) % self.nu],
            ]),
            V @ np.array([
                    self.cpts[iv - 2][(iu + 0) % self.nu],
                    self.cpts[iv - 1][(iu + 0) % self.nu],
                    self.cpts[iv - 0][(iu + 0) % self.nu],
                    self.cpts[iv + 1][(iu + 0) % self.nu],
            ]),
            V @ np.array([
                    self.cpts[iv - 2][(iu + 1) % self.nu],
                    self.cpts[iv - 1][(iu + 1) % self.nu],
                    self.cpts[iv - 0][(iu + 1) % self.nu],
                    self.cpts[iv + 1][(iu + 1) % self.nu],
            ]),
        ])
        # fmt: on

        return U @ P

    def _get_index(self, v, u):
        u %= self.nu

        while u < 0:
            u += self.nu

        return 3 * (v * self.nu + u)

    def jacobian(self, p):
        J = np.zeros((3, 3 * self.nv * self.nu))
        u, v, iu, iv = self.get_uv(*p)

        if iv < 2 or iv >= self.nv:
            return None

        U = np.array([u**3, u**2, u, 1.0]) @ self.basis
        V = np.array([v**3, v**2, v, 1.0]) @ self.basis

        for l in range(4):
            for m in range(4):
                offset = self._get_index(iv + (l - 2), iu + (m - 2))
                J[:, offset : offset + 3] = np.eye(3) * V[l] * U[m]

        return J
