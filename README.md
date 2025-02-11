
## Mandrel B-Spline Surface Fitting

Given a point cloud $\mathbf{P} = \{P_i\}_{i=1}^{N} \subset \mathbb{R^3}$ describing some mandrel surface $S$ we want to find an approximate surface $\bar{S}$ by fitting bicubic B-spline patches.

### Method

Let $n_u$ and $n_v$ be the number of control points along each parameter, with $u$ as the polar axis and $v$ the longitudinal axis.

1. Fit a curve $m(t)$ to the medial axis of the point cloud
    - Use method by [Iddo Hanniel on StackOverflow](https://stackoverflow.com/a/65777239) to find the medial axis
    - Fit a cubic B-spline curve to the medial axis
2. Find an initalization for control points
    - Let `cpts` be an empty $n_v \times n_u$ array
    - Let $`\mathbf{T} = \left\{\frac{k}{n_v - 1} {\Large|}\ k\in\{0, \dots, n_v - 1\}\right\} \subset [0, 1]`$
    - For each $t_i \in \mathbf{T}$
        - Let $\mathbf{C}_i$ denote the plane containing point $q_i = m(t_i)$ with normal $n_i = m'(t_i)$
        - Compute distances for each point in the point cloud to the plane
            - For $p \in \mathbb{R}^3$, $\mathrm{dist}(p, \mathbf{C}_i) = |(p - q_i) \cdot \hat{n}_i|$
        - Keep all points in or below the $k$-th percentile (by default $k=5$)
        - Let $\mathbf{P}_{\mathbf{C}_i} \subset \mathbf{P}$ be the projection of the accepted points onto $\mathbf{C}_i$
            - $\mathrm{proj}_{\mathbf{C}_i}(p) = p - \mathrm{dist}(p, \mathbf{C}_i)\cdot\hat{n}_i$
        - Let $r_i(t)$ be a B-spline curve fit to the points $\mathbf{P}_{\mathbf{C}_i}$ with a uniform knot vector containing $4 + n_u$ knots
        - Let `cpts[i]` be the control points of $r_i$ to `cpts`
    - Let $\bar{S}$ be the surfaced formed by `cpts`
3. (TODO) Optimize control points
    - Let $x$ denote the control points, then want to find an update $\Delta x$ to the control points that brings $\bar{S}$ closer to the point cloud
    - For each point $p_i \in \mathbf{P}$, find the closest point $s_i \in \mathbb{R}^3$ on $\bar{S}$ with parameters $\bar{p}_i = (u_i, v_i) \in \mathbb{R}^2$ such that $\bar{S}(\bar{p}_i) = p_i$
        - Use Newton's method to minimize $\|\bar{S}(u, v) - p_i\|_2$
    - Let $A$ be an $`N \times 3\cdot n_u \cdot n_v`$ matrix
        - Let $N_{\bar{S}}(u, v)$ be the surface normal and $J_{\bar{S}}(u, v)$ be the $`3 \times3\cdot n_u \cdot n_v `$ Jacobian
        - Each row is defined as $`(N_{\bar{S}}(\bar{p}_i))^TJ_{\bar{S}}(\bar{p}_i)`$
    - Let $b$ be a row vector of $N$ entries
        - Each row is defined as $`(p_i - \bar{S}(\bar{p}_i)) \cdot N_{\bar{S}}(\bar{p}_i)`$
    - Find $\Delta x$ that minimizes $\Delta x^T \Delta x$ such that $A\Delta x \geq b$
    - Repeat