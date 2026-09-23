"""
SYDE 572 - Assignment 1, Part 2
Least-squares fitting of a line and a parabola by minimising the MSE.

A model is linear in its parameters theta = (theta_0, ..., theta_{p-1}):

    y_hat(x) = sum_j theta_j * phi_j(x)

    line:      phi = (x, 1)          theta = (m, b)
    parabola:  phi = (x^2, x, 1)     theta = (a, b, c)

    MSE(theta) = (1/n) sum_i (y_hat(x_i) - y_i)^2

Three solvers are provided:
  * analytic_fit           - normal equations  (Phi^T Phi) theta = Phi^T y
  * coordinate_newton_fit  - Newton-Raphson one parameter at a time, holding
                             the others fixed (the procedure the assignment asks for)
  * full_newton_fit        - multivariate Newton with the full Hessian
"""

LINE = dict(name="line", params=("m", "b"),
            basis=(lambda x: x, lambda x: 1.0))
PARABOLA = dict(name="parabola", params=("a", "b", "c"),
                basis=(lambda x: x * x, lambda x: x, lambda x: 1.0))


def predict(model, theta, x):
    return sum(t * phi(x) for t, phi in zip(theta, model["basis"]))


def mse(model, theta, xs, ys):
    n = len(xs)
    return sum((predict(model, theta, x) - y) ** 2 for x, y in zip(xs, ys)) / n


def gradient(model, theta, xs, ys):
    """dMSE/dtheta_j = (2/n) sum_i r_i phi_j(x_i),  r_i = y_hat_i - y_i."""
    n = len(xs)
    r = [predict(model, theta, x) - y for x, y in zip(xs, ys)]
    return [2 / n * sum(ri * phi(x) for ri, x in zip(r, xs)) for phi in model["basis"]]


def hessian(model, xs):
    """d2MSE/dtheta_j dtheta_k = (2/n) sum_i phi_j(x_i) phi_k(x_i)  (constant)."""
    n = len(xs)
    B = model["basis"]
    return [[2 / n * sum(pj(x) * pk(x) for x in xs) for pk in B] for pj in B]


def solve_linear(A, rhs):
    """Gaussian elimination with partial pivoting (small dense systems)."""
    n = len(A)
    M = [list(map(float, A[i])) + [float(rhs[i])] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        for r in range(col + 1, n):
            factor = M[r][col] / M[col][col]
            for c in range(col, n + 1):
                M[r][c] -= factor * M[col][c]
    sol = [0.0] * n
    for r in range(n - 1, -1, -1):
        sol[r] = (M[r][n] - sum(M[r][c] * sol[c] for c in range(r + 1, n))) / M[r][r]
    return sol


def analytic_fit(model, xs, ys):
    """Set grad MSE = 0  ->  normal equations (Phi^T Phi) theta = Phi^T y."""
    B = model["basis"]
    A = [[sum(pj(x) * pk(x) for x in xs) for pk in B] for pj in B]
    rhs = [sum(pj(x) * y for x, y in zip(xs, ys)) for pj in B]
    return solve_linear(A, rhs), A, rhs


def coordinate_newton_fit(model, xs, ys, theta0=None, tol=1e-10, max_sweeps=500):
    """Newton-Raphson one parameter at a time.

    In each sweep, for j = 0..p-1 (all other parameters held at their
    current values):
        theta_j <- theta_j - g_j(theta) / H_jj
    Because the MSE is quadratic in each parameter, a single Newton step
    lands exactly on the minimum along that coordinate.

    Returns (theta, history). history holds the parameters after every
    single-coordinate update (index 0 = the starting guess).
    """
    p = len(model["params"])
    theta = list(theta0) if theta0 is not None else [0.0] * p
    H = hessian(model, xs)
    history = [dict(sweep=0, coord=None, theta=list(theta), mse=mse(model, theta, xs, ys))]
    for sweep in range(1, max_sweeps + 1):
        old = list(theta)
        for j in range(p):
            g = gradient(model, theta, xs, ys)[j]
            theta[j] -= g / H[j][j]
            history.append(dict(sweep=sweep, coord=j, theta=list(theta),
                                mse=mse(model, theta, xs, ys), grad=g, hess=H[j][j]))
        if max(abs(a - b) for a, b in zip(theta, old)) < tol:
            break
    return theta, history


def full_newton_fit(model, xs, ys, theta0=None, tol=1e-12, max_iter=20):
    """Multivariate Newton: theta <- theta - H^{-1} grad.
    For a quadratic objective this converges in exactly one step."""
    p = len(model["params"])
    theta = list(theta0) if theta0 is not None else [0.0] * p
    H = hessian(model, xs)
    history = [dict(theta=list(theta), mse=mse(model, theta, xs, ys))]
    for _ in range(max_iter):
        g = gradient(model, theta, xs, ys)
        step = solve_linear(H, g)
        theta = [t - s for t, s in zip(theta, step)]
        history.append(dict(theta=list(theta), mse=mse(model, theta, xs, ys)))
        if max(abs(s) for s in step) < tol:
            break
    return theta, history


if __name__ == "__main__":
    xs = [0, 2, 1, 3]
    ys = [0.5, 3.5, 1.5, 7.5]
    for model in (LINE, PARABOLA):
        th_a, _, _ = analytic_fit(model, xs, ys)
        th_c, hist_c = coordinate_newton_fit(model, xs, ys)
        th_n, hist_n = full_newton_fit(model, xs, ys)
        fmt = lambda th: ", ".join(f"{n}={v:+.6f}" for n, v in zip(model["params"], th))
        print(f"--- {model['name']} ---")
        print(f"analytic         : {fmt(th_a)}  MSE={mse(model, th_a, xs, ys):.6f}")
        print(f"coordinate Newton: {fmt(th_c)}  MSE={mse(model, th_c, xs, ys):.6f}"
              f"  ({hist_c[-1]['sweep']} sweeps)")
        print(f"full Newton      : {fmt(th_n)}  MSE={mse(model, th_n, xs, ys):.6f}"
              f"  ({len(hist_n) - 1} steps)")
