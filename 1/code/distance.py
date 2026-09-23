"""
SYDE 572 - Assignment 1, Part 1
Shortest distance from a point (x0, y0) to a curve y = f(x).

Two numerical methods are implemented, both minimising the squared distance

    D(x) = (x - x0)^2 + (f(x) - y0)^2

  * Newton-Raphson on D'(x) = 0            (needs f, f', f'')
  * Golden-section search on D(x)          (needs only f and a bracket [a, b])

Every routine returns the full iteration history so the intermediate steps
can be tabulated and plotted.
"""
import math

PHI = (1 + math.sqrt(5)) / 2
RESPHI = 2 - PHI  # = 1/phi^2 ~ 0.381966


# ----------------------------------------------------------------------------
# Objective and its derivatives
# ----------------------------------------------------------------------------
def dist_sq(x, x0, y0, f):
    """Squared distance D(x) between (x0, y0) and the curve point (x, f(x))."""
    return (x - x0) ** 2 + (f(x) - y0) ** 2


def dist_sq_d1(x, x0, y0, f, df):
    """D'(x) = 2(x - x0) + 2(f(x) - y0) f'(x)."""
    return 2 * (x - x0) + 2 * (f(x) - y0) * df(x)


def dist_sq_d2(x, x0, y0, f, df, ddf):
    """D''(x) = 2 + 2 f'(x)^2 + 2(f(x) - y0) f''(x)."""
    return 2 + 2 * df(x) ** 2 + 2 * (f(x) - y0) * ddf(x)


# ----------------------------------------------------------------------------
# Function builders
# ----------------------------------------------------------------------------
def make_parabola(a, b, c):
    """Return f, f', f'' for the parabola y = a x^2 + b x + c."""
    f = lambda x: a * x * x + b * x + c
    df = lambda x: 2 * a * x + b
    ddf = lambda x: 2 * a
    return f, df, ddf


def numeric_derivatives(f, h=1e-5):
    """Central-difference f' and f'' for when the derivatives are not known
    in closed form (the 'slight modification' that lets Newton handle any f)."""
    df = lambda x: (f(x + h) - f(x - h)) / (2 * h)
    ddf = lambda x: (f(x + h) - 2 * f(x) + f(x - h)) / (h * h)
    return df, ddf


# ----------------------------------------------------------------------------
# Algorithm A: Newton-Raphson
# ----------------------------------------------------------------------------
def newton_distance(x0, y0, f, df=None, ddf=None, x_init=0.0, tol=1e-10,
                    max_iter=100, domain=(-math.inf, math.inf)):
    """Minimise D(x) by Newton-Raphson applied to D'(x) = 0.

    Plain Newton update:   x_{k+1} = x_k - D'(x_k) / D''(x_k)

    Two small safeguards make it work for any f (not just the convex case):
      1. If D''(x_k) <= 0 the pure Newton step would climb towards a maximum,
         so the always-positive Gauss-Newton curvature 2 + 2 f'(x)^2 is used.
      2. The step is halved until the new x lies inside `domain` and D does
         not increase (needed for e.g. ln x or sqrt x, which stop at x = 0).
    For the assignment parabola D'' = 12x^2 + 22 > 0, so neither safeguard
    ever triggers and the iteration is the textbook method.

    Returns (distance, x_star, history) where history is a list of dicts.
    """
    if df is None or ddf is None:
        df, ddf = numeric_derivatives(f)
    lo, hi = domain
    x = x_init
    history = []
    for k in range(max_iter):
        d1 = dist_sq_d1(x, x0, y0, f, df)
        d2 = dist_sq_d2(x, x0, y0, f, df, ddf)
        safeguarded = d2 <= 0
        curv = 2 + 2 * df(x) ** 2 if safeguarded else d2
        step = d1 / curv
        x_new = x - step
        D_old = dist_sq(x, x0, y0, f)
        halvings = 0
        while (not (lo < x_new < hi) or dist_sq(x_new, x0, y0, f) > D_old + 1e-15) \
                and halvings < 60:
            step /= 2
            x_new = x - step
            halvings += 1
        history.append(dict(k=k, x=x, D=D_old, d1=d1, d2=d2, x_next=x_new,
                            safeguarded=safeguarded, halvings=halvings))
        if abs(x_new - x) < tol:
            x = x_new
            break
        x = x_new
    return math.sqrt(dist_sq(x, x0, y0, f)), x, history


def newton_distance_multistart(x0, y0, f, df=None, ddf=None, starts=(0.0,),
                               **kw):
    """Run Newton from several starting points and keep the best result.
    D(x) can have several local minima (e.g. a point inside a parabola),
    and Newton only finds the one nearest its start."""
    best = None
    for s in starts:
        res = newton_distance(x0, y0, f, df, ddf, x_init=s, **kw)
        if best is None or res[0] < best[0]:
            best = res
    return best


# ----------------------------------------------------------------------------
# Algorithm B: Golden-section search
# ----------------------------------------------------------------------------
def default_bracket(x0, y0, f, domain=(-math.inf, math.inf)):
    """A bracket guaranteed to contain the global minimiser.

    The curve point (x0, f(x0)) is at distance r = |f(x0) - y0| from the
    query point, so the closest point satisfies |x* - x0| <= r.
    """
    try:
        r = abs(f(x0) - y0)
    except (ValueError, ZeroDivisionError):
        r = 10.0
    r = max(r, 1e-6)
    lo, hi = domain
    return max(x0 - r, lo), min(x0 + r, hi)


def golden_section_distance(x0, y0, f, a=None, b=None, tol=1e-8,
                            domain=(-math.inf, math.inf)):
    """Minimise D(x) on [a, b] by golden-section search (derivative free).

    Two interior points x1 < x2 split [a, b] in the golden ratio. The side
    with the larger D is discarded; the surviving interior point is reused,
    so each iteration costs one new function evaluation and shrinks the
    interval by 1/phi ~ 0.618.

    Returns (distance, x_star, history).
    """
    if a is None or b is None:
        a, b = default_bracket(x0, y0, f, domain)
    D = lambda x: dist_sq(x, x0, y0, f)
    x1 = a + RESPHI * (b - a)
    x2 = b - RESPHI * (b - a)
    f1, f2 = D(x1), D(x2)
    history = []
    k = 0
    while abs(b - a) > tol:
        history.append(dict(k=k, a=a, b=b, x1=x1, x2=x2, D1=f1, D2=f2))
        if f1 < f2:          # minimum lies in [a, x2]
            b, x2, f2 = x2, x1, f1
            x1 = a + RESPHI * (b - a)
            f1 = D(x1)
        else:                # minimum lies in [x1, b]
            a, x1, f1 = x1, x2, f2
            x2 = b - RESPHI * (b - a)
            f2 = D(x2)
        k += 1
    history.append(dict(k=k, a=a, b=b, x1=x1, x2=x2, D1=f1, D2=f2))
    x_best = (a + b) / 2
    return math.sqrt(D(x_best)), x_best, history


# ----------------------------------------------------------------------------
# Closed form for the assignment parabola  y = x^2 + 5,  point (x0, 0)
# ----------------------------------------------------------------------------
def analytic_parabola_x2_plus_5(x0):
    """D'(x) = 0  <=>  2x^3 + 11x - x0 = 0, a depressed cubic
    x^3 + p x + q = 0 with p = 11/2, q = -x0/2. Its discriminant is positive
    for every x0, so Cardano gives the single real root:
        x* = cbrt(x0/4 + s) + cbrt(x0/4 - s),  s = sqrt(x0^2/16 + (11/6)^3)
    """
    cbrt = lambda v: math.copysign(abs(v) ** (1 / 3), v)
    s = math.sqrt(x0 ** 2 / 16 + (11 / 6) ** 3)
    x = cbrt(x0 / 4 + s) + cbrt(x0 / 4 - s)
    return math.sqrt((x - x0) ** 2 + (x * x + 5) ** 2), x


if __name__ == "__main__":
    f, df, ddf = make_parabola(1, 0, 5)
    print(f"{'point':>10} | {'analytic':>22} | {'Newton':>22} | {'Golden':>22}")
    for p in [(0, 0), (-4, 0), (-8, 0), (2, 0), (6, 0)]:
        da, xa = analytic_parabola_x2_plus_5(p[0])
        dn, xn, hn = newton_distance(*p, f, df, ddf, x_init=p[0])
        dg, xg, hg = golden_section_distance(*p, f, -2, 2)
        print(f"{str(p):>10} | x={xa:+.6f} d={da:.6f} | x={xn:+.6f} d={dn:.6f} "
              f"({len(hn)} it) | x={xg:+.6f} d={dg:.6f} ({len(hg) - 1} it)")
