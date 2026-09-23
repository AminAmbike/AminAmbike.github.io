"""
SYDE 572 - Assignment 1
Generates every figure on the project page (saved to ../media/) and a
results.json with all the numbers quoted in the write-up.

    python3 make_figures.py
"""
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from distance import (make_parabola, newton_distance, newton_distance_multistart,
                      golden_section_distance, analytic_parabola_x2_plus_5,
                      dist_sq, dist_sq_d1, default_bracket, numeric_derivatives)
from fitting import (LINE, PARABOLA, analytic_fit, coordinate_newton_fit,
                     full_newton_fit, mse, predict)

HERE = os.path.dirname(os.path.abspath(__file__))
MEDIA = os.path.join(HERE, "..", "media")
os.makedirs(MEDIA, exist_ok=True)

# ---------------------------------------------------------------- style ----
INK = "#1d2330"
MUTED = "#6b6f7a"
GRID = "#e6e4df"
SURFACE = "#fcfcfb"
BLUE = "#2a78d6"      # Newton
VIOLET = "#4a3aa7"    # golden section
ORANGE = "#eb6834"    # query point / data
AQUA = "#1baf7a"      # solution
RED = "#e34948"
BLUE_RAMP = LinearSegmentedColormap.from_list("blue", ["#bcd6f5", "#2a78d6", "#0d3a73"])
VIOLET_RAMP = LinearSegmentedColormap.from_list("violet", ["#d4cff3", "#7a6cd6", "#2a1f75"])

plt.rcParams.update({
    "font.family": "Avenir Next",
    "mathtext.fontset": "stixsans",
    "font.size": 10.5,
    "axes.titlesize": 12,
    "axes.titleweight": "semibold",
    "axes.titlelocation": "left",
    "axes.titlepad": 10,
    "axes.labelcolor": INK,
    "axes.edgecolor": "#c9c6bf",
    "axes.linewidth": 0.8,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "legend.frameon": False,
    "legend.fontsize": 9.5,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.facecolor": SURFACE,
})


def save(fig, name):
    fig.savefig(os.path.join(MEDIA, name))
    plt.close(fig)
    print("saved", name)


RESULTS = {}

# ===================================================== PART 1 : distance ====
f, df, ddf = make_parabola(1, 0, 5)
POINTS = [(0, 0), (-4, 0), (-8, 0), (2, 0), (6, 0)]
GS_BRACKET = (-2.0, 2.0)   # |x*| <= |x0|/11 < 1 for every point, so [-2, 2] brackets all
P1 = []

for (x0, y0) in POINTS:
    da, xa = analytic_parabola_x2_plus_5(x0)
    s = math.sqrt(x0 ** 2 / 16 + (11 / 6) ** 3)
    dn, xn, hn = newton_distance(x0, y0, f, df, ddf, x_init=float(x0), tol=1e-12)
    dg, xg, hg = golden_section_distance(x0, y0, f, *GS_BRACKET, tol=1e-8)
    P1.append(dict(point=[x0, y0], analytic=dict(x=xa, y=f(xa), d=da, s=s,
                   u=x0 / 4 + s, v=x0 / 4 - s,
                   perp=(xa - x0) + (f(xa) - y0) * df(xa)),
                   newton=dict(x=xn, d=dn, iters=len(hn), history=hn),
                   golden=dict(x=xg, d=dg, iters=len(hg) - 1, history=hg)))
RESULTS["part1"] = P1


# --- Figure 1: overview of the five shortest segments ----------------------
fig, ax = plt.subplots(figsize=(9, 5.6))
xx = np.linspace(-3.2, 3.2, 400)
ax.plot(xx, f(xx), color=INK, lw=2.2, zorder=3)
ax.text(2.25, 10.5, r"$y = x^2 + 5$", fontsize=13, color=INK, ha="left")
for r in P1:
    x0, y0 = r["point"]
    xs_, ys_ = r["analytic"]["x"], r["analytic"]["y"]
    ax.plot([x0, xs_], [y0, ys_], color=AQUA, lw=1.8, zorder=2)
    ax.scatter([xs_], [ys_], s=46, color=AQUA, edgecolor="white", lw=1.5, zorder=5)
    ax.scatter([x0], [y0], s=70, color=ORANGE, edgecolor="white", lw=1.5, zorder=5)
    ax.annotate(f"({x0}, {y0})", (x0, y0), xytext=(0, -16), textcoords="offset points",
                ha="center", fontsize=9.5, color=MUTED)
    ax.annotate(f"d = {r['analytic']['d']:.4f}", (x0, y0), xytext=(0, -31),
                textcoords="offset points", ha="center", fontsize=9.5, color=INK,
                fontweight="semibold")
ax.set_xlim(-9, 7)
ax.set_ylim(-3.2, 16)
ax.set_aspect("equal")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Shortest distance from each query point to y = x² + 5")
ax.legend(handles=[
    Line2D([], [], color=INK, lw=2.2, label="curve y = x² + 5"),
    Line2D([], [], marker="o", ls="", color=ORANGE, ms=8, label=r"query point $(x_0, y_0)$"),
    Line2D([], [], marker="o", ls="", color=AQUA, ms=7, label="closest point on curve"),
    Line2D([], [], color=AQUA, lw=1.8, label=r"shortest segment ($\perp$ tangent)")],
    loc="upper left", ncol=2)
save(fig, "p1_overview.png")


# --- Figure 2..6: per-point iteration panels ---------------------------------
def per_point_figure(r, fname):
    x0, y0 = r["point"]
    xs_ = r["analytic"]["x"]
    hn = r["newton"]["history"]
    hg = r["golden"]["history"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw=dict(wspace=0.28))

    # (a) geometry with Newton iterates
    ax = axes[0]
    iters = [h["x"] for h in hn] + [hn[-1]["x_next"]]
    lo = min(min(iters), x0, xs_) - 1
    hi = max(max(iters), x0, xs_) + 1
    xx = np.linspace(lo, hi, 400)
    ax.plot(xx, f(xx), color=INK, lw=2, zorder=3)
    n = len(iters)
    for k, xk in enumerate(iters):
        c = BLUE_RAMP(k / max(n - 1, 1))
        ax.plot([x0, xk], [y0, f(xk)], color=c, lw=1.1, ls="--", zorder=2)
        ax.scatter([xk], [f(xk)], color=c, s=34, edgecolor="white", lw=1, zorder=4)
        if k < 4:
            ax.annotate(f"$x_{k}$", (xk, f(xk)), xytext=(7, -3),
                        textcoords="offset points", fontsize=10, color=c)
    ax.plot([x0, xs_], [y0, f(xs_)], color=AQUA, lw=2.2, zorder=3)
    ax.scatter([xs_], [f(xs_)], s=60, color=AQUA, edgecolor="white", lw=1.5, zorder=6)
    ax.scatter([x0], [y0], s=80, color=ORANGE, edgecolor="white", lw=1.5, zorder=6)
    ax.annotate(f"({x0}, {y0})", (x0, y0), xytext=(0, 10), textcoords="offset points",
                ha="center", color=ORANGE, fontsize=10, fontweight="semibold",
                bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none"))
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("(a) Newton iterates on the curve")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.text(0.98, 0.03, f"x* = {xs_:.6f}\nd* = {r['analytic']['d']:.6f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GRID))

    # (b) Newton on D'(x): tangent lines to D' hitting zero
    ax = axes[1]
    xx = np.linspace(lo, hi, 400)
    Dp = [dist_sq_d1(x, x0, y0, f, df) for x in xx]
    ax.plot(xx, Dp, color=INK, lw=2, zorder=3, label=r"$D'(x) = 2(x-x_0) + 2(f(x)-y_0)\,f'(x)$")
    ax.axhline(0, color=MUTED, lw=0.9)
    for k, h in enumerate(hn[:6]):
        c = BLUE_RAMP(k / max(len(hn) - 1, 1))
        ax.plot([h["x"], h["x_next"]], [h["d1"], 0], color=c, lw=1.4, zorder=4)
        ax.plot([h["x"], h["x"]], [0, h["d1"]], color=c, lw=0.9, ls=":", zorder=2)
        ax.scatter([h["x"]], [h["d1"]], color=c, s=30, edgecolor="white", lw=1, zorder=5)
        if k < 3:
            ax.annotate(f"$x_{k}$", (h["x"], 0), xytext=(0, -14 if h["d1"] > 0 else 6),
                        textcoords="offset points", ha="center", fontsize=10, color=c)
    ax.scatter([xs_], [0], s=60, color=AQUA, edgecolor="white", lw=1.5, zorder=6)
    ax.set_title(r"(b) Newton: tangents of $D'(x)$ $\to$ root")
    ax.set_xlabel("x"); ax.set_ylabel(r"$D'(x) = 2(x-x_0) + 2(f(x)-y_0)\,f'(x)$")

    # (c) golden-section bracket shrinking over D(x)
    ax = axes[2]
    a0, b0 = hg[0]["a"], hg[0]["b"]
    xx = np.linspace(a0, b0, 400)
    Dv = np.array([dist_sq(x, x0, y0, f) for x in xx])
    ax.plot(xx, Dv, color=INK, lw=2, zorder=3, label=r"$D(x) = (x-x_0)^2 + (f(x)-y_0)^2$")
    ymin, ymax = Dv.min(), Dv.max()
    span = ymax - ymin if ymax > ymin else 1.0
    shown = hg[:10]
    for k, h in enumerate(shown):
        c = VIOLET_RAMP(k / (len(shown) - 1))
        yk = ymax + span * (0.10 + 0.075 * k)
        ax.plot([h["a"], h["b"]], [yk, yk], color=c, lw=3, solid_capstyle="round")
        ax.scatter([h["x1"], h["x2"]], [yk, yk], color="white", edgecolor=c, s=18, zorder=5, lw=1.2)
        ax.text(b0 + (b0 - a0) * 0.02, yk, f"k={k}", fontsize=8, color=c, va="center")
    for h in shown[:3]:
        c = VIOLET
        ax.scatter([h["x1"], h["x2"]], [h["D1"], h["D2"]], color=c, s=22, zorder=5,
                   edgecolor="white", lw=0.8)
    ax.axvline(xs_, color=AQUA, lw=1.5, ls="--", zorder=1)
    ax.set_xlim(a0 - 0.05 * (b0 - a0), b0 + 0.14 * (b0 - a0))
    ax.set_ylim(ymin - 0.05 * span, ymax + span * (0.2 + 0.075 * len(shown)))
    ax.set_title(f"(c) Golden section: bracket [a, b] per iteration")
    ax.set_xlabel("x"); ax.set_ylabel(r"$D(x) = (x-x_0)^2 + (f(x)-y_0)^2$")

    fig.suptitle(f"Point ({x0}, {y0})  to  curve y = x² + 5     "
                 f"Newton: {r['newton']['iters']} iterations  ·  "
                 f"Golden section: {r['golden']['iters']} iterations",
                 x=0.07, ha="left", fontsize=13, fontweight="semibold", y=1.03)
    save(fig, fname)


for r in P1:
    x0 = r["point"][0]
    per_point_figure(r, f"p1_point_{'m' if x0 < 0 else ''}{abs(x0)}.png")


# --- Figure 7: convergence of both methods ------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
cats = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e34948"]
for r, c in zip(P1, cats):
    xs_ = r["analytic"]["x"]
    lab = f"({r['point'][0]}, {r['point'][1]})"
    e_n = [max(abs(h["x_next"] - xs_), 1e-17) for h in r["newton"]["history"]]
    axes[0].semilogy(range(1, len(e_n) + 1), e_n, "-o", color=c, ms=5, lw=1.8, label=lab)
    if r["point"][0] == 0:
        continue   # bracket is symmetric about x* = 0, so the midpoint error is exactly 0
    e_g = [max(abs((h["a"] + h["b"]) / 2 - xs_), 1e-17) for h in r["golden"]["history"]]
    axes[1].semilogy(range(len(e_g)), e_g, "-", color=c, lw=1.3, alpha=0.9, label=lab)
axes[0].set_title("Newton-Raphson: quadratic convergence")
axes[1].set_title("Golden section: linear rate 0.618 per iteration")
axes[0].set_xlabel("iteration k"); axes[1].set_xlabel("iteration k")
axes[0].set_ylabel(r"error $|x_k - x^*|$")
k = np.arange(0, 45)
axes[1].semilogy(k, 4 * 0.618 ** k / 2, color=INK, ls="--", lw=1.6,
                 label=r"bound $(b_0-a_0)/2 \cdot 0.618^k$")
axes[1].set_ylim(1e-11, 20)
axes[1].set_ylabel("|midpoint − x*|")
axes[1].legend(title="query point", loc="upper right", fontsize=8.5)
axes[0].legend(title="query point", loc="lower left", fontsize=8.5)
axes[0].set_ylim(1e-17, 50)
save(fig, "p1_convergence.png")


# --- Figure 8: other functions and points -------------------------------------
sqrt_ = lambda x: math.sqrt(x)
EXAMPLES = [
    dict(label="y = 2x² − 3x + 1", tex=r"$y = 2x^2 - 3x + 1$", pt=(2.0, -1.0),
         fs=make_parabola(2, -3, 1), dom=(-math.inf, math.inf), view=(-1.2, 3.2),
         kind="parabola"),
    dict(label="y = −0.5x² + 2x + 3", tex=r"$y = -0.5x^2 + 2x + 3$", pt=(5.0, 6.0),
         fs=make_parabola(-0.5, 2, 3), dom=(-math.inf, math.inf), view=(-1.5, 7),
         kind="parabola"),
    dict(label="y = −x² + 4  (point inside)", tex=r"$y = -x^2 + 4$", pt=(0.0, 1.0),
         fs=make_parabola(-1, 0, 4), dom=(-math.inf, math.inf), view=(-2.6, 2.6),
         kind="parabola", starts=(-1.0, 0.0, 1.0)),
    dict(label="y = eˣ", tex=r"$y = e^{x}$", pt=(0.0, 0.0),
         fs=(math.exp, math.exp, math.exp), dom=(-math.inf, math.inf), view=(-2.5, 1.5),
         kind="exponential"),
    dict(label="y = ln x", tex=r"$y = \ln x$", pt=(0.0, 1.5),
         fs=(math.log, lambda x: 1 / x, lambda x: -1 / x ** 2), dom=(0.0, math.inf),
         view=(0.02, 5.5), kind="logarithmic"),
    dict(label="y = 1/x  (x > 0)", tex=r"$y = 1/x$", pt=(0.0, 0.0),
         fs=(lambda x: 1 / x, lambda x: -1 / x ** 2, lambda x: 2 / x ** 3), dom=(0.0, math.inf),
         view=(0.15, 4), kind="rational", box="upper right"),
    dict(label="y = 4/(1 + x²)", tex=r"$y = \dfrac{4}{1 + x^2}$", pt=(2.0, 3.0),
         fs=(lambda x: 4 / (1 + x * x), None, None), dom=(-math.inf, math.inf),
         view=(-2, 4), kind="rational, numeric derivatives", box="upper right"),
    dict(label="y = √x", tex=r"$y = \sqrt{x}$", pt=(4.0, 0.0),
         fs=(sqrt_, lambda x: 0.5 / math.sqrt(x), lambda x: -0.25 * x ** -1.5),
         dom=(0.0, math.inf), view=(0, 6), kind="radical"),
    dict(label="y = sin x + 0.3x", tex=r"$y = \sin x + 0.3x$", pt=(1.0, 2.5),
         fs=(lambda x: math.sin(x) + 0.3 * x, None, None), dom=(-math.inf, math.inf),
         view=(-1.5, 4.5), kind="trigonometric, numeric derivatives"),
]


def brute_force(x0, y0, fn, lo, hi):
    """Dense grid + local refinement used only as an independent check."""
    grid = np.linspace(lo, hi, 200001)
    vals = [(x - x0) ** 2 + (fn(x) - y0) ** 2 for x in grid]
    i = int(np.argmin(vals))
    return grid[i], math.sqrt(vals[i])


EX_RESULTS = []
fig, axes = plt.subplots(3, 3, figsize=(15, 14), gridspec_kw=dict(hspace=0.36, wspace=0.22))
for ax, ex in zip(axes.flat, EXAMPLES):
    fn, dfn, ddfn = ex["fs"]
    x0, y0 = ex["pt"]
    dom = ex["dom"]
    starts = ex.get("starts", (x0 if dom[0] < x0 < dom[1] else 2.5,))
    dn, xn, hn = newton_distance_multistart(x0, y0, fn, dfn, ddfn, starts=starts,
                                            tol=1e-12, domain=dom)
    a, b = default_bracket(x0, y0, fn, (dom[0] + 1e-9, dom[1]))
    if ex["label"].startswith("y = −x² + 4"):
        a, b = 0.0, 2.5      # two symmetric minima: restrict to one unimodal half
    dg, xg, hg = golden_section_distance(x0, y0, fn, a, b, tol=1e-10)
    xr, dr = brute_force(x0, y0, fn, max(ex["view"][0], dom[0] + 1e-6), ex["view"][1])
    EX_RESULTS.append(dict(label=ex["label"], kind=ex["kind"], point=[x0, y0],
                           newton=dict(x=xn, d=dn, iters=len(hn),
                                       safeguarded=any(h["safeguarded"] or h["halvings"] for h in hn)),
                           golden=dict(x=xg, d=dg, iters=len(hg) - 1, bracket=[a, b]),
                           check=dict(x=xr, d=dr),
                           numeric=dfn is None))
    lo, hi = ex["view"]
    xx = np.linspace(lo, hi, 600)
    yy = np.array([fn(x) for x in xx])
    ax.plot(xx, yy, color=INK, lw=2, zorder=3)
    iters = [h["x"] for h in hn] + [hn[-1]["x_next"]]
    for k, xk in enumerate(iters):
        c = BLUE_RAMP(k / max(len(iters) - 1, 1))
        if lo <= xk <= hi:
            ax.plot([x0, xk], [y0, fn(xk)], color=c, lw=0.9, ls="--", zorder=2)
            ax.scatter([xk], [fn(xk)], color=c, s=26, edgecolor="white", lw=0.8, zorder=4)
    if ex["label"].startswith("y = −x² + 4"):
        ax.plot([x0, -xn], [y0, fn(-xn)], color=AQUA, lw=2.2, zorder=3, alpha=0.55)
        ax.scatter([-xn], [fn(-xn)], s=50, color=AQUA, edgecolor="white", lw=1.2, zorder=6)
    ax.plot([x0, xn], [y0, fn(xn)], color=AQUA, lw=2.2, zorder=3)
    ax.scatter([xn], [fn(xn)], s=55, color=AQUA, edgecolor="white", lw=1.4, zorder=6)
    ax.scatter([x0], [y0], s=75, color=ORANGE, edgecolor="white", lw=1.4, zorder=6)
    ax.annotate(f"({x0:g}, {y0:g})", (x0, y0), xytext=(8, 6), textcoords="offset points",
                color=ORANGE, fontsize=9.5, fontweight="semibold")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(ex["tex"] + f"   ·   {ex['kind']}", fontsize=11.5)
    up = ex.get("box") == "upper right"
    ax.text(0.97, 0.96 if up else 0.04, f"d = {dn:.5f}\nx* = {xn:.5f}\nNewton {len(hn)} it · GSS {len(hg)-1} it",
            transform=ax.transAxes, ha="right", va="top" if up else "bottom", fontsize=8.8,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GRID))
fig.suptitle("Same two routines applied to other parabolas and to non-polynomial functions",
             x=0.12, ha="left", fontsize=14, fontweight="semibold", y=0.93)
save(fig, "p1_other_functions.png")
RESULTS["part1_examples"] = EX_RESULTS


# --- Figure 9: why the safeguard / multi-start matters -----------------------
fn, dfn, ddfn = make_parabola(-1, 0, 4)
x0, y0 = 0.0, 1.0
fig, ax = plt.subplots(figsize=(7.5, 3.8))
xx = np.linspace(-2.4, 2.4, 400)
ax.plot(xx, [dist_sq(x, x0, y0, fn) for x in xx], color=INK, lw=2,
        label="D(x) for y = −x² + 4, point (0, 1)")
xm = math.sqrt(2.5)
ax.scatter([-xm, xm], [dist_sq(xm, x0, y0, fn)] * 2, color=AQUA, s=60, zorder=5,
           edgecolor="white", lw=1.4, label="two global minima x = ±√2.5")
ax.scatter([0], [dist_sq(0, x0, y0, fn)], color=RED, s=60, zorder=5,
           edgecolor="white", lw=1.4, label="local max at x = 0 (D'' = −10)")
ax.set_xlabel("x"); ax.set_ylabel("D(x)")
ax.set_title("A non-convex D(x): pure Newton from x = 0 would stall on the maximum")
ax.legend(loc="upper center", fontsize=9)
ax.set_ylim(0, 14)
save(fig, "p1_nonconvex.png")


# ===================================================== PART 2 : fitting =====
XS = [0, 2, 1, 3]
YS = [0.5, 3.5, 1.5, 7.5]
P2 = {}
for model in (LINE, PARABOLA):
    th_a, A, rhs = analytic_fit(model, XS, YS)
    th_c, hc = coordinate_newton_fit(model, XS, YS, tol=1e-12, max_sweeps=2000)
    th_n, hn = full_newton_fit(model, XS, YS)
    P2[model["name"]] = dict(analytic=th_a, normal_A=A, normal_rhs=rhs,
                             mse=mse(model, th_a, XS, YS),
                             coord=dict(theta=th_c, sweeps=hc[-1]["sweep"],
                                        history=[dict(sweep=h["sweep"], coord=h["coord"],
                                                      theta=h["theta"], mse=h["mse"])
                                                 for h in hc[:3 * len(model["params"]) + 1]]),
                             full=dict(theta=th_n, history=hn))
    # convergence factor of one Gauss-Seidel sweep (spectral radius)
    H = np.array(A, float)
    L = np.tril(H); U = H - L
    rho = max(abs(np.linalg.eigvals(-np.linalg.solve(L, U))))
    P2[model["name"]]["rho"] = float(rho)
    P2[model["name"]]["cond"] = float(np.linalg.cond(H))
    P2[model["name"]]["_hist"] = hc
RESULTS["part2"] = {k: {kk: vv for kk, vv in v.items() if kk != "_hist"} for k, v in P2.items()}

# --- Figure 10: line — MSE contours with coordinate-Newton zig-zag --------
hc = P2["line"]["_hist"]
m_s, b_s = P2["line"]["analytic"]
fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw=dict(width_ratios=[1.1, 1]))
ax = axes[0]
M, B = np.meshgrid(np.linspace(-0.4, 3.2, 300), np.linspace(-1.6, 1.0, 300))
Z = 3.5 * M ** 2 + 3 * M * B + B ** 2 - 15.5 * M - 6.5 * B + 17.75
levels = np.geomspace(0.6, 18, 14)
cs = ax.contour(M, B, Z, levels=levels, cmap=LinearSegmentedColormap.from_list(
    "g", ["#d9d6cf", "#8e8a80"]), linewidths=0.9)
ax.clabel(cs, fmt="%.2f", fontsize=7.5, inline_spacing=2)
path = np.array([h["theta"] for h in hc[:25]])
ax.plot(path[:, 0], path[:, 1], color=BLUE, lw=1.6, zorder=4, label="coordinate Newton (m, then b)")
for i, (mm, bb) in enumerate(path[:9]):
    ax.scatter([mm], [bb], color=BLUE_RAMP(i / 8), s=26, zorder=5, edgecolor="white", lw=0.8)
ax.annotate("start (0, 0)", path[0], xytext=(-6, 8), textcoords="offset points",
            fontsize=9, color=BLUE, ha="right")
ins = ax.inset_axes([0.05, 0.05, 0.42, 0.42])
ins.contour(M, B, Z, levels=np.linspace(0.575, 0.6, 9), colors="#bdb9b0", linewidths=0.7)
ins.plot(path[:, 0], path[:, 1], color=BLUE, lw=1.3)
for i, (mm, bb) in enumerate(path[:25]):
    ins.scatter([mm], [bb], color=BLUE_RAMP(min(i / 12, 1)), s=16, zorder=5, edgecolor="white", lw=0.6)
for i in (1, 2, 3, 4):
    ins.annotate(f"{i}: {'m' if i % 2 else 'b'}", path[i], xytext=(5, -2), textcoords="offset points",
                 fontsize=7.5, color=BLUE)
ins.scatter([m_s], [b_s], marker="*", s=120, color=AQUA, zorder=6, edgecolor="white", lw=0.6)
ins.set_xlim(2.19, 2.32); ins.set_ylim(-0.23, 0.02)
ins.set_title("zoom: alternating m / b updates", fontsize=8.5, loc="left", pad=3)
ins.tick_params(labelsize=7)
ins.set_facecolor("white")
ax.indicate_inset_zoom(ins, edgecolor=MUTED)
fn_path = np.array([h["theta"] for h in P2["line"]["full"]["history"]])
ax.plot(fn_path[:, 0], fn_path[:, 1], color=ORANGE, lw=1.4, ls="--", zorder=3,
        label="full 2-D Newton (1 step)")
ax.scatter([m_s], [b_s], marker="*", s=220, color=AQUA, edgecolor="white", lw=1, zorder=6,
           label=f"optimum m = {m_s:.2f}, b = {b_s:.2f}")
ax.set_xlabel("slope m"); ax.set_ylabel("intercept b")
ax.set_title("Line: MSE(m, b) contours and optimisation paths")
ax.legend(loc="upper right", fontsize=9, frameon=True, facecolor=SURFACE, edgecolor="none")
ax.set_xlim(-0.4, 3.2); ax.set_ylim(-1.6, 1.0)

ax = axes[1]
xx = np.linspace(-0.3, 3.3, 100)
show = [0, 1, 2, 3, 4, 6, 10]
for n_, i in enumerate(show):
    th = hc[i]["theta"]
    tag = "start" if i == 0 else f"after update {i} ({'m' if hc[i]['coord'] == 0 else 'b'})"
    ax.plot(xx, th[0] * xx + th[1], color=BLUE_RAMP(n_ / (len(show) - 1)), lw=1.3,
            alpha=0.9, label=f"{tag}: MSE {hc[i]['mse']:.3f}")
ax.plot(xx, m_s * xx + b_s, color=AQUA, lw=2.6, label=f"final: y = {m_s:.1f}x {b_s:+.1f},  MSE {P2['line']['mse']:.3f}")
ax.scatter(XS, YS, s=70, color=ORANGE, edgecolor="white", lw=1.4, zorder=6, label="data")
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title("Line: fitted model after each single-parameter update")
ax.legend(loc="upper left", fontsize=8.3)
save(fig, "p2_line.png")

# --- Figure 11: parabola — curves and parameter trajectories ---------------
hc = P2["parabola"]["_hist"]
a_s, b_s2, c_s = P2["parabola"]["analytic"]
fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
ax = axes[0]
show = [0, 1, 2, 3, 6, 9, 15, 30]
for n_, i in enumerate(show):
    th = hc[i]["theta"]
    ax.plot(xx, [predict(PARABOLA, th, x) for x in xx],
            color=BLUE_RAMP(n_ / (len(show) - 1)), lw=1.3,
            label=("start" if i == 0 else f"update {i}") + f": MSE {hc[i]['mse']:.4f}")
ax.plot(xx, [predict(PARABOLA, [a_s, b_s2, c_s], x) for x in xx], color=AQUA, lw=2.6,
        label=f"final: y = {a_s:.2f}x² + {b_s2:.2f}x + {c_s:.2f},  MSE {P2['parabola']['mse']:.4f}")
ax.scatter(XS, YS, s=70, color=ORANGE, edgecolor="white", lw=1.4, zorder=6, label="data")
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title("Parabola: fitted model after single-parameter updates")
ax.legend(loc="upper left", fontsize=8.3)

ax = axes[1]
sweeps = [h for h in hc if h["coord"] in (None, 2)]
sw = np.arange(len(sweeps))
for j, (name, col, target) in enumerate(zip("abc", [BLUE, ORANGE, VIOLET], [a_s, b_s2, c_s])):
    vals = [h["theta"][j] for h in sweeps]
    ax.plot(sw[:80], vals[:80], color=col, lw=1.8, label=f"{name}  (target {target:.2f})")
    ax.axhline(target, color=col, lw=0.8, ls=":")
ax.set_xlabel("sweep (one Newton step on a, then b, then c)")
ax.set_ylabel("parameter value")
ax.set_title("Parabola: parameter trajectories (first 80 sweeps)")
ax.legend(loc="center right")
save(fig, "p2_parabola.png")

# --- Figure 12: convergence comparison and final fits ------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
ax = axes[0]
for name, col in (("line", BLUE), ("parabola", ORANGE)):
    hist = P2[name]["_hist"]
    p = 2 if name == "line" else 3
    sweeps = [h for h in hist if h["coord"] in (None, p - 1)]
    gap = [h["mse"] - P2[name]["mse"] for h in sweeps]
    ks = [i for i, g in enumerate(gap) if i >= 1 and g > 1e-14]
    ax.loglog(ks, [gap[i] for i in ks], color=col, lw=2,
              label=f"{name}: {P2[name]['coord']['sweeps']} sweeps to converge, "
                    f"rate ρ = {P2[name]['rho']:.3f} per sweep")
ax.text(0.03, 0.05, "Full multivariate Newton (all parameters at once)\nreaches MSE* exactly in 1 step for both models.",
        transform=ax.transAxes, fontsize=9, color=MUTED)
ax.set_xlabel("sweep (log scale)"); ax.set_ylabel("MSE − MSE*")
ax.set_title("Convergence of coordinate-wise Newton")
ax.legend(loc="upper right", fontsize=8.8)
ax.set_ylim(1e-14, 10)

ax = axes[1]
xx = np.linspace(-0.3, 3.3, 200)
ml, bl = P2["line"]["analytic"]
ax.plot(xx, ml * xx + bl, color=BLUE, lw=2.2, label=f"line  y = 2.3x − 0.2   (MSE = {P2['line']['mse']:.3f})")
ax.plot(xx, [predict(PARABOLA, P2["parabola"]["analytic"], x) for x in xx], color=VIOLET, lw=2.2,
        label=f"parabola  y = 0.75x² + 0.05x + 0.55   (MSE = {P2['parabola']['mse']:.4f})")
for x, y in zip(XS, YS):
    ax.plot([x, x], [y, ml * x + bl], color=BLUE, lw=1, ls=":")
    ax.plot([x + 0.04, x + 0.04], [y, predict(PARABOLA, P2["parabola"]["analytic"], x)],
            color=VIOLET, lw=1.6)
ax.scatter(XS, YS, s=70, color=ORANGE, edgecolor="white", lw=1.4, zorder=6, label="data")
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title("Final fits with residuals")
ax.legend(loc="upper left", fontsize=9)
save(fig, "p2_final.png")


def to_jsonable(o):
    if isinstance(o, dict):
        return {k: to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


with open(os.path.join(HERE, "results.json"), "w") as fh:
    json.dump(to_jsonable(RESULTS), fh, indent=1)
print("wrote results.json")
