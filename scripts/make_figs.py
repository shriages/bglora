JSONL = "./runs/train_metrics.jsonl"   # <-- set your train_metrics.jsonl path here

# pip install matplotlib numpy   (ffmpeg needed only for the .mp4)
import json, numpy as np, matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa

rows = [json.loads(l) for l in open(JSONL)]
step = np.array([r["step"] for r in rows])
R    = sum(k.startswith("pa_") for k in rows[0])          # number of angles (8)
pa   = np.array([[r[f"pa_{i}"] for i in range(1, R+1)] for r in rows])  # (T, R) radians
cov  = np.array([r["coverage"] for r in rows])
HALF = np.pi/2
colors = cm.Blues(np.linspace(0.35, 0.95, R))
def smooth(a, w=9):
    k = np.ones(w)/w
    return np.convolve(np.pad(a,(w//2,w//2),"edge"), k, "valid")[:len(a)]

# ── 1. fanning principal angles ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
for k in range(R):
    ax.plot(step, pa[:, k], color=colors[k], lw=1.4)
ax.axhline(HALF, ls="--", c="0.6", lw=.8); ax.text(step[-1], HALF, " π/2", va="center", c="0.5")
ax.set(xlabel="training step", ylabel="principal angle θ (rad)", ylim=(0, HALF*1.05))
ax.text(step[-1], pa[-1,0], f" θ1", color=colors[0], va="center")
ax.text(step[-1], pa[-1,-1], f" θ{R}", color=colors[-1], va="center")
fig.tight_layout(); fig.savefig("fig_fan.pdf"); fig.savefig("fig_fan.png")

# ── 2. rotation heatmap ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.4, 2.6), dpi=150)
im = ax.imshow(pa.T, aspect="auto", origin="lower", cmap="viridis",
               extent=[step[0], step[-1], 0.5, R+0.5], vmin=0, vmax=HALF)
ax.set(xlabel="training step", ylabel="angle index", yticks=range(1, R+1))
fig.colorbar(im, ax=ax, label="θ (rad)")
fig.tight_layout(); fig.savefig("fig_heatmap.pdf"); fig.savefig("fig_heatmap.png")

# ── 3. angles + coverage (cause & effect) ─────────────────────────────────
fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 5.0), dpi=150, sharex=True,
                             gridspec_kw=dict(height_ratios=[2, 1]))
for k in range(R):
    a1.plot(step, pa[:, k], color=colors[k], lw=1.2)
a1.axhline(HALF, ls="--", c="0.6", lw=.8); a1.set(ylabel="θ (rad)", ylim=(0, HALF*1.05))
a2.plot(step, cov, c="tab:orange", lw=.7, alpha=.3)          # raw
a2.plot(step, smooth(cov), c="tab:orange", lw=2)             # smoothed
a2.axhline(cov[0], ls="--", c="0.6", lw=.8)
a2.set(xlabel="training step", ylabel="coverage")
fig.tight_layout(); fig.savefig("fig_combo.pdf"); fig.savefig("fig_combo.png")

# ── 4. Grassmann sphere animation (mp4) ───────────────────────────────────
# each direction = a point on a sphere; principal angle = distance from pole.
# azimuth is spread evenly (schematic); latitude is the measured angle.
try:
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    phi = np.linspace(0, 2*np.pi, R, endpoint=False)
    X = np.sin(pa)*np.cos(phi); Y = np.sin(pa)*np.sin(phi); Z = np.cos(pa)  # (T,R)
    fig = plt.figure(figsize=(6, 6), dpi=120); ax = fig.add_subplot(111, projection="3d")
    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    def draw(t):
        ax.clear(); ax.set_axis_off()
        ax.plot_surface(np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v),
                        color="tab:blue", alpha=0.08, linewidth=0)
        th = np.linspace(0, 2*np.pi, 100)                    # equator
        ax.plot(np.cos(th), np.sin(th), 0, c="tab:orange", lw=1.5)
        ax.scatter(0, 0, 1, c="k", s=30)                     # pole = t=0
        for k in range(R):
            ax.plot(X[:t+1,k], Y[:t+1,k], Z[:t+1,k], color=colors[k], lw=1.5)
            ax.scatter(X[t,k], Y[t,k], Z[t,k], color=colors[k], s=25)
        ax.set(xlim=(-1,1), ylim=(-1,1), zlim=(-1,1)); ax.set_title(f"step {step[t]}")
    anim = FuncAnimation(fig, draw, frames=range(0, len(step), 2), interval=40)
    anim.save("fig_sphere.mp4", writer=FFMpegWriter(fps=25))
    print("wrote fig_sphere.mp4")
except Exception as e:
    print("sphere animation skipped (need ffmpeg):", e)

print("done:", "fig_fan / fig_heatmap / fig_combo (.pdf/.png)")