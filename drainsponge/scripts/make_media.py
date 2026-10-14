"""
Videos, renders and a gallery for presenting the study (all in out/media/).

  python scripts/make_media.py data        # preview runs with 72 frames each (3 designs x 2 scenarios)
  python scripts/make_media.py anim2d      # side-by-side cross-section animations (mp4 + gif)
  python scripts/make_media.py anim3d      # 3D tissue-on-hole animations per design (mp4)
  python scripts/make_media.py turntable   # real drain CAD + sponge organ, orbiting camera (mp4)
  python scripts/make_media.py stills      # close-up renders of the drains with the organ
  python scripts/make_media.py gallery     # out/media/index.html + README.md (copies the figures)
  python scripts/make_media.py all

What is real and what is computed:
  - drain geometry = your CAD (drainage-3D_cap_dome*.stl); organ = shape made here, placed on the
    drain in its real coordinates, 50 um above the skin / ring tops
  - all time histories, pressures, deformations = reduced-order model (preview.py) with
    placeholder tissue properties (E 10 kPa, k 1e-14 m4/Ns, suction 10 kPa)
"""
import json, os, shutil, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from drainsponge import config
from drainsponge.preview import Preview
from drainsponge.geom import Window, WindowMesh, Z_HOLE, Z_STAND, Z_FILM, Z_TIS, Z_CHAN

OUT = os.path.join("out", "media")
DATA = os.path.join(OUT, "data")
FIG = os.path.join("out", "figures")
DESIGNS = ["current", "standoff", "standoff_opt"]
NAMES = {"current": "Current design (plain hole)", "standoff": "Standoff ring (new design)",
         "standoff_opt": "Optimized ring (suggestion)"}
SCEN = {"closed": "no free liquid - only the liquid inside the tissue",
        "open": "free wound liquid around the drain"}
CFG_FILE = os.path.join("configs", "compare.json")
NFRAMES = 72


def _ffmpeg_writer(path, fps):
    import imageio
    return imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=8, pixelformat="yuv420p")


# ---------------------------------------------------------------- data
def _run(args):
    design, scen = args
    cfg = config.load(CFG_FILE, design=design, boundary={"film_edge": scen})
    pv = Preview(cfg)
    t_end = cfg["time"]["t_end_s"]
    frames = list(np.geomspace(1e-4, t_end, NFRAMES))
    h, snaps = pv.run(t_end=t_end, n_steps=160, snap_times=frames, verbose=False)
    S = {k: np.array([s[k] for s in snaps]) for k in ("t", "p_hole_Pa", "u", "p_Pa", "w_mm_s", "film_h", "film_p_Pa")}
    hist = {k: np.array(h[k], float) for k in ("t", "Q_hole", "Q_edge", "Q_body", "Q_seep", "clearance")}
    np.savez_compressed(os.path.join(DATA, f"{design}_{scen}.npz"), r2=pv.r2, z2=pv.z2, xc=snaps[0]["xc"], zb=pv.zb,
                        g0=pv.g0, hs=pv.hs, rin=pv.rin, rout=pv.rout, a=pv.a, R=pv.R, wall=pv.t_wall, H=pv.H,
                        dp=cfg["suction"]["dp_Pa"], E=cfg["tissue"]["E_Pa"],
                        **{f"s_{k}": v for k, v in S.items()}, **{f"h_{k}": v for k, v in hist.items()})
    return f"{design}_{scen}"


def make_data():
    os.makedirs(DATA, exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(d, s) for d in DESIGNS for s in SCEN]
    with ProcessPoolExecutor(6) as ex:
        for tag in ex.map(_run, jobs):
            print("data", tag, flush=True)


def load(tag):
    return dict(np.load(os.path.join(DATA, f"{tag}.npz")))


def fmt_t(t):
    if t < 1e-3:
        return f"{t*1e6:.0f} us"
    if t < 1:
        return f"{t*1e3:.1f} ms"
    if t < 120:
        return f"{t:.1f} s"
    if t < 7200:
        return f"{t/60:.1f} min"
    return f"{t/3600:.1f} h"


# ---------------------------------------------------------------- 2D side-by-side animation
def draw_section(ax, D, i, title, cbar_ax=None):
    from scipy.interpolate import griddata
    r, z, g0, a, R, t_w = D["r2"], D["z2"], float(D["g0"]), float(D["a"]), float(D["R"]), float(D["wall"])
    hs, rin, rout = float(D["hs"]), float(D["rin"]), float(D["rout"])
    u, p = D["s_u"][i], D["s_p_Pa"][i]
    Rg, Zg = np.meshgrid(r, z)
    Xd, Zd = Rg + u[..., 0], g0 + Zg + u[..., 1]
    Xf = np.hstack([-Xd[:, ::-1], Xd]); Zf = np.hstack([Zd[:, ::-1], Zd]); Pf = np.hstack([p[:, ::-1], p])
    dp = float(D["dp"])
    cs = ax.pcolormesh(Xf, Zf, Pf * 1e-3, cmap="RdBu", vmin=-dp * 1e-3, vmax=dp * 1e-3, shading="gouraud")
    xc, wv = D["xc"], D["s_w_mm_s"][i]
    gx = np.linspace(-R * 0.97, R * 0.97, 120); gz = np.linspace(0.02, 1.2, 50)
    GX, GZ = np.meshgrid(gx, gz)
    pts = np.r_[xc, xc * [-1, 1]]; W = np.r_[wv, wv * [-1, 1]]
    U = griddata(pts, W[:, 0], (GX, GZ)); V = griddata(pts, W[:, 1], (GX, GZ))
    if np.isfinite(U).any() and np.nanmax(np.hypot(U, V)) > 0:
        ax.streamplot(GX, GZ + g0, U, V, color="#222222", linewidth=0.6, density=0.9, arrowsize=0.6)
    ax.fill_between([-R, -a], [-t_w, -t_w], [0, 0], color="#8a9bb0", zorder=3)
    ax.fill_between([a, R], [-t_w, -t_w], [0, 0], color="#8a9bb0", zorder=3)
    ax.fill_between([-a, a], [-t_w - 0.3, -t_w - 0.3], [-t_w, -t_w], color="#cfe3f7", zorder=2)
    hf, zb = D["s_film_h"][i], D["zb"]
    o = r >= a
    for sg in (1, -1):
        ax.fill_between(sg * r[o], zb[o], zb[o] + hf[o], color="#4a90d9", alpha=0.85, lw=0, zorder=4)
        if hs > 0:
            ax.fill_between(sg * np.array([rin, rout]), [0, 0], [hs, hs], color="#4a5568", zorder=5)
    ub = u[0, :, 1]
    low = float((g0 + ub[r < (rin if hs > 0 else a)]).min())
    ax.set_xlim(-R, R); ax.set_ylim(-t_w - 0.3, 1.25); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    col = "#1b7f3b" if low > 0.02 else ("#b8860b" if low > 0 else "#c0392b")
    ax.set_title(title, fontsize=11)
    ax.text(0.02, 0.97, f"tissue {'above' if low >= 0 else 'INSIDE'} the hole: {low*1e3:+.0f} um", transform=ax.transAxes,
            va="top", fontsize=10, color=col, weight="bold", bbox=dict(fc="white", ec="none", alpha=0.8))
    return cs


def anim2d():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for scen, sdesc in SCEN.items():
        Ds = [load(f"{d}_{scen}") for d in DESIGNS]
        T = Ds[0]["s_t"]
        frames = []
        for i in range(len(T)):
            fig = plt.figure(figsize=(16, 6.2), dpi=100)
            gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.42], hspace=0.28, wspace=0.05, left=0.02, right=0.93,
                                  top=0.86, bottom=0.08)
            for k, (d, D) in enumerate(zip(DESIGNS, Ds)):
                ax = fig.add_subplot(gs[0, k])
                cs = draw_section(ax, D, i, NAMES[d])
            cax = fig.add_axes([0.945, 0.42, 0.012, 0.42])
            fig.colorbar(cs, cax=cax, label="liquid pressure in the tissue (kPa)")
            ax = fig.add_subplot(gs[1, :])
            for d, D, c in zip(DESIGNS, Ds, ["#c0392b", "#2f6db3", "#1b9e77"]):
                ax.semilogx(D["h_t"], D["h_clearance"] * 1e3, color=c, lw=2, label=NAMES[d])
            ax.axhline(0, color="k", lw=0.8)
            ax.axvline(T[i], color="#555", lw=1.2, ls="--")
            ax.set_ylabel("tissue vs hole edge (um)", fontsize=9)
            ax.set_xlabel("time after suction is switched on (s, log scale)", fontsize=9)
            ax.legend(fontsize=8, loc="lower left", ncol=3)
            ax.grid(True, which="both", alpha=0.25)
            fig.suptitle(f"One drain hole with a wet sponge-like tissue on it - {sdesc}\n"
                         f"t = {fmt_t(T[i])} after suction on   |   suction {Ds[0]['dp']/1e3:.0f} kPa, tissue stiffness "
                         f"{Ds[0]['E']/1e3:.0f} kPa   |   blue = liquid, grey = drain, lines = liquid flowing inside the tissue",
                         fontsize=11)
            fig.canvas.draw()
            img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
            frames.append(img)
            plt.close(fig)
        name = f"video_cross_section_{scen}"
        w = _ffmpeg_writer(os.path.join(OUT, name + ".mp4"), fps=8)
        for f in frames + [frames[-1]] * 12:
            w.append_data(f)
        w.close()
        import imageio
        small = [f[::2, ::2] for f in frames[::2]] + [frames[-1][::2, ::2]] * 6
        imageio.mimsave(os.path.join(OUT, name + ".gif"), small, duration=0.18, loop=0)
        imageio.imwrite(os.path.join(OUT, name + "_last_frame.png"), frames[-1])
        print("wrote", name, flush=True)


# ---------------------------------------------------------------- 3D window animation
def _half_surface(points, cells, keep):
    import pyvista as pv
    cl = np.hstack([np.full((keep.sum(), 1), 8), cells[keep]]).ravel()
    g = pv.UnstructuredGrid(cl, np.full(keep.sum(), pv.CellType.HEXAHEDRON), points)
    return g.extract_surface(algorithm="dataset_surface").clean()


def anim3d():
    import pyvista as pv
    from scipy.interpolate import RegularGridInterpolator
    for design in DESIGNS:
        for scen in ("closed",):
            D = load(f"{design}_{scen}")
            w = Window.from_dict(config.load(design=design)["window"])
            m = WindowMesh(w)
            cut = np.pi / 4 if w.has_standoff else 0.0
            nrm = np.array([-np.sin(cut), np.cos(cut), 0.0])
            P = m.points
            half = P[m.cells].mean(1) @ nrm > 0
            tis = _half_surface(P, m.cells, half & (m.cell_zone == Z_TIS))
            liq = _half_surface(P, m.cells, half & np.isin(m.cell_zone, [Z_HOLE, Z_STAND, Z_FILM]))
            chan = _half_surface(P, m.cells, half & (m.cell_zone == Z_CHAN))
            K = m.NK + 1
            cells = []
            for k in range(m.k0[1], m.k0[2]):
                lo = m.quads[~m.cell_hole] * K + k
                cells.append(np.c_[lo, lo + 1])
            if w.has_standoff:
                for k in range(m.k0[2], m.k0[3]):
                    lo = m.quads[m.cell_ring] * K + k
                    cells.append(np.c_[lo, lo + 1])
            cells = np.concatenate(cells)
            wall = _half_surface(m.points_all, cells, m.points_all[cells].mean(1) @ nrm > 0)
            r2, z2 = D["r2"], D["z2"]
            x0 = tis.points.copy()
            rr = np.hypot(x0[:, 0], x0[:, 1])
            zrel = np.clip(x0[:, 2] - (w.skin_e(x0[:, 0]) + w.top_feature + w.gap), 0, w.tissue_h)
            lq0 = liq.points.copy()
            lr = np.hypot(lq0[:, 0], lq0[:, 1])
            base = w.skin_e(lq0[:, 0]) + w.top_feature
            frac = np.where(lq0[:, 2] >= base - 1e-9, (lq0[:, 2] - base) / w.gap, -1)
            T = D["s_t"]
            path = os.path.join(OUT, f"video_3d_{design}_{scen}.mp4")
            wr = _ffmpeg_writer(path, fps=8)
            pl = pv.Plotter(off_screen=True, window_size=(1280, 896))
            pl.set_background("white")
            c, s = np.cos(cut), np.sin(cut)
            cam = np.array([3.2 * c + 6.5 * s, 3.2 * s - 6.5 * c, 2.6])
            dp = float(D["dp"])
            last = None
            for i in range(len(T)):
                fu = RegularGridInterpolator((z2, r2), D["s_u"][i][..., 0], bounds_error=False, fill_value=None)
                fw = RegularGridInterpolator((z2, r2), D["s_u"][i][..., 1], bounds_error=False, fill_value=None)
                fp = RegularGridInterpolator((z2, r2), D["s_p_Pa"][i], bounds_error=False, fill_value=None)
                q = np.c_[zrel, np.minimum(rr, r2[-1])]
                ur, uz, pp = fu(q), fw(q), fp(q)
                X = x0.copy()
                cr = np.where(rr > 1e-9, 1 / np.maximum(rr, 1e-9), 0)
                X[:, 0] += ur * x0[:, 0] * cr; X[:, 1] += ur * x0[:, 1] * cr; X[:, 2] += uz
                tis.points = X
                tis["liquid pressure (kPa)"] = pp * 1e-3
                bot = w.skin_e(lq0[:, 0]) + w.top_feature + w.gap + RegularGridInterpolator(
                    (r2,), D["s_u"][i][0, :, 1], bounds_error=False, fill_value=None)(np.minimum(lr, r2[-1])[:, None])
                L = lq0.copy()
                L[:, 2] = np.where(frac >= 0, base + frac * (bot - base), lq0[:, 2])
                liq.points = L
                pl.clear()
                pl.add_mesh(wall, color="#8a9bb0", specular=0.2)
                pl.add_mesh(chan, color="#9cc8f0", opacity=0.25)
                pl.add_mesh(liq, color="#2f8fe0", opacity=0.55)
                pl.add_mesh(tis, scalars="liquid pressure (kPa)", cmap="RdBu", clim=[-dp * 1e-3, dp * 1e-3],
                            scalar_bar_args=dict(title="liquid pressure in the tissue (kPa)", vertical=True,
                                                 position_x=0.86, position_y=0.25, height=0.5, width=0.05,
                                                 title_font_size=12, label_font_size=11))
                low = float((D["g0"] + D["s_u"][i][0, :, 1][r2 < (w.standoff_rin if w.has_standoff else w.a)]).min())
                pl.add_text(f"{NAMES[design]} - no free liquid\nt = {fmt_t(T[i])} after suction on\n"
                            f"tissue {'above' if low >= 0 else 'INSIDE'} the hole: {low*1e3:+.0f} um  (true scale)",
                            font_size=12, color="black")
                pl.camera_position = [tuple(cam), (-0.5 * s, 0.5 * c, 0.1), (0, 0, 1)]
                pl.render()
                img = pl.screenshot(return_img=True)
                wr.append_data(img)
                last = img
            for _ in range(12):
                wr.append_data(last)
            wr.close()
            pl.close()
            import imageio
            imageio.imwrite(os.path.join(OUT, f"render_3d_{design}_{scen}_end.png"), last)
            print("wrote", path, flush=True)


# ---------------------------------------------------------------- real CAD: turntables + stills
def _cad_scene(design):
    import pyvista as pv
    stl = {"current": "drainage-3D_cap_dome.stl", "standoff": "drainage-3D_cap_dome_standoff.stl"}[design]
    drain = pv.read(os.path.join("data", stl))
    organ = pv.read(os.path.join("out", "cad", design, "organ_sponge_global.stl")).compute_normals()
    T = np.array(json.load(open(os.path.join("out", "cad", design, "frame.json")))["local_to_global"])
    from render_cad import sponge_scalars
    organ["s"] = sponge_scalars(organ, 0.35)
    return drain, organ, T


def turntable():
    import pyvista as pv
    sys.path.insert(0, os.path.dirname(__file__))
    for design in ("standoff", "current"):
        drain, organ, T = _cad_scene(design)
        O, X, Y, Z = T[:3, 3], T[:3, 0], T[:3, 1], T[:3, 2]
        near = drain.clip_box((O[0] - 45, O[0] + 45, O[1] - 45, O[1] + 45, O[2] - 45, O[2] + 45), invert=False)
        path = os.path.join(OUT, f"video_turntable_drain_organ_{design}.mp4")
        wr = _ffmpeg_writer(path, fps=24)
        pl = pv.Plotter(off_screen=True, window_size=(1280, 896))
        pl.set_background("white")
        pl.add_mesh(near, color="#8a9bb0", smooth_shading=True, specular=0.35)
        pl.add_mesh(organ, scalars="s", cmap=["#6e1f2c", "#b5525f", "#e59aa4", "#f4c6cb"], show_scalar_bar=False,
                    smooth_shading=True, opacity=0.93)
        pl.add_text(f"{NAMES[design]} with a sponge-like organ lying on it (real drain CAD)", font_size=12, color="black")
        n = 144
        for k in range(n):
            a = 2 * np.pi * k / n
            dirv = np.cos(a) * Z + np.sin(a) * X               # orbit around the drain axis Y
            pos = O + 85 * dirv + 30 * Y
            pl.camera_position = [tuple(pos), tuple(O - 8 * Y), tuple(Y)]
            pl.render()
            wr.append_data(pl.screenshot(return_img=True))
        wr.close()
        pl.close()
        print("wrote", path, flush=True)


def stills():
    import pyvista as pv
    sys.path.insert(0, os.path.dirname(__file__))
    for design in ("standoff", "current"):
        drain, organ, T = _cad_scene(design)
        O, X, Y, Z = T[:3, 3], T[:3, 0], T[:3, 1], T[:3, 2]
        near = drain.clip_box((O[0] - 14, O[0] + 14, O[1] - 14, O[1] + 14, O[2] - 14, O[2] + 14), invert=False)
        # 1: close-up, organ semi-transparent so the holes (and rings) underneath show
        pl = pv.Plotter(off_screen=True, window_size=(1600, 1100))
        pl.set_background("white")
        pl.add_mesh(near, color="#8a9bb0", smooth_shading=True, specular=0.35)
        pl.add_mesh(organ, color="#e8a0a8", smooth_shading=True, opacity=0.33, specular=0.3)
        pl.add_text(f"{NAMES[design]}: organ made see-through to show the holes it lies on", font_size=12, color="black")
        pl.camera_position = [tuple(O + 34 * Z + 14 * X - 16 * Y), tuple(O - 3 * Y), tuple(Y)]
        pl.screenshot(os.path.join(OUT, f"render_closeup_organ_seethrough_{design}.png"))
        pl.close()
        # 2: drain surface only, near the studied hole (design detail)
        pl = pv.Plotter(off_screen=True, window_size=(1600, 1100))
        pl.set_background("white")
        pl.add_mesh(near, color="#8a9bb0", smooth_shading=True, specular=0.45)
        ring = pv.Circle(radius=1.15 if design != "current" else 0.75, resolution=80)
        rp = ring.points @ np.c_[X, Y, Z].T + O + 0.6 * Z
        pl.add_mesh(pv.Spline(np.vstack([rp, rp[:1]]), 120).tube(radius=0.05), color="#ffcc00")
        pl.add_text(f"{NAMES[design]}: drain surface; the modelled hole is circled", font_size=12, color="black")
        pl.camera_position = [tuple(O + 9 * Z + 4 * X - 5 * Y), tuple(O), tuple(Y)]
        pl.screenshot(os.path.join(OUT, f"render_drain_surface_{design}.png"))
        pl.close()
        # 3: organ cut open through the hole, drain cut too (looking along the drain axis)
        pl = pv.Plotter(off_screen=True, window_size=(1600, 1100))
        pl.set_background("white")
        o_cut = organ.clip(normal=tuple(X), origin=tuple(O), invert=False)       # keep the +X half of the organ
        pl.add_mesh(near, color="#8a9bb0", smooth_shading=True, specular=0.35)
        pl.add_mesh(o_cut, scalars="s", cmap=["#6e1f2c", "#b5525f", "#e59aa4", "#f4c6cb"], show_scalar_bar=False,
                    smooth_shading=True)
        rp = pv.Circle(radius=1.15 if design != "current" else 0.75, resolution=80).points @ np.c_[X, Y, Z].T + O + 0.6 * Z
        pl.add_mesh(pv.Spline(np.vstack([rp, rp[:1]]), 120).tube(radius=0.06), color="#ffcc00")
        pl.add_text(f"{NAMES[design]}: organ cut in half through the modelled hole (circled)", font_size=12, color="black")
        pl.camera_position = [tuple(O - 26 * X + 18 * Z - 10 * Y), tuple(O + 2 * X - 2 * Y), tuple(Z)]
        pl.screenshot(os.path.join(OUT, f"render_cut_open_{design}.png"))
        pl.close()
        print("stills", design, flush=True)


# ---------------------------------------------------------------- gallery
ITEMS = [
    ("Real geometry (your CAD) with the sponge organ", [
        ("01_drain_with_sponge_organ_standoff.png", "New standoff-ring drain with the sponge organ lying on it."),
        ("video_turntable_drain_organ_standoff.mp4", "Turntable: standoff drain with the organ (real drain CAD)."),
        ("render_cut_open_standoff.png", "Organ cut in half through the modelled hole (circled): it rests on the ring tops."),
        ("render_closeup_organ_seethrough_standoff.png", "Organ made see-through: the ringed holes it lies on."),
        ("render_drain_surface_standoff.png", "Standoff drain surface: rings with 4 slots around every hole; modelled hole circled."),
        ("04_section_at_hole_standoff.png", "Exact cross-section at the hole (45 deg cut through two ring arcs), organ 50 um above the rings."),
        ("standoff_design.png", "Your standoff design (from new_offset_desing.zip)."),
        ("standoff_section.png", "Your standoff section (from new_offset_desing.zip)."),
        ("01_drain_with_sponge_organ_current.png", "Current dome-cap drain with the organ."),
        ("video_turntable_drain_organ_current.mp4", "Turntable: current drain with the organ."),
        ("render_cut_open_current.png", "Current drain: organ cut in half through the modelled hole."),
        ("render_closeup_organ_seethrough_current.png", "Current drain under the see-through organ."),
        ("render_drain_surface_current.png", "Current drain surface: plain holes."),
        ("04_section_at_hole_current.png", "Exact cross-section at the hole, current design."),
    ]),
    ("The modelled single hole", [
        ("02_hole_window_cutaway_standoff.png", "Single-hole window, standoff design: tissue, liquid gap, ring, hole, channel."),
        ("02_hole_window_cutaway_current.png", "Single-hole window, current design."),
        ("02_hole_window_cutaway_standoff_opt.png", "Single-hole window, optimized ring (0.50 mm high, wider, wider slots)."),
        ("03_mesh_section_standoff.png", "Computational mesh shared by OpenFOAM (liquid) and FEBio (tissue), standoff."),
        ("03_mesh_section_current.png", "Computational mesh, current design."),
    ]),
    ("What happens over time (model results)", [
        ("video_cross_section_closed.mp4", "Animation, no free liquid: current design vs the two ring designs side by side."),
        ("video_cross_section_open.mp4", "Animation, free wound liquid around the drain."),
        ("video_3d_current_closed.mp4", "3D animation: current design, the tissue is sucked into the hole."),
        ("video_3d_standoff_closed.mp4", "3D animation: standoff ring, the tissue rests on the ring."),
        ("video_3d_standoff_opt_closed.mp4", "3D animation: optimized ring."),
        ("video_cross_section_closed_last_frame.png", "After 6 h, no free liquid (last frame of the animation)."),
        ("render_3d_current_closed_end.png", "3D after 6 h: current design."),
        ("render_3d_standoff_closed_end.png", "3D after 6 h: standoff ring."),
        ("render_3d_standoff_opt_closed_end.png", "3D after 6 h: optimized ring."),
        ("09_design_comparison.png", "Comparison over 6 h: tissue position and flow into the hole."),
        ("06_fields_current_closed.png", "Current design: pressure and liquid flow inside the tissue (4 moments)."),
        ("06_fields_standoff_closed.png", "Standoff ring: pressure and liquid flow inside the tissue."),
        ("06_fields_standoff_opt_closed.png", "Optimized ring: pressure and liquid flow inside the tissue."),
        ("05_timeline_current_closed.png", "Current design: time history (no free liquid)."),
        ("05_timeline_standoff_closed.png", "Standoff ring: time history (no free liquid)."),
        ("05_timeline_standoff_opt_closed.png", "Optimized ring: time history (no free liquid)."),
        ("05_timeline_standoff_open.png", "Standoff ring: time history with free liquid."),
        ("05_timeline_current_open.png", "Current design: time history with free liquid."),
    ]),
    ("Design study", [
        ("08_standoff_screening.png", "45 ring variants: height, width, slot width."),
        ("07_parameter_map.png", "Current design: tissue stiffness x suction."),
    ]),
]


def gallery():
    for f in os.listdir(FIG):
        shutil.copy(os.path.join(FIG, f), OUT)
    if os.path.exists(os.path.join("out", "drainsponge_viewer.html")):
        shutil.copy(os.path.join("out", "drainsponge_viewer.html"), OUT)
    for name in ("standoff_design.png", "standoff_section.png"):
        p = os.path.join("data", name)
        if os.path.exists(p):
            shutil.copy(p, OUT)
    sec = []
    md = ["# Drain + sponge tissue: pictures and videos\n",
          "**Real**: the drain geometry is your CAD (dome-cap and standoff drains). The organ is a shape made here and placed "
          "on the drain in its real coordinates, 50 um above the skin (current design) or above the ring tops (standoff).\n",
          "**Model results**: the time histories, pressures and tissue movement come from the reduced-order model "
          "(axisymmetric, small deformation) with placeholder values: tissue stiffness 10 kPa, permeability 1e-14 m^4/(N s), "
          "suction 10 kPa. They show trends and design differences; the full 3D OpenFOAM + FEBio runs (hpc/) give the final numbers.\n"]
    for title, items in ITEMS:
        md.append(f"\n## {title}\n")
        cards = []
        for fn, cap in items:
            if not os.path.exists(os.path.join(OUT, fn)):
                continue
            md.append(f"- `{fn}`: {cap}")
            if fn.endswith(".mp4"):
                cards.append(f'<figure><video src="{fn}" controls loop muted playsinline preload="metadata"></video>'
                             f'<figcaption>{cap}</figcaption></figure>')
            else:
                cards.append(f'<figure><a href="{fn}"><img src="{fn}" loading="lazy" alt="{cap}"></a>'
                             f'<figcaption>{cap}</figcaption></figure>')
        sec.append(f"<h2>{title}</h2><div class='grid'>{''.join(cards)}</div>")
    open(os.path.join(OUT, "README.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Drain Sponge Gallery</title>
<style>
:root{{--bg:#f6f4f2;--panel:#fff;--ink:#1d2430;--muted:#5d6675;--line:#dfe3e8}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#14171c;--panel:#1c2027;--ink:#e7eaef;--muted:#9aa3b2;--line:#2c323c}}}}
:root[data-theme="dark"]{{--bg:#14171c;--panel:#1c2027;--ink:#e7eaef;--muted:#9aa3b2;--line:#2c323c}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}}
main{{max-width:1400px;margin:0 auto;padding:20px 16px 40px}}
h1{{font-size:22px;margin:0 0 6px}} h2{{font-size:17px;margin:28px 0 10px}}
p{{color:var(--muted);max-width:900px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:14px}}
figure{{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:8px}}
img,video{{width:100%;border-radius:6px;background:#fff;display:block}}
figcaption{{font-size:13px;color:var(--muted);padding:6px 2px 0}}
</style></head><body><main>
<h1>Drain + sponge-like tissue: pictures and videos</h1>
<p><b>Real</b>: drain geometry from your CAD; the organ is placed on it in real coordinates.
<b>Model</b>: time histories, pressures and tissue movement from the reduced-order model with placeholder tissue values
(stiffness 10 kPa, suction 10 kPa); the 3D OpenFOAM + FEBio runs on Euler give the final numbers.
Interactive 3D viewer (design / scenario switch, time slider): <a href="drainsponge_viewer.html">drainsponge_viewer.html</a>.</p>
{''.join(sec)}
</main></body></html>"""
    open(os.path.join(OUT, "index.html"), "w", encoding="utf-8").write(html)
    print("gallery written", flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    steps = {"data": make_data, "anim2d": anim2d, "anim3d": anim3d, "turntable": turntable,
             "stills": stills, "gallery": gallery}
    for k in (list(steps) if what == "all" else [what]):
        t = time.time()
        steps[k]()
        print(f"[{k}] {time.time()-t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
