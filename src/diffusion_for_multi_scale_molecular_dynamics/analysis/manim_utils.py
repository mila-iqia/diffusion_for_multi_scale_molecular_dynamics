import numpy as np
from manim import (BLUE, BLUE_E, GREEN, GREY_B, RED, WHITE, YELLOW, Arrow, Dot,
                   FadeOut, Group, GrowArrow, LaggedStartMap, Rectangle,
                   RoundedRectangle, Scene, SVGMobject, Text, VGroup, UP, DOWN, RIGHT)


def make_box_with_atoms(
    center: np.ndarray,
    width: float = 5.5,
    height: float = 3.2,
    n_atoms: int = 16,
    atom_radius: float = 0.14,
    atom_color=BLUE_E,
    seed: int = 2,
):
    """Return (box, atoms_group, positions[N,2], velocities[N,2])."""
    box = Rectangle(width=width, height=height, stroke_width=3).move_to(center)
    rng = np.random.default_rng(seed)
    pad = atom_radius * 3
    xs = rng.uniform(-width / 2 + pad, width / 2 - pad, n_atoms)
    ys = rng.uniform(-height / 2 + pad, height / 2 - pad, n_atoms)
    positions = np.stack([xs, ys], axis=1) + box.get_center()[:2]
    # small random velocities
    speeds = 0.7 * rng.normal(size=(n_atoms, 2))
    atoms = Group(
        *[
            Dot(
                point=[positions[i, 0], positions[i, 1], 0],
                radius=atom_radius,
                color=atom_color,
            ).set_z_index(1)
            for i in range(n_atoms)
        ]
    )
    # subtle breathing highlight (optional)
    for d in atoms:
        d.set_stroke(WHITE, width=1, opacity=0.5)
        d.set_fill(d.get_color(), opacity=0.95)
    return box, atoms, positions, speeds


def build_oracle_panel(center: np.ndarray, width: float = 4.8, height: float = 3.2):
    """Right-side oracle card with mode toggles and a vertical 'cost' meter.
    Returns (panel_group, set_mode_fn, cost_fill_rect)."""
    card = RoundedRectangle(width=width, height=height, corner_radius=0.2)
    title = Text("Force Oracle", font_size=30).next_to(card, UP, buff=0.25)
    mode_dft = Text("DFT", font_size=28)
    mode_pot = Text("Potential", font_size=28)
    tabs = (
        VGroup(mode_dft, mode_pot)
        .arrange(RIGHT, buff=0.6)
        .next_to(card.get_top(), DOWN, buff=0.45)
    )
    # chip = SVGMobject()  # empty; keep layout consistent
    chip = RoundedRectangle(width=1.0, height=0.6, corner_radius=0.1, fill_opacity=0.2, color=WHITE)
    chip_rect = RoundedRectangle(
        width=width * 0.76, height=height * 0.46, corner_radius=0.12, stroke_width=2
    ).move_to(card.get_center() + 0.1 * DOWN)
    chip_label = Text("E = f(positions)", font_size=26).move_to(chip_rect.get_center())
    # cost meter
    meter_border = Rectangle(width=0.35, height=height * 0.55).next_to(
        card, RIGHT, buff=0.35
    )
    meter_label = Text("cost", font_size=22).next_to(meter_border, DOWN, buff=0.2)
    cost_fill = Rectangle(
        width=meter_border.width - 0.04, height=0.001, fill_opacity=0.9
    ).set_fill(RED)
    cost_fill.move_to(
        meter_border.get_bottom() + np.array([0, cost_fill.height / 2 + 0.02, 0])
    )
    panel = Group(
        card, title, tabs, chip_rect, chip_label, meter_border, meter_label, cost_fill
    ).move_to(center)

    # mode styles
    def set_mode(mode: str):
        if mode.lower().startswith("dft"):
            mode_dft.set_color(YELLOW).set_opacity(1.0)
            mode_pot.set_color(GREY_B).set_opacity(0.8)
        else:
            mode_dft.set_color(GREY_B).set_opacity(0.8)
            mode_pot.set_color(GREEN).set_opacity(1.0)

    set_mode("Potential")
    return panel, set_mode, cost_fill


def compute_toy_forces(
    positions: np.ndarray, box_rect: Rectangle, k_rep: float = 0.10, cutoff: float = 0.7
):
    """Very lightweight pairwise repulsion with cutoff; walls push inward."""
    N = positions.shape[0]
    forces = np.zeros_like(positions)

    # Pairwise repulsion
    for i in range(N):
        pi = positions[i]
        # crude neighbor sampling to keep it cheap
        for j in range(i + 1, N):
            d = positions[j] - pi
            dist = np.linalg.norm(d) + 1e-6
            if dist < cutoff:
                f = k_rep * d / (dist**3)  # ~1/r^2 repulsion along direction
                forces[i] -= f
                forces[j] += f

    # Wall forces (soft)
    cx, cy, _ = box_rect.get_center()
    hw, hh = box_rect.width / 2, box_rect.height / 2
    x = positions[:, 0] - cx
    y = positions[:, 1] - cy
    # push back proportionally when outside a soft margin
    wall_k = 0.8
    margin = 0.0
    forces[:, 0] += wall_k * np.tanh((x - (hw - margin)) * 3)
    forces[:, 0] += -wall_k * np.tanh((x + (hw - margin)) * 3)
    forces[:, 1] += wall_k * np.tanh((y - (hh - margin)) * 3)
    forces[:, 1] += -wall_k * np.tanh((y + (hh - margin)) * 3)
    return forces


def integrate_step(positions: np.ndarray, velocities: np.ndarray, forces: np.ndarray,
                   dt: float = 0.08, damping: float = 0.98,
                   box_rect: Rectangle = None):
    """Simple VV-like update + elastic wall bounce."""
    v = velocities * damping + forces * dt
    p = positions + v * dt

    # hard bounce on walls
    cx, cy, _ = box_rect.get_center()
    hw, hh = box_rect.width/2, box_rect.height/2
    for i in range(p.shape[0]):
        if p[i,0] < cx - hw: p[i,0] = cx - hw; v[i,0] *= -0.7
        if p[i,0] > cx + hw: p[i,0] = cx + hw; v[i,0] *= -0.7
        if p[i,1] < cy - hh: p[i,1] = cy - hh; v[i,1] *= -0.7
        if p[i,1] > cy + hh: p[i,1] = cy + hh; v[i,1] *= -0.7
    return p, v


def draw_force_arrows(
    scene: Scene,
    atoms: Group,
    forces: np.ndarray,
    scale: float = 0.7,
    color_attract=BLUE,
    color_repel=RED,
    run_time: float = 0.4,
):
    """Short-lived arrows indicating net forces on each atom."""
    arrows = Group()
    for atom, f in zip(atoms, forces):
        if np.allclose(f, 0, atol=1e-4):
            continue
        start = atom.get_center()
        end = start + np.array([f[0], f[1], 0]) * scale
        col = color_repel if (f[0] ** 2 + f[1] ** 2) > 0 else color_attract
        arrows.add(
            Arrow(
                start,
                end,
                buff=0.06,
                max_tip_length_to_length_ratio=0.25,
                stroke_width=3,
                color=col,
            )
        )
    if len(arrows) == 0:
        return
    scene.play(
        LaggedStartMap(GrowArrow, arrows, lag_ratio=0.08), run_time=run_time * 0.6
    )
    scene.play(FadeOut(arrows, lag_ratio=0.08), run_time=run_time * 0.4)
