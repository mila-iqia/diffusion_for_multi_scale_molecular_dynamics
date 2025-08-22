"""Script to generate animation for excise and repaint with a 2D square lattice.

To generate the animation, run the following command:

manim -pql src/diffusion_for_multi_scale_molecular_dynamics/analysis/excise_and_repaint_animation.py

(replace l with h for a high-quality video)
"""

import math
from typing import List

import numpy as np
from manim import (BLUE, BLUE_E, DOWN, GREEN, GREY_B, LEFT, ORIGIN, PI, RED,
                   RIGHT, UP, WHITE, YELLOW, AnimationGroup, Axes, Brace,
                   Circle, Create, DashedLine, Dot, FadeIn, FadeOut, Group,
                   LaggedStart, Line, Rectangle, Scene, Square, Text,
                   Transform, ValueTracker, VGroup, always_redraw, config,
                   linear, smooth)

from diffusion_for_multi_scale_molecular_dynamics.analysis.manim_utils import (
    compute_toy_forces, draw_force_arrows_snapshot, integrate_step,
    make_box_with_atoms)

BOX_SIZE = 5.0  # Display size of the box in Manim units
ATOM_RADIUS = 0.1
SMALL_BOX_SIZE = 4.0
PADDING = 0.4  # 10% padding inside small box
BOX_GAP = 0.5  # spacing between large and small boxes
CORNER_PADDING = 0.6  # adjustable: distance from box edge to each new atom
SHORT_DELAY = 1  # time between two steps - in second
MEDIUM_DELAY = 2
LONG_DELAY = 3


def generate_2d_grid_positions(n: int, margin: float = 0.1) -> np.ndarray:
    # This is a toy grid of atoms for a first demo
    lin = np.linspace(margin, 1 - margin, n)
    grid_x, grid_y = np.meshgrid(lin, lin)
    positions = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    return positions  # Shape: (n*n, 2)


def create_atoms(positions: np.ndarray, box_size: float = BOX_SIZE) -> VGroup:
    atoms = VGroup()
    for pos in positions:
        display_pos = (pos - 0.5) * box_size  # Center around origin
        atom = Dot(point=[*display_pos, 0], radius=ATOM_RADIUS, color=BLUE)
        atoms.add(atom)
    return atoms


def create_box(box_size: float = BOX_SIZE) -> Square:
    return Square(side_length=box_size, color=WHITE).move_to(ORIGIN)


def find_k_nearest_neighbors(
    center_idx: int, positions: np.ndarray, k: int
) -> List[int]:
    center = positions[center_idx]
    distances = np.linalg.norm(positions - center, axis=1)
    sorted_indices = np.argsort(distances)
    neighbors = [i for i in sorted_indices if i != center_idx][:k]
    return neighbors


class ExciseAndRepaint2DToyModel(Scene):
    def display_initial_state(self, n=5, margin=0.1):
        self.positions = generate_2d_grid_positions(n, margin)
        self.atom_group = create_atoms(self.positions)
        self.box = create_box()

        self.add(self.box, self.atom_group)

        self.step_title = Text("Initial configuration").next_to(self.box, UP)
        self.add(self.step_title)

    def highlight_neighbors(self, center_idx: int, neighbor_indices: List[int]):
        center_atom = self.atom_group[center_idx]
        neighbor_atoms = [self.atom_group[i] for i in neighbor_indices]

        # Create new step title
        new_title = Text("Atom with high uncertainty").next_to(self.box, UP)

        # Animate both title change and atom color updates together
        self.play(
            Transform(self.step_title, new_title),
            center_atom.animate.set_color(RED),
            *[atom.animate.set_color(YELLOW) for atom in neighbor_atoms],
        )

    def excise_environment(self, keep_indices: List[int]):
        # Determine which atoms to fade out
        all_indices = list(range(len(self.atom_group)))
        fade_indices = [i for i in all_indices if i not in keep_indices]

        fade_atoms = [self.atom_group[i] for i in fade_indices]

        # New label
        new_title = Text("Excise the Environment").next_to(self.box, UP)

        # Animate fade-out and label transform in sync
        self.play(
            *[FadeOut(atom) for atom in fade_atoms],
            Transform(self.step_title, new_title),
        )

    def move_atoms_to_smaller_box(self, keep_indices: List[int]):
        new_title = Text("Excise and Embed in a Smaller Unit Cell").next_to(
            self.box, UP
        )

        # Create small box, closer to center
        right_offset = BOX_SIZE / 2 + SMALL_BOX_SIZE / 2 + 0.5
        small_box = Square(side_length=SMALL_BOX_SIZE, color=WHITE)
        small_box.move_to(self.box.get_center() + RIGHT * right_offset)
        self.play(Create(small_box), Transform(self.step_title, new_title))

        # Compute original positions in display space
        original_scene_positions = []
        for i in keep_indices:
            normalized = self.positions[i]
            display_pos = (normalized - 0.5) * BOX_SIZE
            original_scene_positions.append(display_pos)

        # Compute bounding box of these positions
        original_scene_positions = np.array(original_scene_positions)
        min_pos = original_scene_positions.min(axis=0)
        max_pos = original_scene_positions.max(axis=0)
        center = (min_pos + max_pos) / 2
        span = max_pos - min_pos
        span[span == 0] = 1  # avoid divide-by-zero

        # Determine scale factor to fit into small box with padding
        available_span = SMALL_BOX_SIZE * (1 - PADDING)
        scale = available_span / span.max()

        # Scale and translate into small box
        target_positions = (
            original_scene_positions - center
        ) * scale + small_box.get_center()[:2]

        # Animate atoms into new positions
        animations = []
        for atom_idx, target_xy in zip(keep_indices, target_positions):
            atom = self.atom_group[atom_idx]
            animations.append(atom.animate.move_to([*target_xy, 0]))

        self.play(*animations)

        return small_box

    def finalize_small_system(self, keep_indices: List[int], small_box: Square):
        # 1. Fade out the large box and non-selected atoms
        self.play(
            FadeOut(self.box),
            *[
                FadeOut(atom)
                for i, atom in enumerate(self.atom_group)
                if i not in keep_indices
            ],
        )

        # 2. Shift small box and atoms to center
        shift_vector = ORIGIN - small_box.get_center()
        self.play(
            small_box.animate.move_to(ORIGIN),
            *[self.atom_group[i].animate.shift(shift_vector) for i in keep_indices],
        )

        # 3. Create new label AFTER shift, and place it relative to new position
        new_title = Text("Repaint Missing Atoms").next_to(small_box, UP)
        self.play(Transform(self.step_title, new_title))

        # 4. Add 4 new atoms in corners
        half_size = SMALL_BOX_SIZE / 2 - CORNER_PADDING
        corner_offsets = [
            np.array([+half_size, +half_size, 0]),
            np.array([-half_size, +half_size, 0]),
            np.array([-half_size, -half_size, 0]),
            np.array([+half_size, -half_size, 0]),
        ]
        corner_positions = [
            small_box.get_center() + offset for offset in corner_offsets
        ]

        new_atoms = VGroup()
        for pos in corner_positions:
            new_atom = Dot(point=pos, radius=ATOM_RADIUS, color=GREEN)
            new_atoms.add(new_atom)

        self.play(FadeIn(new_atoms))

    def animate_excise_and_repaint(self):
        # Initial setup
        self.display_initial_state(n=5, margin=0.1)
        self.wait(SHORT_DELAY)
        # Select center atom and its neighbors
        # center_idx = 12  # Center atom (can be parameterized)
        center_idx = len(self.positions) // 2  # Center of 5x5 grid = index 12
        neighbor_indices = find_k_nearest_neighbors(center_idx, self.positions, k=4)
        keep_indices = [center_idx] + neighbor_indices
        # Run the animation sequence
        self.highlight_neighbors(center_idx, neighbor_indices)
        self.wait(SHORT_DELAY)
        small_box = self.move_atoms_to_smaller_box(keep_indices)
        self.wait(MEDIUM_DELAY)
        self.finalize_small_system(keep_indices, small_box)
        self.wait(LONG_DELAY)

    def reset_scene(self, animate_fade: bool = True):
        """Clear all mobjects, stop updaters, and reset the camera frame."""
        if not self.mobjects:
            self._reset_camera_frame()
            return

        # Snapshot all current mobjects (avoid mutating while iterating)
        objs = list(self.mobjects)

        # Stop updaters on everything
        for m in objs:
            m.clear_updaters()

        # Use Group (not VGroup) so non-VMobjects are supported
        bucket = Group(*objs)

        if animate_fade:
            self.play(FadeOut(bucket, lag_ratio=0.05))
        # Remove everything from the scene
        self.remove(*objs)
        self.clear()  # clean internal bookkeeping

        # Reset camera (harmless if not a MovingCameraScene)
        self._reset_camera_frame()

    def _reset_camera_frame(self):
        if hasattr(self, "camera") and hasattr(self.camera, "frame"):
            frame = self.camera.frame
            frame.set_width(config.frame_width)
            frame.set_height(config.frame_height)
            frame.move_to(ORIGIN)

    def intro_energy_barrier(
        self,
        box_width=5.5,
        box_height=3.0,
        n_bg_atoms=18,
        ion_color=YELLOW,
        bg_atom_color=BLUE_E,
        run_time=5.0,
        title_text: str | None = "ART Nouveau",
    ):
        title = Text(title_text, font_size=32).to_edge(UP) if title_text else None
        if title:
            self.play(FadeIn(title), run_time=0.25)
        # Layout anchors
        left_panel_center = 3.2 * LEFT
        right_panel_center = 3.6 * RIGHT

        # 1) Left: material box + atoms
        box = Rectangle(width=box_width, height=box_height, stroke_width=3).move_to(
            left_panel_center
        )
        self.add(box)

        # Background atoms (static)
        rng = np.random.default_rng(7)
        atoms = VGroup()
        # Leave a horizontal corridor for the ion by biasing y toward center
        for _ in range(n_bg_atoms):
            x = rng.uniform(-box_width / 2 + 0.35, box_width / 2 - 0.35)
            # push background atoms away from the transit lane (y≈0) a bit
            y_raw = rng.uniform(-box_height / 2 + 0.35, box_height / 2 - 0.35)
            y = math.copysign(1, y_raw) * max(0.2 * box_height, abs(y_raw))
            c = Circle(radius=0.15, color=bg_atom_color, fill_opacity=0.85).move_to(
                box.get_center() + np.array([x, y, 0])
            )
            atoms.add(c)
        self.add(atoms)

        # Ion to travel across
        ion_radius = 0.20
        ion = Circle(radius=ion_radius, color=ion_color, fill_opacity=1.0).set_stroke(
            width=4
        )
        # Path endpoints (inside the box, with a bit of padding)
        pad = 0.35
        xL = box.get_left()[0] + pad
        xR = box.get_right()[0] - pad
        y_lane = box.get_center()[1] + 0.00
        ion.move_to(np.array([xL, y_lane, 0]))
        self.add(ion)

        # A faint dashed lane to imply channel
        lane = DashedLine(
            start=np.array([xL, y_lane, 0]),
            end=np.array([xR, y_lane, 0]),
            dash_length=0.15,
            color=GREY_B,
        )
        self.add(lane)

        # 2) Right: energy profile (barrier)
        axes = Axes(
            x_range=[0, 1, 0.2],
            y_range=[0, 3.5, 0.5],
            x_length=5.5,
            y_length=3.3,
            tips=False,
            axis_config={"include_numbers": False, "stroke_width": 2},
        ).move_to(right_panel_center)

        x_label = Text("position", font_size=28).next_to(axes, DOWN, buff=0.35)
        y_label = (
            Text("energy", font_size=28)
            .next_to(axes.y_axis, LEFT, buff=0.3)
            .rotate(PI / 2)
        )
        plot_title = Text("Energy barrier", font_size=32).next_to(axes, UP, buff=0.35)
        self.add(axes, x_label, y_label, plot_title)

        # Barrier function: smooth bump centered at 0.5 (adjust amplitudes as desired)
        def barrier(x):
            # Base slope term + Gaussian bump
            bump = 2.6 * np.exp(-(((x - 0.5) / 0.14) ** 2))
            base = 0.4 + 0.2 * (x - 0.5)
            return bump + base

        graph = axes.plot(barrier, x_range=[0, 1], stroke_width=6, color=RED)
        self.add(graph)

        # 3) Synchronization between left (ion) and right (energy)
        progress = ValueTracker(0.0)  # 0 -> 1

        # Helpers to map progress → positions
        def pos_from_progress(p):
            x = (1 - p) * xL + p * xR
            return np.array([x, y_lane, 0])

        def curve_point_from_progress(p):
            x = p  # reaction coordinate ∈ [0,1]
            y = barrier(x)
            return axes.coords_to_point(x, y)

        # Tracers on the energy plot
        tracer_dot = always_redraw(
            lambda: Dot(
                curve_point_from_progress(progress.get_value()),
                radius=0.06,
                color=YELLOW,
            )
        )
        vline = always_redraw(
            lambda: DashedLine(
                start=axes.coords_to_point(progress.get_value(), 0),
                end=curve_point_from_progress(progress.get_value()),
                dash_length=0.08,
                color=GREY_B,
            )
        )

        # A brace + label to emphasize the barrier height
        x_peak = 0.5
        peak_point = axes.coords_to_point(x_peak, barrier(x_peak))
        base_point = axes.coords_to_point(x_peak, 0)
        brace = Brace(
            Line(base_point, peak_point), direction=RIGHT, color=WHITE, buff=0.08
        )
        brace_label = brace.get_text("activation energy")  # font_size=28)
        # self.add(tracer_dot, vline)
        self.add(tracer_dot, vline, brace, brace_label)

        # Keep ion and tracers updated
        ion.add_updater(lambda m: m.move_to(pos_from_progress(progress.get_value())))

        # 4) Animate: ion crosses while energy tracer climbs and descends the barrier
        self.play(progress.animate.set_value(1.0), run_time=run_time, rate_func=smooth)

        # 5) Clean up updaters to avoid side-effects if you continue the scene
        ion.clear_updaters()

    def intro_md(
        self,
        n_atoms: int = 18,
        steps_per_cycle: int = 3,
        cycles: int = 3,
        dt: float = 0.11,
        jump_time: float = 0.35,  # shorter = choppier
        pause_time: float = 1.2,  # commentary window
        max_force_arrows: int | None = 10,  # None = all atoms
        title_text: str | None = "Molecular dynamics",
        end_pause_time: float = 2.4,
    ):
        # Centered, larger box
        box, atoms, pos, vel = make_box_with_atoms(
            center=ORIGIN,
            width=7.0,
            height=4.2,
            n_atoms=n_atoms,
            atom_radius=0.16,
            atom_color=BLUE_E,
            seed=9,
        )

        title = Text(title_text, font_size=32).to_edge(UP) if title_text else None
        if title:
            self.play(FadeIn(title), run_time=0.25)
        self.play(
            FadeIn(box),
            LaggedStart(*[FadeIn(a, scale=0.8) for a in atoms], lag_ratio=0.03),
            run_time=0.7,
        )

        # Optional: highlight one atom with a short trail to help the eye
        # atoms[0].set_color(YELLOW).set_stroke(WHITE, 2)
        # trail = TracedPath(atoms[0].get_center, stroke_opacity=[0.7, 0.0],
        #                   stroke_width=3, dissipating_time=1.4, z_index=0.5)
        # self.add(trail)

        # Helper to perform one discrete “frame” of motion
        def jump_once(p, v):
            f = compute_toy_forces(p, box)
            p2, v2 = integrate_step(p, v, f, dt=dt, damping=0.985, box_rect=box)
            # move atoms in one go (choppy effect)
            anims = [
                atoms[i].animate.move_to([p2[i, 0], p2[i, 1], 0])
                for i in range(len(atoms))
            ]
            self.play(
                AnimationGroup(*anims, lag_ratio=0.0),
                run_time=jump_time,
                rate_func=linear,
            )
            return p2, v2

        # Main: a few jumps, then pause with forces; repeat
        for c in range(cycles):
            for _ in range(steps_per_cycle):
                pos, vel = jump_once(pos, vel)
            forces = compute_toy_forces(pos, box)
            draw_force_arrows_snapshot(
                self,
                atoms,
                forces,
                scale=0.4,
                max_arrows=max_force_arrows,
                fade_time=0.25 if c < cycles - 1 else 0.1,
                hold=pause_time if c < cycles - 1 else end_pause_time,
            )

        if title:
            self.play(FadeOut(title), run_time=0.2)

    def construct(self):
        self.intro_md(
            pause_time=SHORT_DELAY, steps_per_cycle=1, end_pause_time=LONG_DELAY
        )
        self.reset_scene()
        self.intro_energy_barrier()
        # self.wait(LONG_DELAY)
        # self.reset_scene()
        # self.wait(SHORT_DELAY)
        # self.animate_excise_and_repaint()
        self.wait(LONG_DELAY)
