"""Script to generate animation for excise and repaint with a 2D square lattice.

To generate the animation, run the following command:

manim -pqh src/diffusion_for_multi_scale_molecular_dynamics/analysis/excise_and_repaint_animation.py \
    ExciseAndRepaint2DToyModel

"""

from typing import List

import numpy as np
from manim import (
    BLUE,
    GREEN,
    ORIGIN,
    RED,
    RIGHT,
    UP,
    WHITE,
    YELLOW,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    Scene,
    Square,
    Text,
    Transform,
    VGroup,
)

BOX_SIZE = 5.0  # Display size of the box in Manim units
ATOM_RADIUS = 0.1
SMALL_BOX_SIZE = 4.0
PADDING = 0.4  # 10% padding inside small box
BOX_GAP = 0.5  # spacing between large and small boxes
CORNER_PADDING = 0.6  # adjustable: distance from box edge to each new atom


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

    def construct(self):
        self.display_initial_state()
        self.wait(1)  # Optional pause

        center_idx = len(self.positions) // 2  # Center of 5x5 grid = index 12
        neighbors = find_k_nearest_neighbors(center_idx, self.positions, k=4)
        self.highlight_neighbors(center_idx, neighbors)

        self.wait(1)  # Optional pause

        keep = [center_idx] + neighbors
        small_box = self.move_atoms_to_smaller_box(keep)
        self.wait(1.5)

        self.finalize_small_system(keep, small_box)
        self.wait(3)
