from typing import List, Tuple, Union

import abc
import dataclasses
import math
import numpy as np
import pygame

CoordType = Union[Tuple[float, float], List[float], np.ndarray]
ArrayLike = Union[List[CoordType], Tuple[CoordType]]


def make_rect(width: float, height: float, outline: bool, dashed: bool = False):
    rad_h = height / 2
    rad_w = width / 2
    points = [
        (-rad_w, rad_h),
        (rad_w, rad_h),
        (rad_w, -rad_h),
        (-rad_w, -rad_h)]
    poly = Poly(points, outline)
    if dashed:
        poly.dashed = True
    return poly


def make_circle(radius, res, outline):
    points = []
    for i in range(res):
        ang = 2 * math.pi * i / res
        points.append((math.cos(ang) * radius, math.sin(ang) * radius))
    return Poly(points, outline)


def make_square(side_length, outline):
    return make_rect(side_length, side_length, outline)


@dataclasses.dataclass
class Transform:
    matrix: np.ndarray

    @classmethod
    def from_matrix(cls, matrix: np.ndarray):
        tr = cls()
        tr.matrix = matrix
        return tr

    @staticmethod
    def create_translation_matrix(translation=(0, 0)):
        return np.asarray([
            [1.0, 0.0, translation[0]],
            [0.0, 1.0, translation[1]],
            [0.0, 0.0, 1.0],
        ])

    @staticmethod
    def create_rotation_matrix(rotation):
        cos = math.cos(rotation)
        sin = math.sin(rotation)
        return np.asarray([
            [cos, -sin, 0.0],
            [sin, cos, 0.0],
            [0.0, 0.0, 1.0],
        ])

    @staticmethod
    def create_scaling_matrix(scale):
        return np.asarray([
            [scale[0], 0.0, 0.0],
            [0.0, scale[1], 0.0],
            [0.0, 0.0, 1.0],
        ])

    @staticmethod
    def rigid_transform(pts: np.ndarray, transform: np.ndarray):
        """Apply a rigid transform on 2D points."""
        assert transform.shape == (3, 3)
        was_2d = np.asarray(pts).ndim > 1
        pts = np.atleast_2d(pts)
        pts_h = np.hstack([pts, np.ones((len(pts), 1))])
        pts_tr = (transform @ pts_h.T).T
        if was_2d:
            return pts_tr[:, :2]
        return tuple(pts_tr.ravel()[:2])

    def __init__(
        self,
        translation: CoordType = np.zeros(2),
        rotation: float = 0.0,
        scale: CoordType = np.ones(2),
    ):
        translation_matrix = Transform.create_translation_matrix(translation)
        rotation_matrix = Transform.create_rotation_matrix(rotation)
        scaling_matrix = Transform.create_scaling_matrix(scale)
        # The order of operations is scale, rotate then translate.
        self.matrix = translation_matrix @ rotation_matrix @ scaling_matrix

    def reset(
        self,
        translation: np.ndarray = np.zeros(2),
        rotation: float = 0.0,
        scale: np.ndarray = np.ones(2),
    ):
        self.__init__(translation, rotation, scale)

    def left_multiply(self, transform: "Transform"):
        """Multiply the matrix on the left by a transform."""
        matrix = transform.matrix @ self.matrix
        return self.from_matrix(matrix)

    def right_multiply(self, transform: "Transform"):
        """Multiply the matrix on the right by a transform."""
        matrix = self.matrix @ transform.matrix
        return self.from_matrix(matrix)

    # Method aliases.
    post_multiply = left_multiply
    pre_multiply = right_multiply


@dataclasses.dataclass
class Stack:
    stack: List[np.ndarray] = dataclasses.field(
        default_factory=lambda: [np.eye(3)])

    def push(self, transform: Transform):
        self.stack.append(self.stack[-1] @ transform.matrix)

    def pop(self):
        self.stack.pop()

    def apply_current_matrix(self, pts: np.ndarray):
        return Transform.rigid_transform(pts, self.stack[-1])


class Geom(abc.ABC):
    def __init__(self):
        self._color = (0, 0, 0, 1)
        self._outline_color = (0, 0, 0, 1)
        self.transforms = []

        self.initial_pts = None  # The initial points.
        self.geom = None  # The transformed points that get rendered.

    @staticmethod
    def convert_color(r: float, g: float, b: float):
        rgb = np.asarray([r, g, b]) * 255
        return tuple(np.round(rgb)) + (1,)

    def render(self, surface: pygame.Surface, stack: Stack):
        for transform in reversed(self.transforms):
            stack.push(transform)
        self.geom = stack.apply_current_matrix(
            np.array(self.initial_pts, copy=True))
        self._render(surface)
        for _ in self.transforms:
            stack.pop()

    @abc.abstractmethod
    def _render(self, surface: pygame.Surface):
        pass

    def add_transform(self, transform: Transform):
        self.transforms.append(transform)

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, value):
        self._color = self.convert_color(*value)

    @property
    def outline_color(self):
        return self._outline_color

    @outline_color.setter
    def outline_color(self, value):
        self._outline_color = self.convert_color(*value)


class Compound(Geom):
    def __init__(self, gs):
        super().__init__()

        self.gs = gs

    def add_transform(self, transform: Transform):
        for g in self.gs:
            g.add_transform(transform)

    def _render(self, surface: pygame.Surface):
        raise NotImplementedError

    def render(self, surface: pygame.Surface, stack: Stack):
        for g in self.gs:
            g.render(surface, stack)


class Poly(Geom):
    """A polygon defined by a list of vertices."""

    def __init__(self, pts: ArrayLike, outline: bool):
        super().__init__()

        self.outline = outline
        self.initial_pts = np.array(pts)
        self.dashed = False

    def _render(self, surface: pygame.Surface):
        ps = self.geom.tolist()
        ps += [ps[0]]

        pygame.draw.polygon(surface, self._color, ps)
        if self.outline:
            for i in range(len(self.geom)):
                a = self.geom[i]
                b = self.geom[(i + 1) % len(self.geom)]
                self.draw_outline(
                    surface,
                    a,
                    b,
                    1,
                    self._outline_color,
                    self.dashed)

    @staticmethod
    def draw_outline(surface, a, b, radius, fill_color, dashed):
        """Modified from https://codereview.stackexchange.com/q/70143"""
        if dashed:
            x1, y1 = a
            x2, y2 = b
            dl = 10
            if (x1 == x2):
                ycoords = [y for y in np.arange(y1, y2, dl if y1 < y2 else -dl)]
                xcoords = [x1] * len(ycoords)
            elif (y1 == y2):
                xcoords = [x for x in np.arange(x1, x2, dl if x1 < x2 else -dl)]
                ycoords = [y1] * len(xcoords)
            else:
                a = abs(x2 - x1)
                b = abs(y2 - y1)
                c = round(math.sqrt(a**2 + b**2))
                dx = dl * a / c
                dy = dl * b / c
                xcoords = [x for x in np.arange(x1, x2, dx if x1 < x2 else -dx)]
                ycoords = [y for y in np.arange(y1, y2, dy if y1 < y2 else -dy)]
            next_coords = list(zip(xcoords[1::2], ycoords[1::2]))
            last_coords = list(zip(xcoords[0::2], ycoords[0::2]))
            for (x1, y1), (x2, y2) in zip(next_coords, last_coords):
                start = (round(x1), round(y1))
                end = (round(x2), round(y2))
                pygame.draw.line(surface, fill_color, start, end, 4)
        else:
            p1 = a
            p2 = b
            r = round(max(1, radius * 2))
            pygame.draw.lines(surface, fill_color, False, [p1, p2], r)
            if r > 2:
                orthog = [abs(p2[1] - p1[1]), abs(p2[0] - p1[0])]
                if orthog[0] == 0 and orthog[1] == 0:
                    return
                scale = radius / (
                    orthog[0] * orthog[0] + orthog[1] * orthog[1]) ** 0.5
                orthog[0] = round(orthog[0] * scale)
                orthog[1] = round(orthog[1] * scale)
                points = [
                    (p1[0] - orthog[0], p1[1] - orthog[1]),
                    (p1[0] + orthog[0], p1[1] + orthog[1]),
                    (p2[0] + orthog[0], p2[1] + orthog[1]),
                    (p2[0] - orthog[0], p2[1] - orthog[1]),
                ]
                pygame.draw.polygon(surface, fill_color, points)
                pygame.draw.circle(
                    surface,
                    fill_color,
                    (round(p1[0]), round(p1[1])),
                    round(radius),
                )
                pygame.draw.circle(
                    surface,
                    fill_color,
                    (round(p2[0]), round(p2[1])),
                    round(radius),
                )


def ego_cam_matrix(
    centre: CoordType,
    new_pos: CoordType,
    rotation: float,
    scale: CoordType,
):
    """Create an ego-centric top-down camera."""
    scale = Transform(scale=(scale[0], scale[1]))
    tr1 = Transform(translation=(new_pos[0], new_pos[1]))
    rot = Transform(rotation=-rotation)
    tr2 = Transform(translation=(-centre[0], -centre[1]))
    return tr2 \
        .post_multiply(rot) \
        .post_multiply(tr1) \
        .post_multiply(scale)


class Viewer:
    """A headless viewer that uses a Pygame surface to render."""

    def __init__(
            self,
            width: int,
            height: int,
            background_rgb: Tuple[float, ...] = (1, 1, 1),
            easy_visuals=False,
    ):
        self.width = width
        self.height = height
        self.background_rgb = Geom.convert_color(*background_rgb)
        self.geoms = []
        self.easy_visuals = easy_visuals

        # Replicating OpenGL's rigid transform stack.
        self.stack = Stack()
        self.transform = None

        # Pygame uses a coordinate system where y points down. But pygame gives
        # us coordinates in a system where y points up. Thus, we need to flip
        # the coordinates up [(x, y) -> (x, height - y)] with an extra
        # transformation matrix.
        self.pygame_transform = Transform(scale=(1.0, -1.0)) \
            .post_multiply(Transform(translation=(0, self.height)))

        # Initialize the screen and fill the background color.
        self.screen = pygame.Surface((self.height, self.width))

    def _clear(self):
        """Clears the screen by filling it with the background color."""
        self.screen.fill(self.background_rgb)

    def draw_grid(self):
        """ Draws a grid on the screen. """
        grid_size = self.width // 3

        # Draw vertical lines
        for i in range(1, 3):
            pygame.draw.line(self.screen, (0,0,0), (i * grid_size, 0), (i * grid_size, self.height))

        # Draw horizontal lines
        for i in range(1, 3):
            pygame.draw.line(self.screen, (0,0,0), (0, i * grid_size), (self.width, i * grid_size))

        row_labels = ['1', '2', '3']
        col_labels = ['A', 'B', 'C']
        pygame.font.init()
        font = pygame.font.SysFont(None, 35)
        for index, label in enumerate(row_labels):
            text = font.render(label, True, (0,0,0))
            # This positions the labels to the right of each row.
            position = (self.width - text.get_width(), index * grid_size + grid_size // 2 - text.get_height() // 2)
            self.screen.blit(text, position)

        # Draw column labels (bottom)
        for index, label in enumerate(col_labels):
            text = font.render(label, True, (0,0,0))
            # This should position the labels at the bottom of each column.
            # Adjust the vertical position if necessary to ensure visibility.
            position = (index * grid_size + grid_size // 2 - text.get_width() // 2, self.height - text.get_height()+3)
            self.screen.blit(text, position)


    def set_bounds(self, left, right, bottom, top):
        assert right > left and top > bottom
        scale_x = self.width / (right - left)
        scale_y = self.height / (top - bottom)
        camera_transform = Transform(
            scale=(scale_x, scale_y),
            translation=(-left * scale_x, -bottom * scale_y)
        )
        self.transform = camera_transform.post_multiply(self.pygame_transform)

    def set_cam_follow(self,
                       source_xy_world,
                       target_xy_01,
                       viewport_hw_world,
                       rotation):
        """Set camera so that point at `source_xy_world` (in world coordinates)
        appears at `screen_target_xy` (in screen coordinates, in [0,1] on each
        axis), and so that viewport covers region defined by
        `viewport_hw_world` in world coordinates. Oh, and the world is rotated
        by `rotation` around the point `source_xy_world` before doing anything
        else.
        """
        world_h, world_w = viewport_hw_world
        scale_x = self.width / world_w
        scale_y = self.height / world_h
        target_x_01, target_y_01 = target_xy_01
        camera_transform = ego_cam_matrix(
            centre=source_xy_world,
            new_pos=(world_w * target_x_01, world_h * target_y_01),
            scale=(scale_x, scale_y),
            rotation=rotation,
        )
        self.transform = camera_transform.post_multiply(self.pygame_transform)

    def reset_geoms(self):
        """Clears the stored list of Geom objects."""
        self.geoms = []

    def add_geom(self, geom: Geom):
        """Adds a Geom object to draw to the screen."""
        self.geoms.append(geom)

    def close(self):
        # Do nothing.
        pass

    def render(self):
        self._clear()
        # Render the arena
        self.geoms[0].render(self.screen, self.stack)
        # Render the grid
        self.draw_grid()
        self.stack.push(self.transform)
        # Render the rest of the entities
        for geom in self.geoms[1:]:
            geom.render(self.screen, self.stack)
        self.stack.pop()
        # array3d returns an array that is indexed by the x-axis first,
        # followed by the y-axis. Thus, we swap the axes before returning the
        # observation.
        obs = pygame.surfarray.array3d(self.screen)
        return obs.swapaxes(0, 1)
