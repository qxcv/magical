"""Some kind of 'entity' abstraction for game world objects."""

import abc
import enum
from enum import auto
import math
import weakref

import milbench.gym_render as r
from milbench.style import LINE_THICKNESS, COLOURS_RGB, darken_rgb
import milbench.geom as gtools
import pymunk as pm

# #############################################################################
# Entity ABC
# #############################################################################


class Entity(abc.ABC):
    """Basic class for logical 'things' that can be displayed on screen and/or
    interact via physics."""
    def setup(self, viewer, space):
        """Set up entity graphics/physics usig a gym_render.Viewer and a
        pm.Space. Only gets called once."""
        self.viewer = weakref.proxy(viewer)
        self.space = weakref.proxy(space)

    def update(self, dt):
        """Do an logic/physics update at some (most likely fixed) time
        interval."""
    def pre_draw(self):
        """Do a graphics state update to, e.g., update state of internal
        `Geom`s. This doesn't have to be done at every physics time step."""


# #############################################################################
# Robot entity
# #############################################################################


class RobotAction(enum.IntFlag):
    NONE = 0
    UP = 1
    DOWN = 2
    LEFT = 4
    RIGHT = 8


class Robot(Entity):
    """Robot body controlled by the agent."""
    def __init__(self, radius, init_pos, init_angle, mass=1.0):
        self.radius = radius
        self.init_pos = init_pos
        self.init_angle = init_angle
        self.mass = mass
        self.rel_turn_angle = 0.0
        self.target_speed = 0.0

    def setup(self, *args, **kwargs):
        super().setup(*args, **kwargs)

        # physics setup, starting with main body
        # signature: moment_for_circle(mass, inner_rad, outer_rad, offset)
        inertia = pm.moment_for_circle(self.mass, 0, self.radius, (0, 0))
        self.robot_body = body = pm.Body(self.mass, inertia)
        body.position = self.init_pos
        body.angle = self.init_angle
        self.space.add(body)

        # For control. The rough joint setup was taken form tank.py in the
        # pymunk examples.
        self.control_body = control_body = pm.Body(body_type=pm.Body.KINEMATIC)
        control_body.position = self.init_pos
        control_body.angle = self.init_angle
        self.space.add(control_body)
        pos_control_joint = pm.PivotJoint(control_body, body, (0, 0), (0, 0))
        pos_control_joint.max_bias = 0
        pos_control_joint.max_force = 3
        self.space.add(pos_control_joint)
        rot_control_joint = pm.GearJoint(control_body, body, 0.0, 1.0)
        rot_control_joint.error_bias = 0.0
        rot_control_joint.max_bias = 2.5
        rot_control_joint.max_force = 1
        self.space.add(rot_control_joint)

        # for collision
        # signature: Circle(body, radius, offset)
        body_shape = pm.Circle(body, self.radius, (0, 0))
        body_shape.friction = 0.5
        self.space.add(body_shape)

        # graphics setup
        # draw a circular body
        circ_body_in = r.make_circle(radius=self.radius - LINE_THICKNESS,
                                     res=100)
        circ_body_out = r.make_circle(radius=self.radius, res=100)
        circ_body_in.set_color(*COLOURS_RGB['grey'])
        circ_body_out.set_color(*darken_rgb(COLOURS_RGB['grey']))
        # draw some cute eyes
        eye_shapes = []
        for x_sign in [-1, 1]:
            eye = r.make_circle(radius=0.2 * self.radius, res=20)
            eye.set_color(1.0, 1.0, 1.0)
            eye.add_attr(r.Transform().set_translation(
                x_sign * 0.4 * self.radius, 0.3 * self.radius))
            pupil = r.make_circle(radius=0.1 * self.radius, res=10)
            pupil.set_color(0.1, 0.1, 0.1)
            pupil.add_attr(r.Transform().set_translation(
                x_sign * self.radius * 0.4 + x_sign * self.radius * 0.03,
                self.radius * 0.3 + x_sign * self.radius * 0.03))
            eye_shapes.extend([eye, pupil])
        # join them together
        self.robot_xform = r.Transform()
        robot_compound = r.Compound([circ_body_out, circ_body_in, *eye_shapes])
        robot_compound.add_attr(self.robot_xform)
        self.viewer.add_geom(robot_compound)

    def set_action(self, action):
        self.rel_turn_angle = 0.0
        self.target_speed = 0.0
        if action & RobotAction.UP:
            self.target_speed += 4.0 * self.radius
        if action & RobotAction.DOWN:
            self.target_speed -= 3.0 * self.radius
        if (action & RobotAction.UP) and (action & RobotAction.DOWN):
            self.target_speed = 0.0
        if action & RobotAction.LEFT:
            self.rel_turn_angle += 1.5
        if action & RobotAction.RIGHT:
            self.rel_turn_angle -= 1.5

    def update(self, dt):
        # target heading
        self.control_body.angle = self.robot_body.angle + self.rel_turn_angle

        # target speed
        x_vel_vector = pm.vec2d.Vec2d(0.0, self.target_speed)
        vel_vector = self.robot_body.rotation_vector.cpvrotate(x_vel_vector)
        self.control_body.velocity = vel_vector

    def pre_draw(self):
        self.robot_xform.set_translation(*self.robot_body.position)
        self.robot_xform.set_rotation(self.robot_body.angle)


class ArenaBoundaries(Entity):
    """Handles physics of arena boundaries to keep everything in one place."""
    def __init__(self, left, right, top, bottom, seg_rad=1):
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
        self.seg_rad = seg_rad

    def setup(self, *args, **kwargs):
        super().setup(*args, **kwargs)

        # thick line segments around the edges
        arena_body = pm.Body(body_type=pm.Body.STATIC)
        rad = self.seg_rad
        points = [(self.left - rad, self.top + rad),
                  (self.right + rad, self.top + rad),
                  (self.right + rad, self.bottom - rad),
                  (self.left - rad, self.bottom - rad)]
        arena_segments = []
        for start_point, end_point in zip(points, points[1:] + points[:1]):
            segment = pm.Segment(arena_body, start_point, end_point, rad)
            segment.friction = 0.8
            arena_segments.append(segment)
        self.space.add(*arena_segments)


# #############################################################################
# Pushable shapes
# #############################################################################


class ShapeType(enum.Enum):
    CIRCLE = auto()
    SQUARE = auto()
    TRIANGLE = auto()
    PENTAGON = auto()


class Shape(Entity):
    """A shape that can be pushed around."""
    def __init__(self,
                 shape_type,
                 colour_name,
                 shape_size,
                 init_pos,
                 init_angle,
                 mass=0.5):
        self.shape_type = shape_type
        # this "size" can be interpreted in different ways depending on the
        # shape type, but area of shape should increase quadratically in this
        # number regardless of shape type
        self.shape_size = shape_size
        self.colour = COLOURS_RGB[colour_name]
        self.init_pos = init_pos
        self.init_angle = init_angle
        self.mass = mass

    def setup(self, *args, **kwargs):
        super().setup(*args, **kwargs)

        # Physics. This joint setup was taken form tank.py in the pymunk
        # examples.

        if self.shape_type == ShapeType.SQUARE:
            self.shape_body = body = pm.Body()
            body.position = self.init_pos
            body.angle = self.init_angle
            self.space.add(body)

            side_len = math.sqrt(math.pi) * self.shape_size
            shape = pm.Poly.create_box(
                body,
                (side_len, side_len),
                # slightly bevelled corners
                0.01 * side_len)
            # FIXME: why is this necessary? Do I need it for the others?
            shape.mass = self.mass
        elif self.shape_type == ShapeType.CIRCLE:
            inertia = pm.moment_for_circle(self.mass, 0, self.shape_size,
                                           (0, 0))
            self.shape_body = body = pm.Body(self.mass, inertia)
            body.position = self.init_pos
            body.angle = self.init_angle
            self.space.add(body)
            shape = pm.Circle(body, self.shape_size, (0, 0))
        elif self.shape_type == ShapeType.TRIANGLE \
                or self.shape_type == ShapeType.PENTAGON:
            # these are free-form shapes b/c no helpers exist in Pymunk
            if self.shape_type == ShapeType.TRIANGLE:
                factor = 0.85  # shrink to make it look more sensible
                num_sides = 3
            elif self.shape_type == ShapeType.PENTAGON:
                factor = 1.0
                num_sides = 5
            side_len = factor * gtools.regular_poly_circ_rad_to_side_length(
                num_sides, self.shape_size)
            poly_verts = gtools.compute_regular_poly_verts(num_sides, side_len)
            inertia = pm.moment_for_poly(self.mass, poly_verts, (0, 0), 0)
            self.shape_body = body = pm.Body(self.mass, inertia)
            body.position = self.init_pos
            body.angle = self.init_angle
            self.space.add(body)
            shape = pm.Poly(body, poly_verts)
        else:
            raise NotImplementedError("haven't implemented", self.shape_type)

        shape.friction = 0.5
        self.space.add(shape)

        trans_joint = pm.PivotJoint(self.space.static_body, body, (0, 0),
                                    (0, 0))
        trans_joint.max_bias = 0
        trans_joint.max_force = 1.5
        self.space.add(trans_joint)
        rot_joint = pm.GearJoint(self.space.static_body, body, 0.0, 1.0)
        rot_joint.max_bias = 0
        rot_joint.max_force = 1
        self.space.add(rot_joint)

        # Drawing
        if self.shape_type == ShapeType.SQUARE:
            geom_inner = r.make_square(side_len - 2 * LINE_THICKNESS)
            geom_outer = r.make_square(side_len)
        elif self.shape_type == ShapeType.CIRCLE:
            geom_inner = r.make_circle(radius=self.shape_size - LINE_THICKNESS,
                                       res=100)
            geom_outer = r.make_circle(radius=self.shape_size, res=100)
        elif self.shape_type == ShapeType.TRIANGLE \
                or self.shape_type == ShapeType.PENTAGON:
            apothem = gtools.regular_poly_side_length_to_apothem(
                num_sides, side_len)
            short_side_len = gtools.regular_poly_apothem_to_side_legnth(
                num_sides, apothem - LINE_THICKNESS)
            short_verts = gtools.compute_regular_poly_verts(
                num_sides, short_side_len)
            geom_inner = r.make_polygon(short_verts)
            geom_outer = r.make_polygon(poly_verts)
        else:
            raise NotImplementedError("haven't implemented", self.shape_type)

        geom_inner.set_color(*self.colour)
        geom_outer.set_color(*darken_rgb(self.colour))
        self.shape_xform = r.Transform()
        shape_compound = r.Compound([geom_outer, geom_inner])
        shape_compound.add_attr(self.shape_xform)
        self.viewer.add_geom(shape_compound)

    def pre_draw(self):
        self.shape_xform.set_translation(*self.shape_body.position)
        self.shape_xform.set_rotation(self.shape_body.angle)
