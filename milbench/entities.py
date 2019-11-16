"""Some kind of 'entity' abstraction for game world objects."""

import abc
import enum
import weakref

import milbench.gym_render as r
import pymunk as pm


class Entity(abc.ABC):
    """Basic class for logical 'things' that can be displayed on screen and/or
    interact via physics."""
    def setup(self, viewer, space):
        """Set up entity graphics/physics usig a gym_render.Viewer and a
        pm.Space. Only gets called once."""
        self.viewer = weakref.proxy(viewer)
        self.space = weakref.proxy(space)

    def set_action(self):
        """Set a persistent action to be used during .update(). Not all objects
        will have actions."""
    def update(self, dt):
        """Do an logic/physics update at some (most likely fixed) time
        interval."""
    def pre_draw(self):
        """Do a graphics state update to, e.g., update state of internal
        `Geom`s. This doesn't have to be done at every physics time step."""


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
        # computed from current action
        self.left_force = None
        self.right_force = None

    def setup(self, *args, **kwargs):
        super().setup(*args, **kwargs)

        # physics setup
        # signature: moment_for_circle(mass, inner_rad, outer_rad, offset)
        inertia = pm.moment_for_circle(self.mass, 0, self.radius, (0, 0))
        self.robot_body = body = pm.Body(self.mass, inertia)
        body.position = self.init_pos
        body.angle = self.init_angle
        # for control
        # TODO: rewrite this to follow the tank example.
        control_body = pm.Body(2.0 * self.mass, 3 * inertia)
        control_body.position = self.init_pos
        control_body.angle = self.init_angle
        control_joint = pm.PivotJoint(body, control_body, (0, 0), (0, 0))
        control_joint.max_force = 10
        control_joint.bias_coef = 0
        trans_joint = pm.DampedRotarySpring(body, control_body, 0.0, 10.0,
                                            0.99)
        # collision setup
        # signature: Circle(body, radius, offset)
        body_shape = pm.Circle(body, self.radius, (0, 0))
        body_shape.friction = 0.5
        self.space.add(body, body_shape, control_joint, trans_joint)

        # graphics setup
        # draw a circular body
        circ_body = r.make_circle(radius=self.radius, res=100)
        circ_body.set_color(0.5, 0.5, 0.5)
        # draw some cute eyes
        eye_shapes = []
        for x_sign in [-1, 1]:
            eye = r.make_circle(radius=0.15 * self.radius, res=20)
            eye.set_color(1.0, 1.0, 1.0)
            eye.add_attr(r.Transform().set_translation(
                x_sign * 0.4 * self.radius, 0.3 * self.radius))
            pupil = r.make_circle(radius=0.05 * self.radius, res=10)
            pupil.set_color(0.1, 0.1, 0.1)
            pupil.add_attr(r.Transform().set_translation(
                x_sign * self.radius * 0.4, self.radius * 0.3))
            eye_shapes.extend([eye, pupil])
        # join them together
        self.robot_xform = r.Transform()
        robot_compound = r.Compound([circ_body, *eye_shapes])
        robot_compound.add_attr(self.robot_xform)
        self.viewer.add_geom(robot_compound)

    def set_action(self, action):
        force_unit = 2.0
        self.left_force = 0.0
        self.right_force = 0.0
        if action & RobotAction.UP:
            self.left_force += force_unit
            self.right_force += force_unit
        if action & RobotAction.DOWN:
            self.left_force -= force_unit
            self.right_force -= force_unit
        if action & RobotAction.LEFT:
            self.left_force += force_unit / 2
            self.right_force -= force_unit / 2
        if action & RobotAction.RIGHT:
            self.left_force -= force_unit / 2
            self.right_force += force_unit / 2

    def update(self, dt):
        if self.right_force:
            self.robot_body.apply_force_at_local_point((0, self.right_force),
                                                       (-self.radius / 2, 0))
        if self.left_force:
            self.robot_body.apply_force_at_local_point((0, self.left_force),
                                                       (self.radius / 2, 0))

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


# class Shape(Entity):
#     """A shape that can be pushed around."""
#     def __init__(self,
#                  shape_type,
#                  shape_size,
#                  colour,
#                  init_pos,
#                  init_rotation,
#                  mass=0.5):
#         self.shape_type = shape_type
#         self.shape_size = shape_size
#         self.colour = colour
#         self.com_position = init_pos
#         self.rotation = init_rotation

#     def setup(self, *args, **kwargs):
#         super().setup(*args, **kwargs)

#     def update(self, dt):
#         pass

#     def pre_draw(self):
#         pass
