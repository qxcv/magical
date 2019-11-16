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

        # for control
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
                x_sign * self.radius * 0.4 + x_sign * self.radius * 0.03,
                self.radius * 0.3 + x_sign * self.radius * 0.03))
            eye_shapes.extend([eye, pupil])
        # join them together
        self.robot_xform = r.Transform()
        robot_compound = r.Compound([circ_body, *eye_shapes])
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


# Here's an example from the tank demo showing how to make boxes that have
# proper friction etc.:
#
# body = add_box(space, 20, 1)
# pivot = pm.PivotJoint(space.static_body, body, (0,0), (0,0))
# space.add(pivot)
# pivot.max_bias = 0
# pivot.max_Force = 1000
# gear = pm.GearJoint(space.static_body, body, 0.0, 1.0)
# space.add(gear)
# gear.max_bias = 0
# gear.max_force = 5000

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
