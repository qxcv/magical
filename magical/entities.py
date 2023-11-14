"""Some kind of 'entity' abstraction for game world objects. The robot, the
different shapes in the environment, and the non-interacting coloured 'goal
regions' are all examples of entities."""

import abc
import enum
import math
import weakref

import numpy as np
import pymunk as pm
import pymunk.autogeometry as autogeom

import magical.geom as gtools
import magical.render as r
from magical.style import (
    COLOURS_RGB, GOAL_LINE_THICKNESS, ROBOT_LINE_THICKNESS,
    SHAPE_LINE_THICKNESS, darken_rgb, lighten_rgb)

# #############################################################################
# Entity ABC
# #############################################################################


class Entity(abc.ABC):
    """Basic class for logical 'things' that can be displayed on screen and/or
    interact via physics."""
    @abc.abstractmethod
    def setup(self, viewer, space, phys_vars, label = None):
        """Set up entity graphics/physics usig a gym_render.Viewer and a
        pm.Space. Only gets called once."""
        self.shapes = []
        self.bodies = []
        self.viewer = weakref.proxy(viewer)
        self.space = weakref.proxy(space)
        self.phys_vars = weakref.proxy(phys_vars)
        self.label = label

    def update(self, dt):
        """Do an logic/physics update at some (most likely fixed) time
        interval."""

    def pre_draw(self):
        """Do a graphics state update to, e.g., update state of internal
        `Geom`s. This doesn't have to be done at every physics time step."""

    def reconstruct_signature(self):
        """Produce signature necessary to reconstruct this entity in its
        current pose. This is useful for creating new scenarios out of existing
        world states.

        Returns: tuple of (cls, kwargs), where:
            cls (type): the class that should be used to construct the
                instance.
            kwargs (dict): keyword arguments that should be passed to the
                constructor for cls."""
        raise NotImplementedError(
            f"no .reconstruct_signature() implementation for object "
            f"'{self}' of type '{type(self)}'")

    def generate_group_id(self):
        """Generate a new, unique group ID. Intended to be called from
        `.setup()` when creating `ShapeFilter`s."""
        if not hasattr(self.space, '_group_ctr'):
            self.space._group_ctr = 999
        self.space._group_ctr += 1
        return self.space._group_ctr

    @staticmethod
    def format_reconstruct_signature(cls, kwargs):
        """String-format a reconstruction signature. Makes things as
        easy to cut-and-paste as possible."""
        prefix = "    "
        kwargs_sig_parts = []
        for k, v in sorted(kwargs.items()):
            v_str = str(v)
            if isinstance(v, pm.vec2d.Vec2d):
                v_str = "(%.5g, %.5g)" % (v.x, v.y)
            elif isinstance(v, float):
                v_str = "%.5g" % v
            part = f"{prefix}{k}={v_str}"
            kwargs_sig_parts.append(part)
        kwargs_sig = ",\n".join(kwargs_sig_parts)
        result = f"{cls.__name__}(\n{kwargs_sig})"
        return result

    def add_to_space(self, *objects):
        """For adding a body or shape to the Pymunk 'space'. Keeps track of
        shapes/bodies so they can be used later. Should be called instead of
        space.add()."""
        for obj in objects:
            self.space.add(obj)
            if isinstance(obj, pm.Body):
                self.bodies.append(obj)
            elif isinstance(obj, pm.Shape):
                self.shapes.append(obj)
            elif isinstance(obj, pm.Constraint):
                pass
            else:
                raise TypeError(
                    f"don't know how to handle object '{obj}' of type "
                    f"'{type(obj)}' in class '{type(self)}'")


# #############################################################################
# Helpers
# #############################################################################


class EntityIndex:
    def __init__(self, entities):
        """Build a reverse index mapping shapes to entities. Assumes that all
        the shapes which make up an entity are stored as attributes on the
        shape, or are attached to bodies which are stored as attributes on the
        shape.

        Args:
            entities ([Entity]): list of entities to process.

        Returns:
            ent_index (dict): dictionary mapping shapes to entities."""
        self._shape_to_ent = dict()
        self._ent_to_shapes = dict()
        for entity in entities:
            shapes = entity.shapes
            self._ent_to_shapes[entity] = shapes
            for shape in shapes:
                assert shape not in self._shape_to_ent, \
                    f"shape {shape} appears in {entity} and " \
                    f"{self._shape_to_ent[shape]}"
                self._shape_to_ent[shape] = entity

    def entity_for(self, shape):
        """Look up the entity associated with a particular shape. Raises
        `KeyError` if no known entity owns the shape."""
        return self._shape_to_ent[shape]

    def shapes_for(self, ent):
        """Return a set of shapes associated with the given entity. Raises
        `KeyError` if the given entity is not in the index."""
        return self._ent_to_shapes[ent]


# #############################################################################
# Robot entity
# #############################################################################


class RobotAction(enum.IntFlag):
    NONE = 0
    UP = 1
    DOWN = 2
    LEFT = 4
    RIGHT = 8
    OPEN = 16
    CLOSE = 32


# We need to have unique, consecutive identifiers for each action so that we
# can use standard tools for categorical action spaces. This list & the
# structures that follow allow us to map robot action flag combinations to
# unique integer identifiers, and vice versa.
ACTION_NUMS_FLAGS_NAMES = (
    # (unique act ID, (up/down, left/right, open/close), act name)
    (0,  (RobotAction.NONE, RobotAction.NONE,  RobotAction.OPEN),  'Open'),  # noqa: E501
    (1,  (RobotAction.UP,   RobotAction.NONE,  RobotAction.OPEN),  'UpOpen'),  # noqa: E501
    (2,  (RobotAction.DOWN, RobotAction.NONE,  RobotAction.OPEN),  'DownOpen'),  # noqa: E501
    (3,  (RobotAction.NONE, RobotAction.LEFT,  RobotAction.OPEN),  'LeftOpen'),  # noqa: E501
    (4,  (RobotAction.UP,   RobotAction.LEFT,  RobotAction.OPEN),  'UpLeftOpen'),  # noqa: E501
    (5,  (RobotAction.DOWN, RobotAction.LEFT,  RobotAction.OPEN),  'DownLeftOpen'),  # noqa: E501
    (6,  (RobotAction.NONE, RobotAction.RIGHT, RobotAction.OPEN),  'RightOpen'),  # noqa: E501
    (7,  (RobotAction.UP,   RobotAction.RIGHT, RobotAction.OPEN),  'UpRightOpen'),  # noqa: E501
    (8,  (RobotAction.DOWN, RobotAction.RIGHT, RobotAction.OPEN),  'DownRightOpen'),  # noqa: E501
    (9,  (RobotAction.NONE, RobotAction.NONE,  RobotAction.CLOSE), 'Close'),  # noqa: E501
    (10, (RobotAction.UP,   RobotAction.NONE,  RobotAction.CLOSE), 'UpClose'),  # noqa: E501
    (11, (RobotAction.DOWN, RobotAction.NONE,  RobotAction.CLOSE), 'DownClose'),  # noqa: E501
    (12, (RobotAction.NONE, RobotAction.LEFT,  RobotAction.CLOSE), 'LeftClose'),  # noqa: E501
    (13, (RobotAction.UP,   RobotAction.LEFT,  RobotAction.CLOSE), 'UpLeftClose'),  # noqa: E501
    (14, (RobotAction.DOWN, RobotAction.LEFT,  RobotAction.CLOSE), 'DownLeftClose'),  # noqa: E501
    (15, (RobotAction.NONE, RobotAction.RIGHT, RobotAction.CLOSE), 'RightClose'),  # noqa: E501
    (16, (RobotAction.UP,   RobotAction.RIGHT, RobotAction.CLOSE), 'UpRightClose'),  # noqa: E501
    (17, (RobotAction.DOWN, RobotAction.RIGHT, RobotAction.CLOSE), 'DownRightClose'),  # noqa: E501
)  # yapf: disable
ACTION_ID_TO_FLAGS = {
    act_id: flags
    for act_id, flags, _ in ACTION_NUMS_FLAGS_NAMES
}
FLAGS_TO_ACTION_ID = {
    flags: act_id
    for act_id, flags, _ in ACTION_NUMS_FLAGS_NAMES
}


def make_finger_vertices(upper_arm_len, forearm_len, thickness, side_sign):
    """Make annoying finger polygons coordinates. Corresponding composite shape
    will have origin at root of upper arm, with upper arm oriented straight
    upwards and forearm above it."""
    up_shift = upper_arm_len / 2
    upper_arm_vertices = gtools.rect_verts(thickness, upper_arm_len)
    forearm_vertices = gtools.rect_verts(thickness, forearm_len)
    # now rotate upper arm into place & then move it to correct position
    upper_start = pm.vec2d.Vec2d(side_sign * thickness / 2, upper_arm_len / 2)
    forearm_offset_unrot = pm.vec2d.Vec2d(-side_sign * thickness / 2,
                                          forearm_len / 2)
    rot_angle = side_sign * math.pi / 8
    forearm_trans = upper_start + forearm_offset_unrot.rotated(rot_angle)
    forearm_trans.y += up_shift
    forearm_vertices_trans = [
        v.rotated(rot_angle) + forearm_trans for v in forearm_vertices
    ]
    for v in upper_arm_vertices:
        v.y += up_shift
    upper_arm_verts_final = [(v.x, v.y) for v in upper_arm_vertices]
    forearm_verts_final = [(v.x, v.y) for v in forearm_vertices_trans]
    return upper_arm_verts_final, forearm_verts_final


class Robot(Entity):
    """Robot body controlled by the agent."""
    def __init__(self, radius, init_pos, init_angle, mass=1.0):
        self.radius = radius
        self.init_pos = init_pos
        self.init_angle = init_angle
        self.mass = mass
        self.rel_turn_angle = 0.0
        self.target_speed = 0.0
        # max angle from vertical on inner side and outer
        self.finger_rot_limit_outer = math.pi / 8
        self.finger_rot_limit_inner = 0.0

    def reconstruct_signature(self):
        cls = type(self)
        kwargs = dict(radius=self.radius,
                      init_pos=self.robot_body.position,
                      init_angle=self.robot_body.angle,
                      mass=self.mass)
        return cls, kwargs

    def setup(self, *args, **kwargs):
        super().setup(*args, **kwargs)
        
        # physics setup, starting with main body
        # signature: moment_for_circle(mass, inner_rad, outer_rad, offset)
        inertia = pm.moment_for_circle(self.mass, 0, self.radius, (0, 0))
        self.robot_body = body = pm.Body(self.mass, inertia)
        body.position = self.init_pos
        body.angle = self.init_angle
        self.add_to_space(body)

        # For control. The rough joint setup was taken form tank.py in the
        # pymunk examples.
        self.control_body = control_body = pm.Body(body_type=pm.Body.KINEMATIC)
        control_body.position = self.init_pos
        control_body.angle = self.init_angle
        self.add_to_space(control_body)
        pos_control_joint = pm.PivotJoint(control_body, body, (0, 0), (0, 0))
        pos_control_joint.max_bias = 0
        pos_control_joint.max_force = self.phys_vars.robot_pos_joint_max_force
        self.add_to_space(pos_control_joint)
        rot_control_joint = pm.GearJoint(control_body, body, 0.0, 1.0)
        rot_control_joint.error_bias = 0.0
        rot_control_joint.max_bias = 2.5
        rot_control_joint.max_force = self.phys_vars.robot_rot_joint_max_force
        self.add_to_space(rot_control_joint)

        # googly eye control bodies & joints
        self.pupil_bodies = []
        for eye_side in [-1, 1]:
            eye_mass = self.mass / 10
            eye_radius = self.radius
            eye_inertia = pm.moment_for_circle(eye_mass, 0, eye_radius, (0, 0))
            eye_body = pm.Body(eye_mass, eye_inertia)
            eye_body.angle = self.init_angle
            eye_joint = pm.DampedRotarySpring(body, eye_body, 0, 0.1, 3e-3)
            eye_joint.max_bias = 3.0
            eye_joint.max_force = 0.001
            self.pupil_bodies.append(eye_body)
            self.add_to_space(eye_body, eye_joint)

        # finger bodies/controls (annoying)
        finger_thickness = 0.25 * self.radius
        finger_upper_length = 1.1 * self.radius
        finger_lower_length = 0.7 * self.radius
        self.finger_bodies = []
        self.finger_motors = []
        finger_vertices = []
        finger_inner_vertices = []
        self.target_finger_angle = 0.0
        for finger_side in [-1, 1]:
            # basic finger body
            finger_verts = make_finger_vertices(
                upper_arm_len=finger_upper_length,
                forearm_len=finger_lower_length,
                thickness=finger_thickness,
                side_sign=finger_side)
            finger_vertices.append(finger_verts)
            finger_inner_verts = make_finger_vertices(
                upper_arm_len=finger_upper_length - ROBOT_LINE_THICKNESS * 2,
                forearm_len=finger_lower_length - ROBOT_LINE_THICKNESS * 2,
                thickness=finger_thickness - ROBOT_LINE_THICKNESS * 2,
                side_sign=finger_side)
            finger_inner_verts = [[(x, y + ROBOT_LINE_THICKNESS)
                                   for x, y in box]
                                  for box in finger_inner_verts]
            finger_inner_vertices.append(finger_inner_verts)
            # these are movement limits; they are useful below, but also
            # necessary to make initial positioning work
            if finger_side < 0:
                lower_rot_lim = -self.finger_rot_limit_inner
                upper_rot_lim = self.finger_rot_limit_outer
            if finger_side > 0:
                lower_rot_lim = -self.finger_rot_limit_outer
                upper_rot_lim = self.finger_rot_limit_inner
            finger_mass = self.mass / 8
            finger_inertia = pm.moment_for_poly(finger_mass,
                                                sum(finger_verts, []))
            finger_body = pm.Body(finger_mass, finger_inertia)
            if finger_side < 0:
                delta_finger_angle = upper_rot_lim
                finger_body.angle = self.init_angle + delta_finger_angle
            else:
                delta_finger_angle = lower_rot_lim
                finger_body.angle = self.init_angle + delta_finger_angle
            # position of finger relative to body
            finger_rel_pos = (finger_side * self.radius * 0.45,
                              self.radius * 0.1)
            finger_rel_pos_rot = gtools.rotate_vec(finger_rel_pos,
                                                   self.init_angle)
            finger_body.position = gtools.add_vecs(body.position,
                                                   finger_rel_pos_rot)
            self.add_to_space(finger_body)
            self.finger_bodies.append(finger_body)

            # pivot joint to keep it in place (it will rotate around this
            # point)
            finger_piv = pm.PivotJoint(body, finger_body, finger_body.position)
            finger_piv.error_bias = 0.0
            self.add_to_space(finger_piv)

            # rotary limit joint to stop it from getting too far out of line
            finger_limit = pm.RotaryLimitJoint(body, finger_body,
                                               lower_rot_lim, upper_rot_lim)
            finger_limit.error_bias = 0.0
            self.add_to_space(finger_limit)
            # motor to move the fingers around (very limited in power so as not
            # to conflict with rotary limit joint)
            finger_motor = pm.SimpleMotor(body, finger_body, 0.0)
            finger_motor.rate = 0.0
            finger_motor.max_bias = 0.0
            finger_motor.max_force = self.phys_vars.robot_finger_max_force
            self.add_to_space(finger_motor)
            self.finger_motors.append(finger_motor)

        # For collision. Main body circle. Signature: Circle(body, radius,
        # offset).
        robot_group = 1
        body_shape = pm.Circle(body, self.radius, (0, 0))
        body_shape.filter = pm.ShapeFilter(group=robot_group)
        body_shape.friction = 0.5
        self.add_to_space(body_shape)
        # the fingers
        finger_shapes = []
        for finger_body, finger_verts, finger_side in zip(
                self.finger_bodies, finger_vertices, [-1, 1]):
            finger_subshapes = []
            for finger_subverts in finger_verts:
                finger_subshape = pm.Poly(finger_body, finger_subverts)
                finger_subshape.filter = pm.ShapeFilter(group=robot_group)
                # grippy fingers
                finger_subshape.friction = 5.0
                finger_subshapes.append(finger_subshape)
            self.add_to_space(*finger_subshapes)
            finger_shapes.append(finger_subshapes)

        # ======================================== #
        # Graphics setup
        # ======================================== #

        # Main robot body.
        circ_body = r.make_circle(self.radius, 100, True)
        robot_colour = COLOURS_RGB['grey']
        dark_robot_colour = darken_rgb(robot_colour)
        light_robot_colour = lighten_rgb(robot_colour, 4)
        circ_body.color = robot_colour
        circ_body.outline_color = dark_robot_colour

        # Fingers.
        self.finger_xforms = []
        finger_outer_geoms = []
        finger_inner_geoms = []
        for finger_outer_subshapes, finger_inner_verts, finger_side in zip(
                finger_shapes, finger_inner_vertices, [-1, 1]):
            finger_xform = r.Transform()
            self.finger_xforms.append(finger_xform)
            for finger_subshape in finger_outer_subshapes:
                vertices = [(v.x, v.y) for v in finger_subshape.get_vertices()]
                finger_outer_geom = r.Poly(vertices, False)
                finger_outer_geom.color = robot_colour
                finger_outer_geom.add_transform(finger_xform)
                finger_outer_geoms.append(finger_outer_geom)
            for vertices in finger_inner_verts:
                finger_inner_geom = r.Poly(vertices, False)
                finger_inner_geom.color = light_robot_colour
                finger_inner_geom.add_transform(finger_xform)
                finger_inner_geoms.append(finger_inner_geom)
        for geom in finger_outer_geoms:
            self.viewer.add_geom(geom)
        for geom in finger_inner_geoms:
            self.viewer.add_geom(geom)

        # Eyes.
        eye_shapes = []
        self.pupil_transforms = []
        for x_sign in [-1, 1]:
            eye = r.make_circle(0.2 * self.radius, 100, outline=False)
            eye.color = (1.0, 1.0, 1.0)  # white color
            eye_base_transform = r.Transform(
                translation=(x_sign * 0.4 * self.radius, 0.3 * self.radius),
            )
            eye.add_transform(eye_base_transform)
            pupil = r.make_circle(0.12 * self.radius, 100, outline=False)
            pupil.color = (0.1, 0.1, 0.1)
            pupil_transform = r.Transform()
            pupil.add_transform(
                r.Transform(translation=(0, self.radius * 0.07)))
            pupil.add_transform(pupil_transform)
            pupil.add_transform(eye_base_transform)
            self.pupil_transforms.append(pupil_transform)
            eye_shapes.extend([eye, pupil])

        self.robot_xform = r.Transform()
        robot_compound = r.Compound([circ_body, *eye_shapes])
        robot_compound.add_transform(self.robot_xform)
        self.viewer.add_geom(robot_compound)

    def set_action(self, move_action):
        self.rel_turn_angle = 0.0
        self.target_speed = 0.0
        if move_action & RobotAction.UP:
            self.target_speed += 4.0 * self.radius
        if move_action & RobotAction.DOWN:
            self.target_speed -= 3.0 * self.radius
        if (move_action & RobotAction.UP) and (move_action & RobotAction.DOWN):
            self.target_speed = 0.0
        if move_action & RobotAction.LEFT:
            self.rel_turn_angle += 1.5
        if move_action & RobotAction.RIGHT:
            self.rel_turn_angle -= 1.5
        if (move_action & RobotAction.OPEN):
            # setting target for LEFT finger
            # (for right finger you'll need to flip across main axis)
            self.target_finger_angle = self.finger_rot_limit_outer
        elif move_action & RobotAction.CLOSE:
            self.target_finger_angle = -self.finger_rot_limit_inner

    def update(self, dt):
        # target heading
        self.control_body.angle = self.robot_body.angle + self.rel_turn_angle

        # target speed
        x_vel_vector = pm.vec2d.Vec2d(0.0, self.target_speed)
        vel_vector = self.robot_body.rotation_vector.cpvrotate(x_vel_vector)
        self.control_body.velocity = vel_vector

        ## close fingers
        # target finger positions, relative to body
        for finger_body, finger_motor, finger_side in zip(
                self.finger_bodies, self.finger_motors, [-1, 1]):
            rel_angle = finger_body.angle - self.robot_body.angle
            # for the left finger, the target angle is measured
            # counterclockwise; for the right, it's measured clockwise
            # (chipmunk is always counterclockwise)
            angle_error = rel_angle + finger_side * self.target_finger_angle
            target_rate = max(-1, min(1, angle_error * 10))
            if abs(target_rate) < 1e-4:
                target_rate = 0.0
            finger_motor.rate = target_rate

    def pre_draw(self):
        self.robot_xform.reset(
            translation=self.robot_body.position,
            rotation=self.robot_body.angle)

        for finger_xform, finger_body in zip(self.finger_xforms,
                                             self.finger_bodies):
            finger_xform.reset(
                translation=finger_body.position,
                rotation=finger_body.angle)

        for pupil_xform, pupil_body in zip(
            self.pupil_transforms, self.pupil_bodies):
            pupil_xform.reset(rotation=pupil_body.angle - self.robot_body.angle)

# #############################################################################
# Arena boundary
# #############################################################################


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
        self.add_to_space(*arena_segments)

        width = self.right - self.left
        height = self.top - self.bottom
        arena_square = r.make_rect(width, height, True)
        arena_square.color = (1, 1, 1)
        arena_square.outline_color = COLOURS_RGB['grey']
        txty = (self.left + width / 2, self.bottom + height / 2)
        centre_xform = r.Transform(translation=txty)
        arena_square.add_transform(centre_xform)

        self.viewer.add_geom(arena_square)


# #############################################################################
# Pushable shapes
# #############################################################################


class ShapeType(str, enum.Enum):
    TRIANGLE = 'triangle'
    SQUARE = 'square'
    PENTAGON = 'pentagon'
    # hexagon is somewhat hard to distinguish from pentagon, and octagon is
    # very hard to distinguish from circle at low resolutions
    HEXAGON = 'hexagon'
    OCTAGON = 'octagon'
    CIRCLE = 'circle'
    STAR = 'star'


class ShapeColour(str, enum.Enum):
    RED = 'red'
    GREEN = 'green'
    BLUE = 'blue'
    YELLOW = 'yellow'


# limited set of types and colours to use for random generation
# (WARNING: not all benchmarks use the two arrays below! Some have used their
# own arrays so that changes to the base SHAPE_TYPES array don't break the
# benchmark's default shape layout.)
SHAPE_TYPES = np.asarray([
    ShapeType.SQUARE,
    ShapeType.PENTAGON,
    ShapeType.STAR,
    ShapeType.CIRCLE,
],
                         dtype='object')

# update the shape type for easier shapes
SHAPE_TYPES = np.asarray([
    ShapeType.SQUARE,
    ShapeType.TRIANGLE,
    ShapeType.CIRCLE,
],
                         dtype='object')

SHAPE_COLOURS = np.asarray([
    ShapeColour.RED,
    ShapeColour.GREEN,
    ShapeColour.BLUE,
    ShapeColour.YELLOW,
],
                           dtype='object')


class Shape(Entity):
    """A shape that can be pushed around."""
    def __init__(self,
                 shape_type,
                 colour_name,
                 shape_size,
                 init_pos,
                 init_angle,
                 mass=0.5,
                 easy_visuals=False):
        self.shape_type = shape_type
        # this "size" can be interpreted in different ways depending on the
        # shape type, but area of shape should increase quadratically in this
        # number regardless of shape type
        self.shape_size = shape_size
        self.colour_name = colour_name
        self.colour = COLOURS_RGB[self.colour_name]
        self.init_pos = init_pos
        self.init_angle = init_angle
        self.mass = mass
        self.easy_visuals = easy_visuals

    def reconstruct_signature(self):
        cls = type(self)
        kwargs = dict(shape_type=self.shape_type,
                      colour_name=self.colour_name,
                      shape_size=self.shape_size,
                      init_pos=self.shape_body.position,
                      init_angle=self.shape_body.angle,
                      mass=self.mass)
        return cls, kwargs

    def setup(self, *args, **kwargs):
        super().setup(*args, **kwargs)

        # Physics. This joint setup was taken form tank.py in the pymunk
        # examples.

        if self.shape_type == ShapeType.SQUARE:
            self.shape_body = body = pm.Body()
            body.position = self.init_pos
            body.angle = self.init_angle
            self.add_to_space(body)

            side_len = math.sqrt(math.pi) * self.shape_size
            shape = pm.Poly.create_box(
                body,
                (side_len, side_len),
                # slightly bevelled corners
                0.01 * side_len)
            # FIXME: why is this necessary? Do I need it for the others?
            shape.mass = self.mass
            shapes = [shape]
            del shape
        elif self.shape_type == ShapeType.CIRCLE:
            inertia = pm.moment_for_circle(self.mass, 0, self.shape_size,
                                           (0, 0))
            self.shape_body = body = pm.Body(self.mass, inertia)
            body.position = self.init_pos
            body.angle = self.init_angle
            self.add_to_space(body)
            shape = pm.Circle(body, self.shape_size, (0, 0))
            shapes = [shape]
            self.shape = shape
            # del shape
        elif self.shape_type == ShapeType.STAR:
            star_npoints = 5
            star_out_rad = 1.3 * self.shape_size
            star_in_rad = 0.5 * star_out_rad
            star_verts = gtools.compute_star_verts(star_npoints, star_out_rad,
                                                   star_in_rad)
            # create an exact convex decpomosition
            convex_parts = autogeom.convex_decomposition(
                star_verts + star_verts[:1], 0)
            star_hull = autogeom.to_convex_hull(star_verts, 1e-5)
            star_inertia = pm.moment_for_poly(self.mass, star_hull, (0, 0), 0)
            self.shape_body = body = pm.Body(self.mass, star_inertia)
            body.position = self.init_pos
            body.angle = self.init_angle
            self.add_to_space(body)
            shapes = []
            star_group = self.generate_group_id()
            for convex_part in convex_parts:
                shape = pm.Poly(body, convex_part)
                # avoid self-intersection with a shape filter
                shape.filter = pm.ShapeFilter(group=star_group)
                shapes.append(shape)
                del shape
        else:
            # these are free-form shapes b/c no helpers exist in Pymunk
            if self.shape_type == ShapeType.TRIANGLE:
                # shrink to make it look more sensible and easier to grasp
                factor = 0.8
                num_sides = 3
            elif self.shape_type == ShapeType.PENTAGON:
                factor = 1.0
                num_sides = 5
            elif self.shape_type == ShapeType.HEXAGON:
                factor = 1.0
                num_sides = 6
            elif self.shape_type == ShapeType.OCTAGON:
                factor = 1.0
                num_sides = 8
            else:
                raise NotImplementedError("haven't implemented",
                                          self.shape_type)
            side_len = factor * gtools.regular_poly_circ_rad_to_side_length(
                num_sides, self.shape_size)
            poly_verts = gtools.compute_regular_poly_verts(num_sides, side_len)
            inertia = pm.moment_for_poly(self.mass, poly_verts, (0, 0), 0)
            self.shape_body = body = pm.Body(self.mass, inertia)
            body.position = self.init_pos
            body.angle = self.init_angle
            self.add_to_space(body)
            shape = pm.Poly(body, poly_verts)
            shapes = [shape]
            del shape

        for shape in shapes:
            shape.friction = 0.5
            self.add_to_space(shape)

        trans_joint = pm.PivotJoint(self.space.static_body, body, (0, 0),
                                    (0, 0))
        trans_joint.max_bias = 0
        trans_joint.max_force = self.phys_vars.shape_trans_joint_max_force
        self.add_to_space(trans_joint)
        rot_joint = pm.GearJoint(self.space.static_body, body, 0.0, 1.0)
        rot_joint.max_bias = 0
        rot_joint.max_force = self.phys_vars.shape_rot_joint_max_force
        self.add_to_space(rot_joint)

        # Drawing
        geoms_outer = []
        if self.shape_type == ShapeType.SQUARE:
            if self.easy_visuals:
               geoms = [r.make_square(side_len, outline=True, label=self.label)]
            else:
                geoms = [r.make_square(side_len, outline=True)]
        elif self.shape_type == ShapeType.CIRCLE:
            if self.easy_visuals:
                geoms = [r.make_circle(self.shape_size, 100, True, label=self.label)]
            else:
                geoms = [r.make_circle(self.shape_size, 100, True)]
        elif self.shape_type == ShapeType.STAR:
            star_short_verts = gtools.compute_star_verts(
                star_npoints, star_out_rad - SHAPE_LINE_THICKNESS,
                star_in_rad - SHAPE_LINE_THICKNESS)
            short_convex_parts = autogeom.convex_decomposition(
                star_short_verts + star_short_verts[:1], 0)
            geoms = []
            coord = np.array([0.0, 0.0])
            for part in short_convex_parts:
                coord += np.mean(np.array(part), axis=0)
                geoms.append(r.Poly(part, outline=False))
            coord = coord / 6
            # create the label of star
            if self.easy_visuals:
                geoms.append(r.Poly(coord, outline=False, label=self.label))
            geoms_outer = []
            for part in convex_parts:
                geoms_outer.append(r.Poly(part, outline=False))
        elif self.shape_type == ShapeType.OCTAGON \
                or self.shape_type == ShapeType.HEXAGON \
                or self.shape_type == ShapeType.PENTAGON \
                or self.shape_type == ShapeType.TRIANGLE:
                if self.easy_visuals:
                    geoms = [r.Poly(poly_verts, outline=True, label=self.label)]
                else:
                    geoms = [r.Poly(poly_verts, outline=True)]
        else:
            raise NotImplementedError("haven't implemented", self.shape_type)

        if self.shape_type == ShapeType.STAR:
            for g in geoms_outer:
                if self.easy_visuals:
                    # set a black boundary line around the shape
                    g.color = (0, 0, 0)
                g.color = darken_rgb(self.colour)
            for g in geoms:
                g.color = self.colour
        else:
            for g in geoms:
                g.color = self.colour
                if self.easy_visuals:
                     # set a black boundary line around the shape
                    g.outline_color = (0, 0, 0)
                g.outline_color = darken_rgb(self.colour)

        self.shape_xform = r.Transform()
        shape_compound = r.Compound(geoms_outer + geoms)
        shape_compound.add_transform(self.shape_xform)
        self.viewer.add_geom(shape_compound)

    def pre_draw(self):
        self.shape_xform.reset(
            translation=self.shape_body.position,
            rotation=self.shape_body.angle)


# #############################################################################
# Sensor region for pushing shapes into.
# #############################################################################


class GoalRegion(Entity):
    """A goal region that the robot should push certain shapes into. It's up to
    the caller to figure out exactly which shapes & call methods for collision
    checking/scoring."""
    def __init__(self, x, y, h, w, colour_name):
        self.x = x
        self.y = y
        assert h > 0, w > 0
        self.h = h
        self.w = w
        self.colour_name = colour_name
        self.base_colour = COLOURS_RGB[colour_name]

    def reconstruct_signature(self):
        kwargs = dict(x=self.goal_body.position[0] - self.w / 2,
                      y=self.goal_body.position[1] + self.h / 2,
                      h=self.h,
                      w=self.w,
                      colour_name=self.colour_name)
        return type(self), kwargs

    def setup(self, *args, **kwargs):
        super().setup(*args, **kwargs)

        # the space only needs a sensor body
        self.goal_body = pm.Body(body_type=pm.Body.STATIC)
        self.goal_shape = pm.Poly.create_box(self.goal_body, (self.w, self.h))
        self.goal_shape.sensor = True
        self.goal_body.position = (self.x + self.w / 2, self.y - self.h / 2)
        self.add_to_space(self.goal_body, self.goal_shape)

        # Graphics.
        outer_colour = self.base_colour
        inner_colour = lighten_rgb(self.base_colour, times=2)
        inner_rect = r.make_rect(self.w, self.h, True, dashed=True)
        inner_rect.color = inner_colour
        inner_rect.outline_color = outer_colour
        self.goal_xform = r.Transform()
        inner_rect.add_transform(self.goal_xform)
        self.viewer.add_geom(inner_rect)

    def get_overlapping_ents(self,
                             ent_index,
                             contained=False,
                             com_overlap=False):
        """Get all entities overlapping this region.

        Args:
            ent_index (EntityIndex): index of entities to query over.
            contained (bool): set this to True to only return entities that are
                fully contained in the regions. Otherwise, if this is False,
                all entities that overlap the region at all will be returned.

        Returns:
            ents ([Entity]): list of entities intersecting the current one."""

        # first look up all overlapping shapes
        shape_results = self.space.shape_query(self.goal_shape)
        overlap_shapes = {r.shape for r in shape_results}

        # if necessary, do total containment check on shapes
        if contained:
            # This does a containment check based *only* on axis-aligned
            # bounding boxes. This is valid if our goal region is an
            # axis-aligned bounding box, but could lead to false positives if
            # the goal region were a different shape, or if it was rotated.
            goal_bb = self.goal_shape.bb
            overlap_shapes = {
                s
                for s in overlap_shapes if goal_bb.contains(s.bb)
            }
        if com_overlap:
            goal_bb = self.goal_shape.bb
            overlap_shapes = {
                s
                for s in overlap_shapes
                if goal_bb.contains_vect(s.body.position)
            }

        # now look up all indexed entities that own at least one overlapping
        # shape
        relevant_ents = set()
        for shape in overlap_shapes:
            try:
                ent = ent_index.entity_for(shape)
            except KeyError:
                # shape not in index
                continue
            relevant_ents.add(ent)

        # if necessary, filter the entities so that only those with *all*
        # shapes within the region (or with COMs of all bodies in the region)
        # are included
        if contained or com_overlap:
            new_relevant_ents = set()
            for relevant_ent in relevant_ents:
                shapes = set(ent_index.shapes_for(relevant_ent))
                if shapes <= overlap_shapes:
                    new_relevant_ents.add(relevant_ent)
            relevant_ents = new_relevant_ents

        return relevant_ents

    def pre_draw(self):
        self.goal_xform.reset(
            translation=self.goal_body.position,
            rotation=self.goal_body.angle)
