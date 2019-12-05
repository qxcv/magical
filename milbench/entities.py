"""Some kind of 'entity' abstraction for game world objects."""

import abc
import enum
import math
import weakref

import pymunk as pm

import milbench.geom as gtools
import milbench.gym_render as r
from milbench.style import COLOURS_RGB, LINE_THICKNESS, darken_rgb, lighten_rgb

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
        pass

    def pre_draw(self):
        """Do a graphics state update to, e.g., update state of internal
        `Geom`s. This doesn't have to be done at every physics time step."""
        pass


# #############################################################################
# Helpers
# #############################################################################


def _recursive_ent_shapes(container, level=0):
    assert level < 10
    rv = set()
    if isinstance(container, Entity):
        assert level == 0, f"found entity attached to entity ({container})"
        for attr_value in container.__dict__.values():
            rv.update(_recursive_ent_shapes(attr_value, level=level+1))
    elif isinstance(container, pm.Shape):
        rv.add(container)
    elif isinstance(container, pm.Body):
        for shape in container.shapes:
            rv.add(shape)
    elif isinstance(container, (list, dict, tuple, set)):
        for elem in container:
            rv.update(_recursive_ent_shapes(elem, level=level+1))
    return rv


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
            shapes = _recursive_ent_shapes(entity)
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


class LazyEntityIndex:
    """Version of EntityIndex that constructs itself only when actual methods
    are needed. This is a silly hack to get around the fact that I create this
    during the .on_reset() for an environment, but entities don't actually get
    set up with shapes, bodies, etc. until their .setup() method is called
    after .on_reset(). Probably there is a better refactoring that doesn't
    require this hack."""
    def __init__(self, *args, **kwargs):
        self._index = None
        self._args = args
        self._kwargs = kwargs

    def _init(self):
        if self._index is None:
            self._index = EntityIndex(*self._args, **self._kwargs)

    def entity_for(self, *args, **kwargs):
        self._init()
        return self._index.entity_for(*args, **kwargs)

    def shapes_for(self, *args, **kwargs):
        self._init()
        return self._index.shapes_for(*args, **kwargs)


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


def make_finger_vertices(upper_arm_len, forearm_len, thickness, side_sign):
    """Make annoying finger polygons coordinates. Corresponding composite shape
    will have origin in the middle of the upper arm, with upper arm oriented
    straight upwards and forearm above it."""
    upper_arm_vertices = gtools.rect_verts(thickness, upper_arm_len)
    forearm_vertices = gtools.rect_verts(thickness, forearm_len)
    # now rotate upper arm into place & then move it to correct position
    upper_start = pm.vec2d.Vec2d(side_sign * thickness / 2, upper_arm_len / 2)
    forearm_offset_unrot = pm.vec2d.Vec2d(-side_sign * thickness / 2,
                                          forearm_len / 2)
    rot_angle = side_sign * math.pi / 8
    forearm_trans = upper_start + forearm_offset_unrot.rotated(rot_angle)
    forearm_vertices_trans = [
        v.rotated(rot_angle) + forearm_trans for v in forearm_vertices
    ]
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
            self.space.add(eye_body, eye_joint)

        # finger bodies/controls (annoying)
        finger_thickness = 0.25 * self.radius
        finger_upper_length = 1.4 * self.radius
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
            # this is just for drawing
            finger_inner_verts = make_finger_vertices(
                upper_arm_len=finger_upper_length - LINE_THICKNESS * 2,
                forearm_len=finger_lower_length - LINE_THICKNESS * 2,
                thickness=finger_thickness - LINE_THICKNESS * 2,
                side_sign=finger_side)
            finger_inner_verts = [[(x, y + LINE_THICKNESS) for x, y in box]
                                  for box in finger_inner_verts]
            finger_inner_vertices.append(finger_inner_verts)
            # now create body
            finger_mass = self.mass / 8
            finger_inertia = pm.moment_for_poly(finger_mass,
                                                sum(finger_verts, []))
            finger_body = pm.Body(finger_mass, finger_inertia)
            finger_body.angle = self.init_angle
            # attach somewhere on the inner side of actual finger position
            finger_attach_delta = pm.vec2d.Vec2d(
                -finger_side * self.radius * 0.5, -self.radius * 0.8)
            # position of finger relative to body
            finger_rel_pos = (finger_side * self.radius * 0.35,
                              self.radius * 0.5)
            finger_rel_pos_rot = gtools.rotate_vec(finger_rel_pos,
                                                   self.init_angle)
            finger_body.position = gtools.add_vecs(body.position,
                                                   finger_rel_pos_rot)
            self.space.add(finger_body)
            self.finger_bodies.append(finger_body)

            # pin joint to keep it in place (it will rotate around this point)
            finger_pin = pm.PinJoint(
                body,
                finger_body,
                gtools.add_vecs(finger_rel_pos, finger_attach_delta),
                finger_attach_delta,
            )
            finger_pin.error_bias = 0.0
            self.space.add(finger_pin)
            # rotary limit joint to stop it from getting too far out of line
            if finger_side < 0:
                lower_rot_lim = -self.finger_rot_limit_inner
                upper_rot_lim = self.finger_rot_limit_outer
            if finger_side > 0:
                lower_rot_lim = -self.finger_rot_limit_outer
                upper_rot_lim = self.finger_rot_limit_inner
            finger_limit = pm.RotaryLimitJoint(body, finger_body,
                                               lower_rot_lim, upper_rot_lim)
            finger_limit.error_bias = 0.0
            self.space.add(finger_limit)
            # motor to move the fingers around (very limited in power so as not
            # to conflict with rotary limit joint)
            finger_motor = pm.SimpleMotor(body, finger_body, 0.0)
            finger_motor.rate = 0.0
            finger_motor.max_bias = 0.0
            finger_motor.max_force = 4
            self.space.add(finger_motor)
            self.finger_motors.append(finger_motor)

        # For collision. Main body circle. Signature: Circle(body, radius,
        # offset).
        robot_group = 1
        body_shape = pm.Circle(body, self.radius, (0, 0))
        body_shape.filter = pm.ShapeFilter(group=robot_group)
        body_shape.friction = 0.5
        self.space.add(body_shape)
        # the fingers
        finger_shapes = []
        for finger_body, finger_verts, finger_side in zip(
                self.finger_bodies, finger_vertices, [-1, 1]):
            finger_subshapes = []
            for finger_subverts in finger_verts:
                finger_subshape = pm.Poly(finger_body, finger_subverts)
                finger_subshape.filter = pm.ShapeFilter(group=robot_group)
                # grippy fingers
                finger_subshape.friction = 4.0
                finger_subshapes.append(finger_subshape)
            self.space.add(*finger_subshapes)
            finger_shapes.append(finger_subshapes)

        # graphics setup
        # draw a circular body
        circ_body_in = r.make_circle(radius=self.radius - LINE_THICKNESS,
                                     res=100)
        circ_body_out = r.make_circle(radius=self.radius, res=100)
        robot_colour = COLOURS_RGB['grey']
        dark_robot_colour = darken_rgb(robot_colour)
        light_robot_colour = lighten_rgb(robot_colour, 5)
        circ_body_in.set_color(*robot_colour)
        circ_body_out.set_color(*dark_robot_colour)

        # draw the two fingers
        self.finger_xforms = []
        finger_outer_geoms = []
        finger_inner_geoms = []
        for finger_outer_subshapes, finger_inner_verts, finger_side in zip(
                finger_shapes, finger_inner_vertices, [-1, 1]):
            finger_xform = r.Transform()
            self.finger_xforms.append(finger_xform)
            for finger_subshape in finger_outer_subshapes:
                vertices = [(v.x, v.y) for v in finger_subshape.get_vertices()]
                finger_outer_geom = r.make_polygon(vertices)
                finger_outer_geom.set_color(*robot_colour)
                finger_outer_geom.add_attr(finger_xform)
                finger_outer_geoms.append(finger_outer_geom)

            for vertices in finger_inner_verts:
                finger_inner_geom = r.make_polygon(vertices)
                finger_inner_geom.set_color(*light_robot_colour)
                finger_inner_geom.add_attr(finger_xform)
                finger_inner_geoms.append(finger_inner_geom)

        for geom in finger_outer_geoms:
            self.viewer.add_geom(geom)
        for geom in finger_inner_geoms:
            self.viewer.add_geom(geom)

        # draw some cute eyes
        eye_shapes = []
        self.pupil_transforms = []
        for x_sign in [-1, 1]:
            eye = r.make_circle(radius=0.2 * self.radius, res=20)
            eye.set_color(1.0, 1.0, 1.0)
            eye_base_transform = r.Transform().set_translation(
                x_sign * 0.4 * self.radius, 0.3 * self.radius)
            eye.add_attr(eye_base_transform)
            pupil = r.make_circle(radius=0.12 * self.radius, res=10)
            pupil.set_color(0.1, 0.1, 0.1)
            # The pupils point forward slightly
            pupil_transform = r.Transform()
            pupil.add_attr(r.Transform().set_translation(
                0, self.radius * 0.07))
            pupil.add_attr(pupil_transform)
            pupil.add_attr(eye_base_transform)
            self.pupil_transforms.append(pupil_transform)
            eye_shapes.extend([eye, pupil])
        # join them together
        self.robot_xform = r.Transform()
        robot_compound = r.Compound([circ_body_out, circ_body_in, *eye_shapes])
        robot_compound.add_attr(self.robot_xform)
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
        self.robot_xform.set_translation(*self.robot_body.position)
        self.robot_xform.set_rotation(self.robot_body.angle)
        for finger_xform, finger_body in zip(self.finger_xforms,
                                             self.finger_bodies):
            finger_xform.set_translation(*finger_body.position)
            finger_xform.set_rotation(finger_body.angle)
        for pupil_xform, pupil_body in zip(self.pupil_transforms,
                                           self.pupil_bodies):
            pupil_xform.set_rotation(pupil_body.angle - self.robot_body.angle)


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
    TRIANGLE = 'triangle'
    SQUARE = 'square'
    PENTAGON = 'pentagon'
    HEXAGON = 'hexagon'
    OCTAGON = 'octagon'
    CIRCLE = 'circle'


# limited set of types and colours to use for random generation
SHAPE_TYPES = [
    ShapeType.SQUARE,
    ShapeType.PENTAGON,
    ShapeType.OCTAGON,
    ShapeType.CIRCLE,
]
SHAPE_COLOURS = ['red', 'green', 'blue', 'yellow']


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
        else:
            # these are free-form shapes b/c no helpers exist in Pymunk
            if self.shape_type == ShapeType.TRIANGLE:
                factor = 0.8  # shrink to make it look more sensible
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
            self.space.add(body)
            shape = pm.Poly(body, poly_verts)

        shape.friction = 0.5
        self.space.add(shape)

        trans_joint = pm.PivotJoint(self.space.static_body, body, (0, 0),
                                    (0, 0))
        trans_joint.max_bias = 0
        trans_joint.max_force = 1.5
        self.space.add(trans_joint)
        rot_joint = pm.GearJoint(self.space.static_body, body, 0.0, 1.0)
        rot_joint.max_bias = 0
        rot_joint.max_force = 0.1
        self.space.add(rot_joint)

        # Drawing
        if self.shape_type == ShapeType.SQUARE:
            geom_inner = r.make_square(side_len - 2 * LINE_THICKNESS)
            geom_outer = r.make_square(side_len)
        elif self.shape_type == ShapeType.CIRCLE:
            geom_inner = r.make_circle(radius=self.shape_size - LINE_THICKNESS,
                                       res=100)
            geom_outer = r.make_circle(radius=self.shape_size, res=100)
        elif self.shape_type == ShapeType.OCTAGON \
                or self.shape_type == ShapeType.HEXAGON \
                or self.shape_type == ShapeType.PENTAGON \
                or self.shape_type == ShapeType.TRIANGLE:
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


# #############################################################################
# Sensor region for pushing shapes into.
# #############################################################################


class GoalRegion(Entity):
    """A goal region that the robot should push certain shapes into. It's up to
    the caller to figure out exactly which shapes & call methods for collision
    checking/scoring."""
    def __init__(self, x, y, h, w, colour_name, ent_index):
        self.x = x
        self.y = y
        assert h > 0, w > 0
        self.h = h
        self.w = w
        self.base_colour = COLOURS_RGB[colour_name]
        self.ent_index = ent_index

    def setup(self, *args, **kwargs):
        super().setup(*args, **kwargs)

        # the space only needs a sensor body
        self.goal_body = pm.Body(body_type=pm.Body.STATIC)
        self.goal_shape = pm.Poly.create_box(self.goal_body, (self.w, self.h))
        self.goal_shape.sensor = True
        self.goal_body.position = (self.x + self.w / 2, self.y - self.h / 2)
        self.space.add(self.goal_body, self.goal_shape)

        # Making visual display: region should consist of very lightly shaded
        # rectangle, surrounded by darker stippled border. Ideally corners on
        # the rectangle should be rounded.
        self.rect_xform = r.Transform()
        self.rect_xform.set_translation(*self.goal_body.position)

        inner_colour = lighten_rgb(self.base_colour, times=2)
        inner_rect = r.make_rect(width=self.w, height=self.h, filled=True)
        inner_rect.set_color(*inner_colour)
        inner_rect.add_attr(self.rect_xform)
        self.viewer.add_geom(inner_rect)

        outer_colour = self.base_colour
        outer_rect = r.make_rect(width=self.w, height=self.h, filled=False)
        outer_rect.set_color(*outer_colour)
        outer_rect.add_attr(r.LineStyle(0x00FF))
        outer_rect.set_linewidth(250 * LINE_THICKNESS)
        outer_rect.add_attr(self.rect_xform)
        self.viewer.add_geom(outer_rect)

    def update(self, dt):
        # nothing really needs to be done here, AFAICT
        pass

    def get_overlapping_ents(self, contained=False):
        """Get all entities overlapping this region.

        Args:
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
                s for s in overlap_shapes if goal_bb.contains(s.bb)
            }

        # now look up all indexed entities that own at least one overlapping
        # shape
        relevant_ents = set()
        for shape in overlap_shapes:
            try:
                ent = self.ent_index.entity_for(shape)
            except KeyError:
                # shape not in index
                continue
            relevant_ents.add(ent)

        # if necessary, filter the entities so that only those with *all*
        # shapes within the region are included
        if contained:
            new_relevant_ents = set()
            for relevant_ent in relevant_ents:
                shapes = self.ent_index.shapes_for(relevant_ent)
                if shapes <= overlap_shapes:
                    new_relevant_ents.add(relevant_ent)
            relevant_ents = new_relevant_ents

        return relevant_ents

    def pre_draw(self):
        # just so we can see if the thing accidentally moves :-)
        self.rect_xform.set_translation(*self.goal_body.position)
