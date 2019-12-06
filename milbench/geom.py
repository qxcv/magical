"""Geometry."""

import math
import warnings

import numpy as np
import pymunk as pm
from pymunk.vec2d import Vec2d


def regular_poly_circumrad(n_sides, side_length):
    """Circumradius of a regular polygon."""
    return side_length / (2 * math.sin(math.pi / n_sides))


def regular_poly_circ_rad_to_side_length(n_sides, rad):
    """Find side length that gives regular polygon with `n_sides` sides an
    equivalent area to a circle with radius `rad`."""
    p_n = math.pi / n_sides
    return 2 * rad * math.sqrt(p_n * math.tan(p_n))


def regular_poly_apothem_to_side_legnth(n_sides, apothem):
    """Compute side length for regular polygon with given apothem."""
    return 2 * apothem * math.tan(math.pi / n_sides)


def regular_poly_side_length_to_apothem(n_sides, side_length):
    """Compute apothem for regular polygon with given side length."""
    return side_length / (2 * math.tan(math.pi / n_sides))


def compute_regular_poly_verts(n_sides, side_length):
    """Vertices for regular polygon."""
    assert n_sides >= 3
    vertices = []
    step_angle = 2 * math.pi / n_sides
    radius = regular_poly_circumrad(n_sides, side_length)
    first_vertex = pm.vec2d.Vec2d(0, radius)
    for point_num in range(n_sides):
        angle = point_num * step_angle
        vertices.append(first_vertex.rotated(angle))
    vertices = [(v.x, v.y) for v in vertices]
    return vertices


def _convert_vec(v):
    if isinstance(v, pm.vec2d.Vec2d):
        return v.x, v.y
    if isinstance(v, (float, int)):
        return (v, v)
    x, y = v
    return (x, y)


def add_vecs(vec1, vec2):
    """Elementwise add vectors represented as vec2ds or tuples or whatever
    (even scalars, in which case they get broadcast). Return result as
    tuple."""
    x1, y1 = _convert_vec(vec1)
    x2, y2 = _convert_vec(vec2)
    return (x1 + x2, y1 + y2)


def mul_vecs(vec1, vec2):
    """Elementwise multiply vectors represented as vec2ds or tuples or
    whatever. Return result as tuple."""
    x1, y1 = _convert_vec(vec1)
    x2, y2 = _convert_vec(vec2)
    return (x1 * x2, y1 * y2)


def rotate_vec(vec, angle):
    # FIXME: this and related functions seem like design mistakes. I should
    # probably be using vec2d.Vec2d everywhere instead of using tuples.
    if not isinstance(vec, pm.vec2d.Vec2d):
        vec = pm.vec2d.Vec2d(*_convert_vec(vec))
    vec_r = vec.rotated(angle)
    return (vec_r.x, vec_r.y)


def rect_verts(w, h):
    # counterclockwise from top right
    return [
        pm.vec2d.Vec2d(w / 2, h / 2),
        pm.vec2d.Vec2d(-w / 2, h / 2),
        pm.vec2d.Vec2d(-w / 2, -h / 2),
        pm.vec2d.Vec2d(w / 2, -h / 2),
    ]


class PlacementError(Exception):
    """Raised when `pm_randomise_pose` cannot find an appropriate
    (non-colliding) pose for the given object."""
    pass


def pm_randomise_pose(space,
                      bodies,
                      arena_lrbt,
                      rng,
                      rand_pos=True,
                      rand_rot=True,
                      rejection_tests=()):
    """Do rejection sampling to choose a position and/or orientation which
    ensures the given bodies and their attached shapes do not collide with any
    other collidable shape in the space, while still falling entirely within
    arena_xyhw. Note that position/orientation will be chosen in terms of the
    first body in the given list of bodies, then later bodies attached to it
    will be repositioned and rotated accordingly.

    Args:
        space (pm.Space): the space to place the given bodies in.
        bodies ([pm.Body]): a list of bodies to place. They should maintain the
            same relative positions and orientations. Usually you'll only need
            to pass one body, although passing multiple bodies can be useful
            when the bodies have a pin joint (e.g. the robot's body is attached
            to its fingers this way).
        arena_lrbt ([int]): bounding box to place the bodies in.
        rand_pos (bool): should position be randomised?
        rand_rot (bool): should rotation be randomised?
        rejection_tests ([(locals()) -> bool]): additional rejection tests to
            apply. If any one of these functions returns "True", then the shape
            pose will be rejected and re-sampled. Useful for, e.g., ensuring
            that placed shapes do not coincide with certain existing objects.

    Returns (int): number of random placements attempted before finding a
        successful one."""
    assert rand_pos or rand_rot, \
        "need to randomise at least one thing, or placement may be impossible"
    assert len(bodies) >= 1, "no bodies given (?)"
    main_body = bodies[0]

    # Need to compute coordinates of other bodies in frame of main body
    saved_positions = [Vec2d(body.position) for body in bodies]
    saved_angles = [float(body.angle) for body in bodies]
    orig_main_angle = float(main_body.angle)
    orig_main_pos = Vec2d(main_body.position)
    local_pos_offsets = [
        (body.position - orig_main_pos).rotated(-orig_main_angle)
        for body in bodies
    ]
    local_angle_deltas = [body.angle - orig_main_angle for body in bodies]

    shape_set = set()
    for body in bodies:
        shape_set.update(body.shapes)

    arena_l, arena_r, arena_b, arena_t = arena_lrbt

    # If we exceed this many tries then fitting is probably impossible, or
    # impractically hard. We'll warn if we get anywhere close to that number.
    max_tries = 10000
    warn_tries = int(max_tries / 10)

    n_tries = 0
    while n_tries < max_tries:
        # generate random position
        if rand_pos:
            new_main_body_pos = Vec2d(
                rng.uniform(arena_l, arena_r), rng.uniform(arena_b, arena_t))
        else:
            new_main_body_pos = orig_main_pos

        # generate random orientation
        if rand_rot:
            new_angle = rng.uniform(-np.pi, np.pi)
        else:
            new_angle = orig_main_angle

        # apply new position/orientation to all bodies
        for idx, body in enumerate(bodies):
            new_local_offset = local_pos_offsets[idx].rotated(new_angle)
            body.position = new_main_body_pos + new_local_offset
            body.angle = new_angle + local_angle_deltas[idx]
            space.reindex_shapes_for_body(body)

        # apply collision tests
        collide_shapes = set()
        for shape in shape_set:
            query_result = space.shape_query(shape)
            collide_shapes.update(r.shape for r in query_result)
        # reject if we have any (non-self-)collisions
        reject = len(collide_shapes - shape_set) > 0

        # apply custom rejection tests, if any
        if not reject:
            for rejection_test in rejection_tests:
                reject = reject or rejection_test(locals())
                if reject:
                    break

        if not reject:
            # if we get to here without rejecting, then this is a good
            # orientation
            break

        n_tries += 1
    else:
        # reset original positions before raising exception
        for body, saved_pos, saved_angle in zip(bodies, saved_positions,
                                                saved_angles):
            body.position = saved_pos
            body.angle = saved_angle
            space.reindex_shapes_for_body(body)

        raise PlacementError(
            f"Could not place bodies {bodies} in space {space} after "
            f"{n_tries} attempts. rand_pos={rand_pos}, rand_rot={rand_rot}, "
            f"arena_lrbt={arena_lrbt}.")

    if n_tries > warn_tries:
        warnings.warn(f"Took {n_tries}>{warn_tries} samples to place shape.")

    return n_tries
