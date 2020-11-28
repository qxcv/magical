"""Tools for working with geometric primitives and randomising aspects of
geometry (shape, pose, etc.)."""

from collections.abc import Iterable, Sequence
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


def compute_star_verts(n_points, out_radius, in_radius):
    """Vertices for a star. `n_points` controls the number of points;
    `out_radius` controls distance from points to centre; `in_radius` controls
    radius from "depressions" (the things between points) to centre."""
    assert n_points >= 3
    vertices = []
    out_vertex = pm.vec2d.Vec2d(0, out_radius)
    in_vertex = pm.vec2d.Vec2d(0, in_radius)
    for point_num in range(n_points):
        out_angle = point_num * 2 * math.pi / n_points
        vertices.append(out_vertex.rotated(out_angle))
        in_angle = (2 * point_num + 1) * math.pi / n_points
        vertices.append(in_vertex.rotated(in_angle))
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


def pm_randomise_pose(space,
                      bodies,
                      arena_lrbt,
                      rng,
                      rand_pos=True,
                      rand_rot=True,
                      rel_pos_linf_limit=None,
                      rel_rot_limit=None,
                      ignore_shapes=None,
                      rejection_tests=()):
    r"""Do rejection sampling to choose a position and/or orientation which
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
        rand_pos (bool or [bool]): should position be randomised? (optionally
            specified separately for each entity)
        rand_rot (bool or [bool]): should rotation be randomised? (optionally
            specified for each entity)
        rel_pos_linf_limit (float or [float]): bound on the $\ell_\infty$
            distance between new sampled position and original position.
            (optionally specified per-entity)
        rel_rot_limit (float or [float]): maximum difference (in radians)
            between original main body orientation and new main body
            orientation. (optionally per-entity)
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

    shape_set = set()
    for body in bodies:
        shape_set.update(body.shapes)

    if ignore_shapes is not None:
        ignore_set = set(ignore_shapes)
    else:
        ignore_set = set()

    arena_l, arena_r, arena_b, arena_t = arena_lrbt

    if rel_pos_linf_limit is not None:
        assert 0 <= rel_pos_linf_limit
        init_x, init_y = main_body.position
        pos_x_minmax = (max(arena_l, init_x - rel_pos_linf_limit),
                        min(arena_r, init_x + rel_pos_linf_limit))
        pos_y_minmax = (max(arena_b, init_y - rel_pos_linf_limit),
                        min(arena_t, init_y + rel_pos_linf_limit))
    else:
        pos_x_minmax = (arena_l, arena_r)
        pos_y_minmax = (arena_b, arena_t)

    if rel_rot_limit is not None:
        assert 0 <= rel_rot_limit
        rot_min = orig_main_angle - rel_rot_limit
        rot_max = orig_main_angle + rel_rot_limit
    else:
        rot_min = -np.pi
        rot_max = np.pi

    # If we exceed this many tries then fitting is probably impossible, or
    # impractically hard. We'll warn if we get anywhere close to that number.
    max_tries = 10000
    warn_tries = int(max_tries / 10)

    n_tries = 0
    while n_tries < max_tries:
        # generate random position
        if rand_pos:
            new_main_body_pos = Vec2d(rng.uniform(*pos_x_minmax),
                                      rng.uniform(*pos_y_minmax))
        else:
            new_main_body_pos = orig_main_pos

        # generate random orientation
        if rand_rot:
            new_angle = rng.uniform(rot_min, rot_max)
        else:
            new_angle = orig_main_angle

        # apply new position/orientation to all bodies
        pm_shift_bodies(space,
                        bodies,
                        position=new_main_body_pos,
                        angle=new_angle)

        # apply collision tests
        reject = False
        for shape in shape_set:
            query_result = space.shape_query(shape)
            collisions = set(r.shape for r in query_result) - ignore_set
            # reject if we have any (non-self-)collisions
            if len(collisions) > 0:
                reject = True
                break

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


def _listify(value, n):
    if isinstance(value, Iterable):
        # if `value` is already an iterable, then cast it to a sequence type
        # and return it
        if not isinstance(value, Sequence):
            rv = list(value)
        else:
            rv = value
        assert len(rv) == n, (len(rv), n)
        return rv
    # otherwise, duplicate value `n` times
    return [value] * n


def pm_randomise_all_poses(space,
                           entities,
                           arena_lrbt,
                           rng,
                           rand_pos=True,
                           rand_rot=True,
                           rel_pos_linf_limits=None,
                           rel_rot_limits=None,
                           ignore_shapes=None,
                           max_retries=10,
                           rejection_tests=()):
    """Randomise poses of *all* entities in the given list of entities."""
    # create placeholder limits if necessary
    nents = len(entities)
    rel_pos_linf_limits = _listify(rel_pos_linf_limits, nents)
    rel_rot_limits = _listify(rel_rot_limits, nents)
    rand_pos = _listify(rand_pos, nents)
    rand_rot = _listify(rand_rot, nents)

    for retry in range(max_retries):
        # disable collisions for all entities
        ent_filters = []
        for entity in entities:
            shape_filters = []
            for s in entity.shapes:
                shape_filters.append(s.filter)
                # categories=0 makes it collide with nothing
                s.filter = s.filter._replace(categories=0)
            ent_filters.append(shape_filters)

        for (entity, shape_filters, pos_limit, rot_limit, should_rand_pos,
             should_rand_rot) in zip(entities, ent_filters,
                                     rel_pos_linf_limits, rel_rot_limits,
                                     rand_pos, rand_rot):
            # re-enable collisions for this entity (previous entities will
            # already have collisions enabled, and later entities will still
            # have collisions disabled)
            for s, filt in zip(entity.shapes, shape_filters):
                s.filter = filt

            # now randomise pose, avoiding entities that have previously been
            # placed or which are not in the supplied list
            try:
                pm_randomise_pose(space,
                                  entity.bodies,
                                  arena_lrbt,
                                  rng,
                                  rand_pos=should_rand_pos,
                                  rand_rot=should_rand_rot,
                                  rel_pos_linf_limit=pos_limit,
                                  rel_rot_limit=rot_limit,
                                  ignore_shapes=ignore_shapes,
                                  rejection_tests=rejection_tests)
            except PlacementError as ex:
                if retry == max_retries - 1:
                    raise
                print(f"Got PlacementError ({ex}) on retry {retry + 1}"
                      f"/{max_retries}, restarting")
                break
        else:
            break


def randomise_hw(min_side, max_side, rng, current_hw=None, linf_bound=None):
    """Randomise height and width parameters within some supplied bounds.
    Useful for randomising goal region height/width in a reasonably uniform
    way."""
    assert min_side <= max_side
    minima = np.asarray((min_side, min_side))
    maxima = np.asarray((max_side, max_side))
    if linf_bound is not None:
        assert linf_bound == float(linf_bound)
        assert current_hw is not None
        assert len(current_hw) == 2
        current_hw = np.asarray(current_hw)
        minima = np.maximum(minima, current_hw - linf_bound)
        maxima = np.minimum(maxima, current_hw + linf_bound)
    h, w = rng.uniform(minima, maxima)
    return h, w


def pm_shift_bodies(space, bodies, position=None, angle=None):
    """Apply a rigid transform to the given bodies to move them into the given
    position and/or angle. Note that position and angle are specified for the
    first body only; later bodies are modelled as if attached to the first."""
    assert len(bodies) >= 1

    root_angle = bodies[0].angle
    root_position = bodies[0].position
    if angle is None:
        angle = root_angle
    if position is None:
        position = root_position

    # cast to right types
    position = pm.Vec2d(position)
    angle = float(angle)

    for body in bodies:
        local_angle_delta = body.angle - root_angle
        local_pos_delta = body.position - root_position
        body.angle = angle + local_angle_delta
        body.position = position + local_pos_delta.rotated(angle - root_angle)
        space.reindex_shapes_for_body(body)
