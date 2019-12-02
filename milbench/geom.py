"""Geometry."""

import math
import warnings

import pymunk as pm


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
                      arena_xyhw,
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
            same relative positions and orientations.
        arena_xhyw ([int]): bounding box to place the bodies in.
        rand_pos (bool): should position be randomised?
        rand_rot (bool): should rotation be randomised?
        rejection_tests ([(*locals()) -> bool]): additional rejection tests to
            apply. If any one of these functions returns "True", then the shape
            pose will be rejected and re-sampled. Useful for, e.g., ensuring
            that placed shapes do not coincide with certain existing objects.

    Returns (int): number of random placements attempted before finding a
        successful one."""
    assert len(bodies) >= 1, "no bodies given (?)"

    # If we exceed this many tries then fitting is probably impossible, or
    # impractically hard. We'll warn if we get close to that number.
    max_tries = 10000
    warn_tries = int(max_tries / 10)

    n_tries = 0
    while n_tries < max_tries:
        # I think easiest way to do this is to first rotate every body *about
        # the origin*, then pick a random translation within the given bounds.
        # There might exist easier and more general ways to do this, though.
        raise NotImplementedError(
            "still need to sample random rigid transform, apply it, and test "
            "for collisions")
        n_tries += 1
    else:
        raise PlacementError(
            f"Could not place bodies {bodies} in space {space} after "
            f"{n_tries} attempts. rand_pos={rand_pos}, rand_rot={rand_rot}, "
            f"arena_xhyw={arena_xyhw}.")

    if n_tries > warn_tries:
        warnings.warn(f"Took {n_tries}>{warn_tries} samples to place shape.")

    return n_tries
