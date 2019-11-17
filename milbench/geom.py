"""Geometry."""

import math

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
