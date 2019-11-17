"""Common visual settings and colour tools for this benchmark."""

import colorsys


def rgb(r, g, b):
    return (r / 255.0, g / 255.0, b / 255.0)


LINE_THICKNESS = 0.015
COLOURS_RGB = {
    # Most of these colours are from Wikipedia. They are all pastel base
    # colours useful for shape bodies; use darken_rgb to get highlights (e.g.
    # borders) and lighten_rgb to get get fades useful for backgrounds.
    'red': rgb(255, 192, 203),  # pink
    'green': rgb(170, 240, 209),  # magic mint
    'blue': rgb(137, 207, 240),  # baby blue
    'yellow': rgb(255, 229, 180),  # peach
    'grey': rgb(162, 163, 175),  # cool grey (not sure which one)
    'brown': rgb(224, 171, 118),  # buff
}


def darken_rgb(rgb):
    """Produce a darker version of a base colour."""
    h, l, s = colorsys.rgb_to_hls(*rgb)
    hls_new = (h, max(0, l * 0.9), s)
    return colorsys.hls_to_rgb(*hls_new)


def lighten_rgb(rgb):
    """Produce a lighter version of a given base colour."""
    h, l, s = colorsys.rgb_to_hls(*rgb)
    hls_new = (h, min(0, l * 1.1), s)
    return colorsys.hls_to_rgb(*hls_new)
