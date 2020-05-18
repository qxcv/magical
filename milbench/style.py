"""Common visual settings and colour tools for this benchmark."""

import colorsys


def rgb(r, g, b):
    return (r / 255.0, g / 255.0, b / 255.0)


def darken_rgb(rgb):
    """Produce a darker version of a base colour."""
    h, l, s = colorsys.rgb_to_hls(*rgb)
    hls_new = (h, max(0, l * 0.9), s)
    return colorsys.hls_to_rgb(*hls_new)


def lighten_rgb(rgb, times=1):
    """Produce a lighter version of a given base colour."""
    h, l, s = colorsys.rgb_to_hls(*rgb)
    mult = 1.4 ** times
    hls_new = (h, 1 - (1 - l) / mult, s)
    return colorsys.hls_to_rgb(*hls_new)


GOAL_LINE_THICKNESS = 0.01
SHAPE_LINE_THICKNESS = 0.015
ROBOT_LINE_THICKNESS = 0.01
COLOURS_RGB = {
    # Original colours from Wikipedia. They were all pastel base colours useful
    # for shape bodies; can use darken_rgb to get highlights (e.g. borders) and
    # lighten_rgb to get get fades useful for backgrounds.

    # Unfortunately pastel RGBY originally looked too much like Google or
    # Microsoft's corporate brands. I tried to replace them with
    # Berkeley-branded versions of these colours from
    # https://brand.berkeley.edu/colors/ (possibly lightened).

    # 'red': rgb(255, 192, 203),  # pink
    # 'green': rgb(170, 240, 209),  # magic mint
    # 'blue': rgb(137, 207, 240),  # baby blue
    # 'yellow': rgb(255, 229, 180),  # peach

    'blue': lighten_rgb(rgb(0x3B, 0x7E, 0xA1), 1.7),  # founder's rock
    'yellow': lighten_rgb(rgb(0xFD, 0xB5, 0x15), 1.7),  # california gold
    'red': lighten_rgb(rgb(0xEE, 0x1F, 0x60), 1.7),  # rose garden
    'green': lighten_rgb(rgb(0x85, 0x94, 0x38), 1.7),  # soybean

    'grey': rgb(162, 163, 175),  # cool grey (not sure which one)
    'brown': rgb(224, 171, 118),  # buff
}
# "zoom out" factor when rendering arena; values above 1 will show parts of the
# arena border
ARENA_ZOOM_OUT = 1.02
