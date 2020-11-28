"""
2D rendering framework from Gym. MIT license notice below.

Copyright (c) 2016 OpenAI (https://openai.com)

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import division

import os
import sys
import warnings

import six

if "Apple" in sys.version:
    if 'DYLD_FALLBACK_LIBRARY_PATH' in os.environ:
        os.environ['DYLD_FALLBACK_LIBRARY_PATH'] += ':/usr/lib'
        # (JDS 2016/04/15): avoid bug on Anaconda 2.3.0 / Yosemite

try:
    import pyglet
except ImportError:
    raise ImportError('''
    Cannot import pyglet. HINT: you can install pyglet directly via 'pip
    install pyglet'. But if you really just want to install all Gym
    dependencies and not have to think about it, 'pip install -e .[all]' or
    'pip install gym[all]' will do it.
    ''')

try:
    from pyglet import gl
except ImportError:
    raise ImportError('''
    Error occurred while running `from pyglet.gl import *` HINT: make sure you
    have OpenGL install. On Ubuntu, you can run 'apt-get install
    python-opengl'. If you're running on a server, you may need a virtual frame
    buffer; something like this should work: 'xvfb-run -s \"-screen 0
    1400x900x24\" python <your_script.py>'
    ''')

import math  # noqa: E402

import numpy as np  # noqa: E402

from magical.pyglet_backport.image import Framebuffer  # noqa: E402
from magical.pyglet_backport.image import Renderbuffer, Texture

RAD2DEG = 57.29577951308232


def get_display(spec):
    """Convert a display specification (such as :0) into an actual Display
    object.
    Pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return None
    elif isinstance(spec, six.string_types):
        return pyglet.canvas.Display(spec)
    else:
        raise RuntimeError(
            'Invalid display spec: {}. (Must be string like :0 or None.)'.
            format(spec))


def get_offscreen_fbo(width, height, msaa_samples=4):
    fbo = Framebuffer()
    # using None for the format specifier in Texture.create means we have to
    # allocate memory ourselves, which is important here because we seem to
    # need to pass nullptr to allocation routine's destination arg (why?).
    if msaa_samples > 1:
        fbo._colour_texture = Texture.create(
            width,
            height,
            target=gl.GL_TEXTURE_2D_MULTISAMPLE,
            internalformat=None)
        gl.glTexImage2DMultisample(gl.GL_TEXTURE_2D_MULTISAMPLE, msaa_samples,
                                   gl.GL_RGB, width, height, True)
    else:
        fbo._colour_texture = Texture.create(width,
                                             height,
                                             internalformat=None)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, width, height, 0,
                        gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None)
    fbo.attach_texture(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0,
                       fbo._colour_texture)
    fbo._depth_rb = Renderbuffer(width,
                                 height,
                                 gl.GL_DEPTH_COMPONENT,
                                 samples=msaa_samples)
    fbo.attach_renderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT,
                            fbo._depth_rb)
    assert fbo.is_complete, \
        "FramebufferObject not complete after attaching all buffers (bug?); " \
        f"status {fbo.get_status()}"
    return fbo


def blit_fbo(width, height, src_id, target_id, target_image=gl.GL_BACK):
    # For drawing a multisampled FBO to a non-multisampled FBO or to the
    # screen. See
    # https://www.khronos.org/opengl/wiki/Multisampling#Allocating_a_Multisample_Render_Target
    gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, src_id)
    gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, target_id)
    gl.glDrawBuffer(target_image)
    gl.glBlitFramebuffer(0, 0, width, height, 0, 0, width, height,
                         gl.GL_COLOR_BUFFER_BIT, gl.GL_NEAREST)


class Viewer(object):
    def __init__(self,
                 width,
                 height,
                 visible,
                 display=None,
                 background_rgb=(1, 1, 1)):
        display = get_display(display)

        self.width = width
        self.height = height
        self.background_rgb = background_rgb
        self.window = pyglet.window.Window(width=width,
                                           height=height,
                                           display=display,
                                           visible=visible)
        if not visible:
            # get around stupid bug (?) where OpenGL refuses to render anything
            # to FBOs until the window is displayed
            self.window.set_visible(True)
            self.window.set_visible(False)
        # need to use a second FBO to actually get image data
        self.render_fbo = get_offscreen_fbo(width, height, msaa_samples=1)
        self.no_msaa_fbo = get_offscreen_fbo(width, height, msaa_samples=1)
        self.window.on_close = self.window_closed_by_user
        self.isopen = True
        self.reset_geoms()
        self.transforms = Transform()

        gl.glEnable(gl.GL_BLEND)
        # tricks from OpenAI's multiagent particle env repo:
        # gl.glEnable(gl.GL_MULTISAMPLE)
        gl.glEnable(gl.GL_LINE_SMOOTH)
        # gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_DONT_CARE)
        gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
        gl.glLineWidth(2.0)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    def reset_geoms(self):
        self.geoms = []
        self.onetime_geoms = []

    def close(self):
        self.window.close()

    def window_closed_by_user(self):
        self.isopen = False

    def set_bounds(self, left, right, bottom, top, rotation=0.0):
        assert right > left and top > bottom
        scalex = self.width / (right - left)
        scaley = self.height / (top - bottom)
        self.transform = Transform(translation=(-left * scalex,
                                                -bottom * scaley),
                                   scale=(scalex, scaley))

    def set_cam_follow(self, source_xy_world, target_xy_01, viewport_hw_world,
                       rotation):
        """Set camera so that point at `source_xy_world` (in world coordinates)
        appears at `screen_target_xy` (in screen coordinates, in [0,1] on each
        axis), and so that viewport covers region defined by
        `viewport_hw_world` in world coordinates. Oh, and the world is rotated
        by `rotation` around the point `source_xy_world` before doing anything
        else."""
        world_h, world_w = viewport_hw_world
        scalex = self.width / world_w
        scaley = self.height / world_h
        target_x_01, target_y_01 = target_xy_01
        self.transform = TransformEgocentric(centre=source_xy_world,
                                             newpos=(world_w * target_x_01,
                                                     world_h * target_y_01),
                                             scale=(scalex, scaley),
                                             rotation=rotation)

    def add_geom(self, geom):
        self.geoms.append(geom)

    def add_onetime(self, geom):
        self.onetime_geoms.append(geom)

    def render(self, return_rgb_array=False, update_foreground=True):
        # switch to window and ONLY render to FBO
        self.window.switch_to()
        self.render_fbo.bind()

        # actual rendering
        gl.glClearColor(*self.background_rgb, 1)
        self.window.clear()
        self.window.dispatch_events()
        self.transform.enable()
        for geom in self.geoms:
            geom.render()
        for geom in self.onetime_geoms:
            geom.render()
        self.transform.disable()
        self.onetime_geoms = []

        # done, don't need FBO in main context
        self.render_fbo.unbind()

        # optionally write RGB array
        arr = None
        if return_rgb_array:
            # this initial blit call will crash if you're calling it from a
            # fork()ed subprocess; use the 'spawn' MP method instead of 'fork'
            # to make it work
            blit_fbo(self.width, self.height, self.render_fbo.id,
                     self.no_msaa_fbo.id, gl.GL_COLOR_ATTACHMENT0)
            image_data = self.no_msaa_fbo._colour_texture.get_image_data(
                fmt='RGB', gl_format=gl.GL_RGB)
            arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            arr = arr.reshape(self.height, self.width, 3)[::-1]

        # optionally blit to main window (should be on by default, but we can
        # skip it if we only want an offscreen render)
        if update_foreground:
            gl.glClearColor(*self.background_rgb, 1)
            self.window.clear()
            blit_fbo(self.width, self.height, self.render_fbo.id, 0)
            self.window.flip()

        return arr if return_rgb_array else self.isopen

    # Convenience
    def draw_circle(self, radius=10, res=30, filled=True, **attrs):
        geom = make_circle(radius=radius, res=res, filled=filled)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_polygon(self, v, filled=True, **attrs):
        geom = make_polygon(v=v, filled=filled)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_polyline(self, v, **attrs):
        geom = make_polyline(v=v)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_line(self, start, end, **attrs):
        geom = Line(start, end)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def get_array(self):
        self.window.flip()
        image_data = pyglet.image.get_buffer_manager() \
            .get_color_buffer().get_image_data()
        self.window.flip()
        arr = np.fromstring(image_data.data, dtype=np.uint8, sep='')
        arr = arr.reshape(self.height, self.width, 4)
        return arr[::-1, :, 0:3]

    def __del__(self):
        try:
            self.close()
        except Exception as ex:
            warnings.warn(f'Exception on env auto-close, you probably need '
                          f'to manually close your Gym envs. Error messsage: '
                          f'{ex!s}')


def _add_attrs(geom, attrs):
    if "color" in attrs:
        geom.set_color(*attrs["color"])
    if "linewidth" in attrs:
        geom.set_linewidth(attrs["linewidth"])


class Geom(object):
    def __init__(self):
        self._color = Color((0, 0, 0, 1.0))
        self.attrs = [self._color]

    def render(self):
        for attr in reversed(self.attrs):
            attr.enable()
        self.render1()
        for attr in self.attrs:
            attr.disable()

    def render1(self):
        raise NotImplementedError

    def add_attr(self, attr):
        self.attrs.append(attr)
        return self

    def set_color(self, r, g, b):
        self._color.vec4 = (r, g, b, 1)
        return self


class Attr(object):
    def enable(self):
        raise NotImplementedError

    def disable(self):
        pass


class Transform(Attr):
    def __init__(self, translation=(0.0, 0.0), rotation=0.0, scale=(1, 1)):
        self.set_translation(*translation)
        self.set_rotation(rotation)
        self.set_scale(*scale)

    def enable(self):
        gl.glPushMatrix()
        # translate to GL loc ppint
        gl.glTranslatef(self.translation[0], self.translation[1], 0)
        gl.glRotatef(RAD2DEG * self.rotation, 0, 0, 1.0)
        gl.glScalef(self.scale[0], self.scale[1], 1)

    def disable(self):
        gl.glPopMatrix()

    def set_translation(self, newx, newy):
        self.translation = (float(newx), float(newy))
        return self

    def set_rotation(self, new):
        self.rotation = float(new)
        return self

    def set_scale(self, newx, newy):
        self.scale = (float(newx), float(newy))
        return self


class TransformEgocentric(Attr):
    """Transform class for egocentric top-down camera. Rotates the world around
    a particular point, moves that point to another position, then scales the
    world."""
    def __init__(self, centre, newpos, rotation, scale):
        self.centre = centre
        self.newpos = newpos
        self.rotation = rotation
        self.scale = scale

    def enable(self):
        gl.glPushMatrix()
        gl.glScalef(self.scale[0], self.scale[1], 1)
        gl.glTranslatef(self.newpos[0], self.newpos[1], 0)
        gl.glRotatef(RAD2DEG * -self.rotation, 0, 0, 1.0)
        gl.glTranslatef(-self.centre[0], -self.centre[1], 0)

    def disable(self):
        gl.glPopMatrix()


class Color(Attr):
    def __init__(self, vec4):
        self.vec4 = vec4

    def enable(self):
        gl.glColor4f(*self.vec4)


class LineStyle(Attr):
    def __init__(self, style):
        self.style = style

    def enable(self):
        gl.glEnable(gl.GL_LINE_STIPPLE)
        gl.glLineStipple(1, self.style)

    def disable(self):
        gl.glDisable(gl.GL_LINE_STIPPLE)


class LineWidth(Attr):
    def __init__(self, stroke):
        self.stroke = stroke

    def enable(self):
        gl.glLineWidth(self.stroke)


class Point(Geom):
    def __init__(self):
        Geom.__init__(self)

    def render1(self):
        gl.glBegin(gl.GL_POINTS)  # draw point
        gl.glVertex3f(0.0, 0.0, 0.0)
        gl.glEnd()


class FilledPolygon(Geom):
    def __init__(self, v):
        Geom.__init__(self)
        self.v = v

    def render1(self):
        if len(self.v) == 4:
            gl.glBegin(gl.GL_QUADS)
        elif len(self.v) > 4:
            gl.glBegin(gl.GL_POLYGON)
        else:
            gl.glBegin(gl.GL_TRIANGLES)
        for p in self.v:
            gl.glVertex3f(p[0], p[1], 0)  # draw each vertex
        gl.glEnd()


def make_circle(radius=10, res=30, filled=True):
    points = []
    for i in range(res):
        ang = 2 * math.pi * i / res
        points.append((math.cos(ang) * radius, math.sin(ang) * radius))
    if filled:
        return FilledPolygon(points)
    else:
        return PolyLine(points, True)


def make_rect(width=10, height=10, filled=True):
    rad_h = height / 2
    rad_w = width / 2
    points = [(-rad_w, rad_h), (rad_w, rad_h), (rad_w, -rad_h),
              (-rad_w, -rad_h)]
    if filled:
        return FilledPolygon(points)
    else:
        return PolyLine(points, True)


def make_square(side_length=10, filled=True):
    return make_rect(side_length, side_length, filled)


def make_polygon(v, filled=True):
    if filled:
        return FilledPolygon(v)
    else:
        return PolyLine(v, True)


def make_polyline(v):
    return PolyLine(v, False)


def make_capsule(length, width):
    l, r, t, b = 0, length, width / 2, -width / 2
    box = make_polygon([(l, b), (l, t), (r, t), (r, b)])
    circ0 = make_circle(width / 2)
    circ1 = make_circle(width / 2)
    circ1.add_attr(Transform(translation=(length, 0)))
    geom = Compound([box, circ0, circ1])
    return geom


class Compound(Geom):
    def __init__(self, gs):
        Geom.__init__(self)
        self.gs = gs

    def render1(self):
        for g in self.gs:
            g.render()


class PolyLine(Geom):
    def __init__(self, v, close):
        Geom.__init__(self)
        self.v = v
        self.close = close
        self.linewidth = LineWidth(1)
        self.add_attr(self.linewidth)

    def render1(self):
        gl.glBegin(gl.GL_LINE_LOOP if self.close else gl.GL_LINE_STRIP)
        for p in self.v:
            gl.glVertex3f(p[0], p[1], 0)  # draw each vertex
        gl.glEnd()

    def set_linewidth(self, x):
        self.linewidth.stroke = x


class Line(Geom):
    def __init__(self, start=(0.0, 0.0), end=(0.0, 0.0)):
        Geom.__init__(self)
        self.start = start
        self.end = end
        self.linewidth = LineWidth(1)
        self.add_attr(self.linewidth)

    def render1(self):
        gl.glBegin(gl.GL_LINES)
        gl.glVertex2f(*self.start)
        gl.glVertex2f(*self.end)
        gl.glEnd()


class Image(Geom):
    def __init__(self, fname, width, height):
        Geom.__init__(self)
        self.width = width
        self.height = height
        img = pyglet.image.load(fname)
        self.img = img
        self.flip = False

    def render1(self):
        self.img.blit(-self.width / 2,
                      -self.height / 2,
                      width=self.width,
                      height=self.height)


# ================================================================


class SimpleImageViewer(object):
    def __init__(self, display=None, maxwidth=500):
        self.window = None
        self.isopen = False
        self.display = display
        self.maxwidth = maxwidth

    def imshow(self, arr):
        if self.window is None:
            height, width, _channels = arr.shape
            if width > self.maxwidth:
                scale = self.maxwidth / width
                width = int(scale * width)
                height = int(scale * height)
            self.window = pyglet.window.Window(width=width,
                                               height=height,
                                               display=self.display,
                                               vsync=False,
                                               resizable=True)
            self.width = width
            self.height = height
            self.isopen = True

            @self.window.event
            def on_resize(width, height):
                self.width = width
                self.height = height

            @self.window.event
            def on_close():
                self.isopen = False

        assert len(arr.shape) == 3, \
            "You passed in an image with the wrong number of dimensions"
        image = pyglet.image.ImageData(arr.shape[1],
                                       arr.shape[0],
                                       'RGB',
                                       arr.tobytes(),
                                       pitch=arr.shape[1] * -3)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER,
                           gl.GL_NEAREST)
        texture = image.get_texture()
        texture.width = self.width
        texture.height = self.height
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        texture.blit(0, 0)  # draw
        self.window.flip()

    def close(self):
        if self.isopen and sys.meta_path:
            # ^^^ check sys.meta_path to avoid 'ImportError: sys.meta_path is
            # None, Python is likely shutting down'
            self.window.close()
            self.isopen = False

    def __del__(self):
        self.close()
