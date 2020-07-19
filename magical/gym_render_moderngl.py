#!/usr/bin/env python3
"""
2D rendering framework from Gym, ported to ModernGL, EGL, and OpenGL ES.

Copyright (c) 2016 OpenAI (https://openai.com)
Copyright (c) 2020 Sam Toyer

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
import abc
import math
import warnings
import weakref

import moderngl
import moderngl_window as mglw
import numpy as np
import pyglet

from magical import geom

SHADERS = {
    'simple_triangles_2d': {
        # vertex shader does a simple transform + draw
        'vertex_shader':
        '''#version 330
           in vec2 in_vert;
           uniform mat3 u_xform;
           void main() {
               vec3 pos_pad = u_xform * vec3(in_vert, 1.0);
               gl_Position = vec4(pos_pad.x, pos_pad.y, 0.0, 1.0);
           }''',
        # fragment shader simply copies uniform colour across
        'fragment_shader':
        '''#version 330
           uniform vec3 u_color;
           out vec4 f_color;
           void main() {
               f_color = vec4(u_color, 1.0);
           }''',
    }
}


def get_offscreen_fbo(width, height, msaa_samples=4):
    # FIXME: do this with ModernGL and maybe EGL, not Pyglet
    raise NotImplementedError("need to re-implement this with ModernGL")

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
        "FramebufferObject not complete after attaching all buffers (?)"
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

        # FIXME(sam): none of these options are likely to work in OpenGL ES.
        # Need to re-implement everything as shaders.
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

    def render(self, return_rgb_array=False, update_foreground=True):
        # switch to window and ONLY render to FBO
        # FIXME(sam): this is a bad idea. Instead, I should render to EGL by
        # default.
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
            arr = np.frombuffer(image_data.data, dtype=np.uint8)
            arr = arr.reshape(self.height, self.width, 3)[::-1]

        # optionally blit to main window (should be on by default, but we can
        # skip it if we only want an offscreen render)
        if update_foreground:
            gl.glClearColor(*self.background_rgb, 1)
            self.window.clear()
            blit_fbo(self.width, self.height, self.render_fbo.id, 0)
            self.window.flip()

        return arr if return_rgb_array else self.isopen

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


class ProgramIndex:
    """An index of OpenGL programs. Will compile each program under a given
    context each time it needs to be accessed."""
    def __init__(self, ctx, prog_dict):
        self._ctx = ctx
        self._source_dict = prog_dict
        self._compiled_dict = {}

    def __getattr__(self, name):
        try:
            return self[name]
        except IndexError as ex:
            raise AttributeError(str(ex))

    def __getitem__(self, name):
        if name not in self._compiled_dict and name in self._source_dict:
            source = self._source_dict[name]
            self._compiled_dict[name] = self._ctx.program(**source)
        if name in self._compiled_dict:
            return self._compiled_dict[name]
        raise AttributeError(f"no program '{name}' (available: "
                             f"{sorted(self._source_dict.keys())})")

    def __dir__(self):
        return [super().__dir__(), *self._source_dict.keys()]


class DrawContext:
    """Represents a single rendering context (e.g. an OpenGL context attached
    to a window, or an OpenGL context attached to an offscreen FBO). This API
    is primarily intended to be used by Drawables."""
    def __init__(self, moderngl_ctx):
        self.moderngl_ctx = moderngl_ctx
        self.programs = ProgramIndex(moderngl_ctx, SHADERS)
        # use weak keys just in case one of the drawables keeps a reference to
        # this context…
        self._drawable_data = weakref.WeakKeyDictionary()

    def set_drawable_data(self, drawable, return_data):
        self._drawable_data[drawable] = return_data

    def get_drawable_data(self, drawable):
        return self._drawable_data.get(drawable, {})


class Drawable(abc.ABC):
    """Interface for drawable objects."""
    @abc.abstractmethod
    def render(self, draw_context, view_matrix, **kwargs):
        """Render the object under action of a 3✕3 camera matrix, with render
        context variables supplied as kwargs."""
        raise NotImplementedError()


class Compound(Drawable):
    def __init__(self, gs):
        super().__init__()
        self._gs = gs

    def render(self, view_matrix):
        for g in self._gs:
            g.render(view_matrix)


class SimplePolygonFilled(Drawable):
    """A simple polygon filled with uniform colour."""
    def __init__(self, poly_verts, colour=None, xform=None):
        # FIXME: duplicating vertices is wasteful for large polygons. I should
        # use an index array instead.
        self._tri_verts = geom.triangulate_simple_polygon_ogl(poly_verts)
        self.colour = colour or (0.5, 0.5, 0.5)
        self.xform = xform

    # FIXME: turn these into property setters
    @property
    def colour(self):
        """The RGBA colour of the shape (RGB values will be converted to RGBA
        automatically)."""
        return self._colour

    @colour.setter
    def _set_colour(self, colour):
        colour = np.asarray(colour, dtype='float32')
        if colour.shape == (3, ):
            colour = np.concatenate((colour, (1.0, )))
        assert colour.shape == (4, )
        assert np.all(colour >= 0.0)
        assert np.all(colour <= 1.0)
        self._colour = colour

    @property
    def xform(self):
        return self.xform

    @xform.setter
    def _set_xform(self, xform):
        assert xform.shape == (3, 3), "expect 3x3 transformation matrix"
        self._xform = xform.astype('float32')

    def context_data(self, draw_context):
        vbo = draw_context.mgl_context.buffer(self._tri_verts)
        vao = draw_context.mgl_context.vertex_array(
            draw_context.programs.render_tris_unif_colour_2d,
            [(vbo, '2f', 'in_vert')])

        return {
            'vao': vao,
        }

    def render(self, draw_context, view_xform, vao):
        vao.program['u_color'] = self.colour
        vao.program['u_xform'] = view_xform @ self.model_xform
        vao.render(moderngl.TRIANGLES)


class XformBuilder:
    """Fluent API for constructing a 3✕3 rigid transform matrices."""
    def __init__(self, xform=None):
        self.xform = xform if xform is not None else np.eye(3)

    def translate(self, newx, newy):
        translation_matrix = np.asarray([[
            [1.0, 0.0, newx],
            [0.0, 1.0, newy],
            [0.0, 0.0, 1.0],
        ]])
        self.xform = self.xform @ translation_matrix
        return self

    def rotate(self, new):
        """Rotate counterclockwise by some amount"""
        rotation_matrix = np.asarray([
            [np.cos(new), -np.sin(new), 0.0],
            [np.sin(new), np.cos(new), 0.0],
            [0.0, 0.0, 1.0],
        ])
        self.xform = self.xform @ rotation_matrix
        return self

    def scale(self, newx, newy):
        scale_matrix = np.asarray([
            [newx, 0.0, 0.0],
            [0.0, newy, 0.0],
            [0.0, 0.0, 1.0],
        ],
                                  dtype='float32')
        self.xform = self.xform @ scale_matrix
        return self


def ego_cam_matrix(centre, newpos, rotation, scale):
    return XformBuilder() \
        .scale(scale[0], scale[1]) \
        .translate(newpos[0], newpos[1]) \
        .rotate(-rotation) \
        .translate(-centre[0], -centre[1], 0)


def make_circle(radius=10, res=30):
    points = []
    for i in range(res):
        ang = 2 * math.pi * i / res
        points.append((math.cos(ang) * radius, math.sin(ang) * radius))
    return SimplePolygonFilled(points)


def make_rect(width=10, height=10):
    rad_h = height / 2
    rad_w = width / 2
    points = [(-rad_w, rad_h), (rad_w, rad_h), (rad_w, -rad_h),
              (-rad_w, -rad_h)]
    return SimplePolygonFilled(points)


def make_square(side_length=10):
    return make_rect(side_length, side_length)


class ViewerWindow(mglw.WindowConfig):
    gl_version = (3, 3)
    window_size = (384, 384)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.draw_context = DrawContext(self.moderngl_ctx)
        make_circle()

    def render(self, time, frametime):
        pass


def main():
    """Demo program to show use of the new renderer."""
    mglw.run_window_config(ViewerWindow)


if __name__ == '__main__':
    main()
