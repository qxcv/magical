#!/usr/bin/env python3
"""Demo program to evaluate the features I'll need in my Pyglet -> ModernGL
port."""

import moderngl
import moderngl_window as mglw
import numpy as np
import triangle

from magical import geom


class MainWindow(mglw.WindowConfig):
    # this class is from ModernGL examples
    gl_version = (3, 3)
    title = "ModernGL Example"
    window_size = (1280, 720)
    aspect_ratio = 1.0
    resizable = True
    samples = 4

    def __init__(self, **kwargs):
        # __init__ here is copied from the ModernGL triangle example
        super().__init__(**kwargs)

        self.prog = self.ctx.program(
            # vertex shader does a simple transform + draw
            vertex_shader='''
                #version 330
                in vec2 in_vert;
                uniform mat3 u_xform;
                void main() {
                    vec3 pos_pad = u_xform * vec3(in_vert, 1.0);
                    gl_Position = vec4(pos_pad.x, pos_pad.y, 0.0, 1.0);
                }
            ''',
            # fragment shader simply copies uniform colour across
            fragment_shader='''
                #version 330
                uniform vec3 u_color;
                out vec4 f_color;
                void main() {
                    f_color = vec4(u_color, 1.0);
                }
            ''')

        # shape_verts = geom.rect_verts(1.5, 0.8)
        # shape_verts = geom.compute_star_verts(5, 0.8, 0.4)
        # shape_verts = geom.compute_circle_verts(30, 0.2)
        shape_verts = geom.compute_regular_poly_verts(5, 0.6)
        tri_verts = geom.triangulate_simple_polygon_ogl(shape_verts)
        self.tri_verts = tri_verts.astype('f4')
        # Point coordinates only
        # vertices = np.array(
        #     [
        #         # x, y
        #         0.0,
        #         0.8,
        #         -0.6,
        #         -0.8,
        #         0.6,
        #         -0.8,
        #     ],
        #     dtype='f4')

        self.vbo = self.ctx.buffer(self.tri_verts)

        # We control the 'in_vert' and `in_color' variables
        self.vao = self.ctx.vertex_array(
            self.prog,
            [
                # Map in_vert to the first 2 floats
                (self.vbo, '2f', 'in_vert'),
            ])

    def render(self, time: float, frame_time: float):
        self.ctx.clear(1.0, 1.0, 1.0)
        self.prog['u_color'] = ((np.sin(time / 2) + 1) / 2,
                                (np.sin(time / 3) + 1) / 2,
                                (np.sin(time / 5) + 1) / 2)
        rot_mat = np.eye(3)
        self.prog['u_xform'] = tuple(rot_mat.flatten())
        # self.vao.render(moderngl.LINE_LOOP)
        # self.vao.render(moderngl.POINTS)
        # self.vao.render(moderngl.TRIANGLE_STRIP)
        self.vao.render()

        # for idx, inds in enumerate(self.tri_inds):
        #     self.prog['u_color'] = (idx / len(self.tri_inds), 1.0 - idx / len(self.tri_inds), 0.5)
        #     vao = self.ctx.vertex_array(
        #         self.prog,
        #         [
        #             (self.ctx.buffer(self.tri_verts[inds].flatten()), '2f', 'in_vert'),
        #         ],
        #     )
        #     vao.render(moderngl.LINE_LOOP)

    @classmethod
    def run(cls):
        mglw.run_window_config(cls)


if __name__ == '__main__':
    MainWindow.run()
