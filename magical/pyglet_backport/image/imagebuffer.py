# This file was taken from Pyglet 2.0dev0 at
# 94bf013cc76312ed91cbeedd81f4bde1b9bb9273

# ----------------------------------------------------------------------------
# pyglet
# Copyright (c) 2006-2008 Alex Holkner
# Copyright (c) 2008-2019 pyglet contributors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#  * Neither the name of pyglet nor the names of its
#    contributors may be used to endorse or promote products
#    derived from this software without specific prior written
#    permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# ----------------------------------------------------------------------------
from pyglet import gl


def get_max_color_attachments():
    """Get the maximum allow Framebuffer Color attachements"""
    number = gl.GLint()
    gl.glGetIntegerv(gl.GL_MAX_COLOR_ATTACHMENTS, number)
    return number.value


class Renderbuffer:
    """OpenGL Renderbuffer Object"""
    def __init__(self, width, height, internal_format, samples=1):
        """Create an instance of a Renderbuffer object."""
        self._id = gl.GLuint()
        self._width = width
        self._height = height
        self._internal_format = internal_format

        gl.glGenRenderbuffers(1, self._id)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self._id)

        if samples > 1:
            gl.glRenderbufferStorageMultisample(gl.GL_RENDERBUFFER, samples,
                                                internal_format, width, height)
        else:
            gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, internal_format,
                                     width, height)

        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, 0)

    @property
    def id(self):  # noqa: A003
        return self._id.value

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    def bind(self):
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self._id)

    @staticmethod
    def unbind():
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, 0)

    def delete(self):
        gl.glDeleteRenderbuffers(1, self._id)

    def __del__(self):
        try:
            gl.glDeleteRenderbuffers(1, self._id)
            # Python interpreter is shutting down:
        except ImportError:
            pass

    def __repr__(self):
        return "{}(id={})".format(self.__class__.__name__, self._id.value)


class Framebuffer:
    """OpenGL Framebuffer Object"""

    _max_color_attachments = get_max_color_attachments()

    def __init__(self, target=gl.GL_FRAMEBUFFER):
        """Create an OpenGL Framebuffer object.

        :rtype: :py:class:`~pyglet.image.Framebuffer`

        .. versionadded:: 2.0
        """
        self._id = gl.GLuint()
        gl.glGenFramebuffers(1, self._id)
        self._attachment_types = 0
        self._width = 0
        self._height = 0

    @property
    def id(self):  # noqa: A003
        return self._id.value

    @property
    def width(self):
        """The width of the widest attachment."""
        return self._width

    @property
    def height(self):
        """The width of the widest attachment."""
        return self._height

    def bind(self, target=gl.GL_FRAMEBUFFER):
        gl.glBindFramebuffer(target, self._id)

    @staticmethod
    def unbind(target=gl.GL_FRAMEBUFFER):
        gl.glBindFramebuffer(target, 0)

    def clear(self):
        if self._attachment_types:
            self.bind()
            gl.glClear(self._attachment_types)
            self.unbind()

    def delete(self):
        gl.glDeleteFramebuffers(1, self._id)

    @property
    def is_complete(self):
        return gl.glCheckFramebufferStatus(
            gl.GL_FRAMEBUFFER) == gl.GL_FRAMEBUFFER_COMPLETE

    @staticmethod
    def get_status():
        states = {
            gl.GL_FRAMEBUFFER_UNSUPPORTED:
            "Framebuffer unsupported. Try another format.",
            gl.GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
            "Framebuffer incomplete attachment.",
            gl.GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
            "Framebuffer missing attachment.",
            gl.GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT:
            "Framebuffer unsupported dimension.",
            gl.GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT:
            "Framebuffer incomplete formats.",
            gl.GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
            "Framebuffer incomplete draw buffer.",
            gl.GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
            "Framebuffer incomplete read buffer.",
            gl.GL_FRAMEBUFFER_COMPLETE: "Framebuffer is complete.",
        }

        gl_status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)

        return states.get(gl_status, "Unknown error")

    def attach_texture(self, target, attachment, texture):
        """Attach a Texture to the Framebuffer

        :Parameters:
            `target` : int
                Specifies the framebuffer target. target must be
                gl.GL_DRAW_FRAMEBUFFER, gl.GL_READ_FRAMEBUFFER, or
                gl.GL_FRAMEBUFFER. gl.GL_FRAMEBUFFER is equivalent to
                gl.GL_DRAW_FRAMEBUFFER.
            `attachment` : int
                Specifies the attachment point of the framebuffer. attachment
                must be gl.GL_COLOR_ATTACHMENTi, gl.GL_DEPTH_ATTACHMENT,
                gl.GL_STENCIL_ATTACHMENT or gl.GL_DEPTH_STENCIL_ATTACHMENT.
            `texture` : pyglet.image.Texture
                Specifies the texture object to attach to the framebuffer
                attachment point named by attachment.
        """
        self.bind()
        gl.glFramebufferTexture2D(target, attachment, texture.target,
                                  texture.id, texture.level)
        # gl.glFramebufferTexture2D(target, attachment, texture.target,
        # texture.id, texture.level)
        self._attachment_types |= attachment
        self._width = max(texture.width, self._width)
        self._height = max(texture.height, self._height)
        self.unbind()

    def attach_texture_layer(self, target, attachment, texture, level, layer):
        """Attach a Texture layer to the Framebuffer

        :Parameters:
            `target` : int
                Specifies the framebuffer target. target must be
                gl.GL_DRAW_FRAMEBUFFER, gl.GL_READ_FRAMEBUFFER, or
                gl.GL_FRAMEBUFFER. gl.GL_FRAMEBUFFER is equivalent to
                gl.GL_DRAW_FRAMEBUFFER.
            `attachment` : int
                Specifies the attachment point of the framebuffer. attachment
                must be gl.GL_COLOR_ATTACHMENTi, gl.GL_DEPTH_ATTACHMENT,
                gl.GL_STENCIL_ATTACHMENT or gl.GL_DEPTH_STENCIL_ATTACHMENT.
            `texture` : pyglet.image.TextureArray
                Specifies the texture object to attach to the framebuffer
                attachment point named by attachment.
            `level` : int
                Specifies the mipmap level of texture to attach.
            `layer` : int
                Specifies the layer of texture to attach.
        """
        self.bind()
        gl.glFramebufferTextureLayer(target, attachment, texture.id, level,
                                     layer)
        self._attachment_types |= attachment
        self._width = max(texture.width, self._width)
        self._height = max(texture.height, self._height)
        self.unbind()

    def attach_renderbuffer(self, target, attachment, renderbuffer):
        """"Attach a Renderbuffer to the Framebuffer

        :Parameters:
            `target` : int
                Specifies the framebuffer target. target must be
                gl.GL_DRAW_FRAMEBUFFER, gl.GL_READ_FRAMEBUFFER, or
                gl.GL_FRAMEBUFFER. gl.GL_FRAMEBUFFER is equivalent to
                gl.GL_DRAW_FRAMEBUFFER.
            `attachment` : int
                Specifies the attachment point of the framebuffer. attachment
                must be gl.GL_COLOR_ATTACHMENTi, gl.GL_DEPTH_ATTACHMENT,
                gl.GL_STENCIL_ATTACHMENT or gl.GL_DEPTH_STENCIL_ATTACHMENT.
            `renderbuffer` : pyglet.image.Renderbuffer
                Specifies the Renderbuffer to attach to the framebuffer
                attachment point named by attachment.
        """
        self.bind()
        gl.glFramebufferRenderbuffer(target, attachment, gl.GL_RENDERBUFFER,
                                     renderbuffer.id)
        self._attachment_types |= attachment
        self._width = max(renderbuffer.width, self._width)
        self._height = max(renderbuffer.height, self._height)
        self.unbind()

    def __del__(self):
        try:
            gl.glDeleteFramebuffers(1, self._id)
        # Python interpreter is shutting down:
        except ImportError:
            pass

    def __repr__(self):
        return "{}(id={})".format(self.__class__.__name__, self._id.value)
