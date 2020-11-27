from ctypes import byref

import pyglet
from pyglet import gl
from pyglet.image import ImageException
from pyglet.image.codecs import ImageEncodeException, get_encoders

from magical.pyglet_backport.image.imagebuffer import (  # noqa: F401
    Framebuffer, Renderbuffer, get_max_color_attachments)


class AbstractImage:
    """Abstract class representing an image.
    :Parameters:
        `width` : int
            Width of image
        `height` : int
            Height of image
        `anchor_x` : int
            X coordinate of anchor, relative to left edge of image data
        `anchor_y` : int
            Y coordinate of anchor, relative to bottom edge of image data
    """
    anchor_x = 0
    anchor_y = 0

    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __repr__(self):
        return "{}(size={}x{})".format(self.__class__.__name__, self.width,
                                       self.height)

    def get_image_data(self):
        """Get an ImageData view of this image.

        Changes to the returned instance may or may not be reflected in this
        image.
        :rtype: :py:class:`~pyglet.image.ImageData`
        .. versionadded:: 1.1
        """
        raise ImageException('Cannot retrieve image data for %r' % self)

    def get_texture(self, rectangle=False):
        """A :py:class:`~pyglet.image.Texture` view of this image.
        By default, textures are created with dimensions that are powers of
        two. Smaller images will return a
        :py:class:`~pyglet.image.TextureRegion` that covers just the image
        portion of the larger texture. This restriction is required on older
        video cards, and for compressed textures, or where texture repeat modes
        will be used, or where mipmapping is desired. If the `rectangle`
        parameter is ``True``, this restriction is ignored and a texture the
        size of the image may be created if the driver supports the
        ``GL_ARB_texture_rectangle`` or ``GL_NV_texture_rectangle`` extensions.
        If the extensions are not present, the image already is a texture, or
        the image has power 2 dimensions, the `rectangle` parameter is ignored.
        Examine `Texture.target` to determine if the returned texture is a
        rectangle (``GL_TEXTURE_RECTANGLE``) or not (``GL_TEXTURE_2D``).
        Changes to the returned instance may or may not be reflected in this
        image.
        :Parameters:
            `rectangle` : bool
                True if the texture can be created as a rectangle.
                .. versionadded:: 1.1.4.
        :rtype: :py:class:`~pyglet.image.Texture`
        .. versionadded:: 1.1
        """
        raise ImageException('Cannot retrieve texture for %r' % self)

    def get_mipmapped_texture(self):
        """Retrieve a :py:class:`~pyglet.image.Texture` instance with all
        mipmap levels filled in. Requires that image dimensions be powers of 2.
        :rtype: :py:class:`~pyglet.image.Texture`
        .. versionadded:: 1.1
        """
        raise ImageException('Cannot retrieve mipmapped texture for %r' % self)

    def get_region(self, x, y, width, height):
        """Retrieve a rectangular region of this image.
        :Parameters:
            `x` : int
                Left edge of region.
            `y` : int
                Bottom edge of region.
            `width` : int
                Width of region.
            `height` : int
                Height of region.
        :rtype: AbstractImage
        """
        raise ImageException('Cannot get region for %r' % self)

    def save(self, filename=None, file=None, encoder=None):
        """Save this image to a file.
        :Parameters:
            `filename` : str
                Used to set the image file format, and to open the output file
                if `file` is unspecified.
            `file` : file-like object or None
                File to write image data to.
            `encoder` : ImageEncoder or None
                If unspecified, all encoders matching the filename extension
                are tried.  If all fail, the exception from the first one
                attempted is raised.
        """
        if not file:
            file = open(filename, 'wb')

        if encoder:
            encoder.encode(self, file, filename)
        else:
            first_exception = None
            for encoder in get_encoders(filename):
                try:
                    encoder.encode(self, file, filename)
                    return
                except ImageEncodeException as e:
                    first_exception = first_exception or e
                    file.seek(0)

            if not first_exception:
                raise ImageEncodeException('No image encoders are available')
            raise first_exception

    def blit(self, x, y, z=0):
        """Draw this image to the active framebuffers.

        The image will be drawn with the lower-left corner at
        (``x -`` `anchor_x`, ``y -`` `anchor_y`, ``z``).
        """
        raise ImageException('Cannot blit %r.' % self)

    def blit_into(self, source, x, y, z):
        """Draw `source` on this image.
        `source` will be copied into this image such that its anchor point
        is aligned with the `x` and `y` parameters.  If this image is a 3D
        texture, the `z` coordinate gives the image slice to copy into.

        Note that if `source` is larger than this image (or the positioning
        would cause the copy to go out of bounds) then you must pass a
        region of `source` to this method, typically using get_region().
        """
        raise ImageException('Cannot blit images onto %r.' % self)

    def blit_to_texture(self, target, level, x, y, z=0):
        """Draw this image on the currently bound texture at `target`.

        This image is copied into the texture such that this image's anchor
        point is aligned with the given `x` and `y` coordinates of the
        destination texture.  If the currently bound texture is a 3D texture,
        the `z` coordinate gives the image slice to blit into.
        """
        raise ImageException('Cannot blit %r to a texture.' % self)


class Texture(AbstractImage):
    """An image loaded into video memory that can be efficiently drawn
    to the framebuffer.
    Typically you will get an instance of Texture by accessing the `texture`
    member of any other AbstractImage.
    :Parameters:
        `region_class` : class (subclass of TextureRegion)
            Class to use when constructing regions of this texture.
        `tex_coords` : tuple
            12-tuple of float, named (u1, v1, r1, u2, v2, r2, ...).  u, v, r
            give the 3D texture coordinates for vertices 1-4.  The vertices
            are specified in the order bottom-left, bottom-right, top-right
            and top-left.
        `target` : int
            The GL texture target (e.g., ``GL_TEXTURE_2D``).
        `level` : int
            The mipmap level of this texture.
    """

    region_class = None  # Set to TextureRegion after it's defined
    tex_coords = (0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0)
    tex_coords_order = (0, 1, 2, 3)
    level = 0
    images = 1
    x = y = z = 0
    default_min_filter = gl.GL_LINEAR
    default_mag_filter = gl.GL_LINEAR

    def __init__(self, width, height, target, tex_id):
        super(Texture, self).__init__(width, height)
        self.target = target
        self.id = tex_id
        self._context = pyglet.gl.current_context

    def __del__(self):
        try:
            self._context.delete_texture(self.id)
        except:  # noqa: E722
            pass

    @classmethod
    def create(cls,
               width,
               height,
               target=gl.GL_TEXTURE_2D,
               internalformat=gl.GL_RGBA,
               min_filter=None,
               mag_filter=None):
        """Create a Texture
        Create a Texture with the specified dimentions, target and format.
        On return, the texture will be bound.
        :Parameters:
            `width` : int
                Width of texture in pixels.
            `height` : int
                Height of texture in pixels.
            `target` : int
                GL constant giving texture target to use, typically
                ``GL_TEXTURE_2D``.
            `internalformat` : int
                GL constant giving internal format of texture; for example,
                ``GL_RGBA``. If ``None``, the texture will be created but not
                initialized.
            `min_filter` : int
                The minifaction filter used for this texture, commonly
                ``GL_LINEAR`` or ``GL_NEAREST``
            `mag_filter` : int
                The magnification filter used for this texture, commonly
                ``GL_LINEAR`` or ``GL_NEAREST``
        :rtype: :py:class:`~pyglet.image.Texture`
        """
        min_filter = min_filter or cls.default_min_filter
        mag_filter = mag_filter or cls.default_mag_filter

        tex_id = gl.GLuint()
        gl.glGenTextures(1, byref(tex_id))
        gl.glBindTexture(target, tex_id.value)
        if target != gl.GL_TEXTURE_2D_MULTISAMPLE:
            gl.glTexParameteri(target, gl.GL_TEXTURE_MIN_FILTER, min_filter)
            gl.glTexParameteri(target, gl.GL_TEXTURE_MAG_FILTER, mag_filter)

        if internalformat is not None:
            blank = (gl.GLubyte * (width * height * 4))()
            gl.glTexImage2D(target, 0, internalformat, width, height, 0,
                            gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, blank)
            gl.glFlush()

        texture = cls(width, height, target, tex_id.value)
        texture.min_filter = min_filter
        texture.mag_filter = mag_filter
        if target is gl.GL_TEXTURE_RECTANGLE:
            texture.tex_coords = (0, 0, 0, width, 0, 0, width, height, 0, 0,
                                  height, 0)
        else:
            texture.tex_coords = cls.tex_coords

        return texture

    def get_image_data(self, z=0, fmt='RGBA', gl_format=gl.GL_RGBA):
        """Get the image data of this texture.
        Changes to the returned instance will not be reflected in this
        texture.
        :Parameters:
            `z` : int
                For 3D textures, the image slice to retrieve.
        :rtype: :py:class:`~pyglet.image.ImageData`
        """
        gl.glBindTexture(self.target, self.id)

        # # Always extract complete RGBA data.  Could check internalformat
        # # to only extract used channels. XXX
        # fmt = 'RGBA'
        # gl_format = gl.GL_RGBA

        gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
        buffer = (gl.GLubyte *
                  (self.width * self.height * self.images * len(fmt)))()
        gl.glGetTexImage(self.target, self.level, gl_format,
                         gl.GL_UNSIGNED_BYTE, buffer)

        data = pyglet.image.ImageData(self.width, self.height, fmt, buffer)
        if self.images > 1:
            data = data.get_region(0, z * self.height, self.width, self.height)
        return data

    def get_texture(self, rectangle=False):
        if rectangle and not self.target == gl.GL_TEXTURE_RECTANGLE:
            raise gl.ImageException(
                'Texture is not a rectangle, it must be created as a '
                'rectangle.')
        return self

    # no implementation of blit_to_texture yet (could use aux buffer)

    def blit(self, x, y, z=0, width=None, height=None):
        x1 = x - self.anchor_x
        y1 = y - self.anchor_y
        x2 = x1 + (width is None and self.width or width)
        y2 = y1 + (height is None and self.height or height)
        vertices = x1, y1, z, x2, y1, z, x2, y2, z, x1, y2, z

        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(self.target, self.id)

        pyglet.graphics.draw_indexed(4, gl.GL_TRIANGLES, [0, 1, 2, 0, 2, 3],
                                     ('v3f', vertices),
                                     ('t3f', self.tex_coords))

        gl.glBindTexture(self.target, 0)

    def blit_into(self, source, x, y, z):
        gl.glBindTexture(self.target, self.id)
        source.blit_to_texture(self.target, self.level, x, y, z)

    def get_region(self, x, y, width, height):
        return self.region_class(x, y, 0, width, height, self)

    def get_transform(self, flip_x=False, flip_y=False, rotate=0):
        """Create a copy of this image applying a simple transformation. The
        transformation is applied to the texture coordinates only;
        :py:meth:`~pyglet.image.ImageData.get_image_data` will return the
        untransformed data. The transformation is applied around the anchor
        point.
        :Parameters:
            `flip_x` : bool
                If True, the returned image will be flipped horizontally.
            `flip_y` : bool
                If True, the returned image will be flipped vertically.
            `rotate` : int
                Degrees of clockwise rotation of the returned image.  Only
                90-degree increments are supported.
        :rtype: :py:class:`~pyglet.image.TextureRegion`
        """
        transform = self.get_region(0, 0, self.width, self.height)
        bl, br, tr, tl = 0, 1, 2, 3
        transform.anchor_x = self.anchor_x
        transform.anchor_y = self.anchor_y
        if flip_x:
            bl, br, tl, tr = br, bl, tr, tl
            transform.anchor_x = self.width - self.anchor_x
        if flip_y:
            bl, br, tl, tr = tl, tr, bl, br
            transform.anchor_y = self.height - self.anchor_y
        rotate %= 360
        if rotate < 0:
            rotate += 360
        if rotate == 0:
            pass
        elif rotate == 90:
            bl, br, tr, tl = br, tr, tl, bl
            transform.anchor_x, transform.anchor_y \
                = transform.anchor_y, transform.width - transform.anchor_x
        elif rotate == 180:
            bl, br, tr, tl = tr, tl, bl, br
            transform.anchor_x = transform.width - transform.anchor_x
            transform.anchor_y = transform.height - transform.anchor_y
        elif rotate == 270:
            bl, br, tr, tl = tl, bl, br, tr
            transform.anchor_x, transform.anchor_y \
                = transform.height - transform.anchor_y, transform.anchor_x
        else:
            assert False, 'Only 90 degree rotations are supported.'
        if rotate in (90, 270):
            transform.width, transform.height \
                = transform.height, transform.width
        transform._set_tex_coords_order(bl, br, tr, tl)
        return transform

    def _set_tex_coords_order(self, bl, br, tr, tl):
        tex_coords = (self.tex_coords[:3], self.tex_coords[3:6],
                      self.tex_coords[6:9], self.tex_coords[9:])
        self.tex_coords = tex_coords[bl] + tex_coords[br] + tex_coords[
            tr] + tex_coords[tl]

        order = self.tex_coords_order
        self.tex_coords_order = (order[bl], order[br], order[tr], order[tl])

    def __repr__(self):
        return "{}(id={}, size={}x{})".format(self.__class__.__name__, self.id,
                                              self.width, self.height)
