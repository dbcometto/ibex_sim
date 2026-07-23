"""Defines the world/terrain the vehicle operates over."""

from abc import ABC, abstractmethod

import numpy as np


class Terrain(ABC):
    """Base class for anything that answers 'what's the ground elevation
    at this (x, y)?' Also provides a generic finite-difference slope,
    which subclasses can override with something exact/cheaper.
    """

    @abstractmethod
    def elevation(self, x, y):
        """Ground z at world (x, y). x, y may be scalars or arrays."""
        raise NotImplementedError

    def gradient(self, x, y, eps=1e-3):
        """(dz/dx, dz/dy) via central finite difference."""
        dzdx = (self.elevation(x + eps, y) - self.elevation(x - eps, y)) / (2 * eps)
        dzdy = (self.elevation(x, y + eps) - self.elevation(x, y - eps)) / (2 * eps)
        return dzdx, dzdy


class FlatTerrain(Terrain):
    """Constant-elevation ground plane.

    Also exposes a fake heightmap/resolution/origin -- a constant-value
    array of the given shape -- purely so this satisfies the same
    interface HeightmapTerrain gives the visualizer (which expects
    terrain.heightmap/.resolution/.origin to draw a background). The
    real elevation()/gradient() logic below never touches this array;
    it's display-only scaffolding, not used for any actual physics.
    """

    def __init__(self, elevation=0.0, shape=(256, 256), resolution=1.0, origin=(0.0, 0.0)):
        self._elevation = elevation
        self.resolution = resolution
        self.origin = np.array(origin, dtype=float)
        self.heightmap = np.full(shape, elevation, dtype=float)

    def elevation(self, x, y):
        x = np.asarray(x, dtype=float)
        return np.full_like(x, self._elevation) if x.shape else self._elevation

    def gradient(self, x, y, eps=1e-3):
        return 0.0, 0.0  # exact, no need to finite-difference a constant


class ParametricTerrain(Terrain):
    """Wraps an arbitrary z = func(x, y), e.g. a hill, ramp, or sine field."""

    def __init__(self, func):
        self.func = func

    def elevation(self, x, y):
        return self.func(x, y)


class HeightmapTerrain(Terrain):
    """Terrain from a 2D array of elevations, with bilinear interpolation
    between grid cells. Use from_image() to build one from a grayscale
    image instead of a raw array.
    """

    def __init__(self, heightmap, resolution, origin=(0.0, 0.0)):
        """
        heightmap: 2D array, shape (rows, cols); heightmap[row, col] is
            the elevation at grid cell (row, col).
        resolution: world units per grid cell (e.g. meters/pixel).
        origin: world (x, y) of heightmap[0, 0].
        """
        self.heightmap = np.asarray(heightmap, dtype=float)
        self.resolution = resolution
        self.origin = np.array(origin, dtype=float)

    @classmethod
    def from_image(cls, path, resolution, origin=(0.0, 0.0), z_scale=1.0, z_offset=0.0):
        """Grayscale image -> heightmap: pixel intensity [0, 1] * z_scale + z_offset.
        z_scale/z_offset are yours to set; raw pixel intensity has no
        inherent physical units.
        """
        from PIL import Image
        img = Image.open(path).convert("L")
        heightmap = np.asarray(img, dtype=float) / 255.0 * z_scale + z_offset
        return cls(heightmap, resolution, origin)

    def elevation(self, x, y):
        rows, cols = self.heightmap.shape
        col = (np.asarray(x, dtype=float) - self.origin[0]) / self.resolution
        row = (np.asarray(y, dtype=float) - self.origin[1]) / self.resolution

        row = np.clip(row, 0, rows - 1 - 1e-9)
        col = np.clip(col, 0, cols - 1 - 1e-9)

        r0 = np.floor(row).astype(int)
        c0 = np.floor(col).astype(int)
        r1 = np.minimum(r0 + 1, rows - 1)
        c1 = np.minimum(c0 + 1, cols - 1)
        fr = row - r0
        fc = col - c0

        z00, z01 = self.heightmap[r0, c0], self.heightmap[r0, c1]
        z10, z11 = self.heightmap[r1, c0], self.heightmap[r1, c1]
        return (z00 * (1 - fc) + z01 * fc) * (1 - fr) + (z10 * (1 - fc) + z11 * fc) * fr

    def gradient(self, x, y, eps=None):
        """Same finite-difference approach as the base class, but with
        eps scaled to the grid resolution instead of a fixed 1e-3. That
        fixed value samples both points inside the same bilinear cell,
        which gives an exact in-cell slope but jumps discontinuously at
        cell boundaries (bilinear interpolation is continuous in value,
        not in derivative). Using eps ~ resolution straddles the
        boundary instead, smoothing the transition as x, y cross cells.
        """
        if eps is None:
            eps = self.resolution
        return super().gradient(x, y, eps=eps)


def generate_heightmap_image(path, shape=(256, 256), num_hills=8, seed=None):
    """Synthesize a smooth random terrain (a sum of randomly-placed
    Gaussian 'hills', normalized to [0, 1]) and save it as a grayscale
    PNG suitable for HeightmapTerrain.from_image(). This is a quick way
    to get *some* varied terrain to test against, not a realistic model
    of any real landscape.
    """
    from PIL import Image

    rng = np.random.default_rng(seed)
    rows, cols = shape
    yy, xx = np.mgrid[0:rows, 0:cols]
    heightmap = np.zeros(shape, dtype=float)

    for _ in range(num_hills):
        cx = rng.uniform(0, cols)
        cy = rng.uniform(0, rows)
        sigma = rng.uniform(min(shape) * 0.05, min(shape) * 0.2)
        amplitude = rng.uniform(0.3, 1.0)
        heightmap += amplitude * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))

    heightmap -= heightmap.min()
    heightmap /= heightmap.max()

    Image.fromarray((heightmap * 255).astype(np.uint8), mode="L").save(path)
    return heightmap


if __name__ == "__main__":
    heightmap = generate_heightmap_image("sample_terrain.png", shape=(256, 256), seed=0)

    terrain = HeightmapTerrain.from_image(
        "sample_terrain.png", resolution=1.0, z_scale=20.0
    )
    print("Sample terrain saved to sample_terrain.png")
    print("Elevation at (0, 0):", terrain.elevation(0.0, 0.0))
    print("Elevation at (128, 128):", terrain.elevation(128.0, 128.0))
    print("Gradient at (128, 128):", terrain.gradient(128.0, 128.0))