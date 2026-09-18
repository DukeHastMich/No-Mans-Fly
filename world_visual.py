#!/usr/bin/env python3
"""Shared physical visual-surface model for No Man's Fly.

This module is deliberately below both sensory and observer presentation layers.
A habitat body has one deterministic world-space spectral surface.  Yar samples that
surface through compound-eye rays and opsins; the human main viewer samples the same
surface and converts the resulting spectrum to display RGB.  The Life pane remains a
separate analytical visualization.
"""
from __future__ import annotations

from functools import lru_cache
import threading
import hashlib
import math
import world_assets

import numpy as np

# Keep this grid identical to the compound-eye spectral integration grid.
SPECTRAL_WAVELENGTH_NM = np.asarray(
    [330., 345., 355., 375., 405., 437., 478., 508., 550., 600.], dtype=np.float64
)
SOURCE_ANCHOR_WAVELENGTH_NM = np.asarray([345., 375., 437., 478., 508.], dtype=np.float64)

# A compact equirectangular material atlas is generated once per deterministic body.
# It stores spectral *reflectance* only.  Illumination remains view-independent and is
# applied at each physical surface hit, so both observer and fly reuse the expensive
# terrain/material work without sharing a camera projection.
_SURFACE_ATLAS_W = 128
_SURFACE_ATLAS_H = 64
_SURFACE_ATLAS_MAX = 48
_CACHE_LOCK = threading.RLock()


def _stable_digest(text: str) -> bytes:
    return hashlib.sha256(str(text).encode("utf-8")).digest()


def body_render_kind(body_id: str) -> str:
    """Deterministic visual material family; physics still treats both as habitats."""
    if world_assets.available():return world_assets.template_for(body_id)
    return "moon" if _stable_digest(body_id)[2] < 72 else "planet"


def _gaussian(mu: float, sigma: float) -> np.ndarray:
    w = SPECTRAL_WAVELENGTH_NM
    return np.exp(-0.5 * np.square((w - float(mu)) / max(float(sigma), 1e-9)))


# Broad RGB-ish reflectance bases plus a small UV floor.  These are material spectra,
# not fly receptor curves and not display primaries.  The same spectra are consumed by
# both sensory and human-render paths.
_R_BASIS = _gaussian(610., 70.) + 0.10 * _gaussian(520., 95.)
_G_BASIS = _gaussian(535., 55.) + 0.08 * _gaussian(430., 90.)
_B_BASIS = _gaussian(450., 48.) + 0.18 * _gaussian(355., 38.)
_UV_FLOOR = 0.06 * _gaussian(355., 42.)
_BASIS_NORM = np.maximum(_R_BASIS + _G_BASIS + _B_BASIS, 1e-12)


def _rgb_to_reflectance(rgb) -> np.ndarray:
    r, g, b = np.clip(np.asarray(rgb, dtype=np.float64).reshape(3) / 255.0, 0.0, 1.0)
    # Divide by the summed basis so neutral RGB remains approximately spectrally flat.
    spec = (r * _R_BASIS + g * _G_BASIS + b * _B_BASIS) / _BASIS_NORM
    spec = 0.035 + 0.90 * spec + _UV_FLOOR * (0.25 + 0.75 * b)
    return np.clip(spec, 0.015, 1.0)


def _body_palette_rgb(body_id: str, kind: str) -> np.ndarray:
    d = _stable_digest(body_id)
    if kind == "moon":
        return np.asarray([176., 181., 187.], dtype=np.float64) * (0.83 + 0.17 * (d[10] / 255.0))
    palettes = np.asarray([
        [68., 122., 166.], [99., 146., 91.], [176., 111., 70.], [140., 111., 174.],
        [190., 158., 88.], [92., 157., 151.], [156., 91., 94.],
    ], dtype=np.float64)
    return palettes[d[11] % len(palettes)]


def _material_reflectance_direct(body_id: str, normals: np.ndarray, kind: str) -> np.ndarray:
    """Generate deterministic world-space spectral material for normalized normals."""
    n = np.array(normals, dtype=np.float64, copy=True).reshape((-1, 3))
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    x, y, z = n[:, 0], n[:, 1], n[:, 2]
    d = _stable_digest(body_id)
    phase = (int.from_bytes(d[4:8], "little") / 2**32) * 2.0 * math.pi
    f1 = 2.2 + (d[8] / 255.0) * 4.0
    f2 = 2.0 + (d[9] / 255.0) * 5.0
    tex = 0.84 + 0.16 * np.sin(math.pi * f1 * (x + 0.31*z) + 0.7*np.sin(math.pi*f2*(y - 0.19*x) + phase) + phase)
    base = _rgb_to_reflectance(_body_palette_rgb(body_id, kind))
    if kind == "moon":
        crater = (0.88 + 0.12*np.sin((x*7. + y*11. + z*3.)*math.pi + phase)
                  * np.sin((x*13. - y*5. + z*9.)*math.pi - phase))
        factor = tex * crater
    else:
        band = 0.84 + 0.16*np.sin((z*(3. + d[12] % 5))*math.pi + phase)
        continent = 0.90 + 0.10*np.sin((x*4.3 + y*2.7 - z*3.6)*math.pi + phase*0.61)
        polar = 1.0 + 0.12*np.clip((np.abs(z) - 0.62)/0.38, 0.0, 1.0)
        factor = tex * band * continent * polar
    return np.clip(factor[:, None] * base[None, :], 0.005, 1.0)


def _build_surface_atlas(body_id: str, kind: str) -> np.ndarray:
    lon = np.linspace(-math.pi, math.pi, _SURFACE_ATLAS_W, endpoint=False, dtype=np.float64)
    lat = np.linspace(math.pi*0.5, -math.pi*0.5, _SURFACE_ATLAS_H, dtype=np.float64)
    cl = np.cos(lat)[:, None]
    normals = np.stack((
        cl*np.cos(lon)[None, :],
        cl*np.sin(lon)[None, :],
        np.broadcast_to(np.sin(lat)[:, None], (_SURFACE_ATLAS_H, _SURFACE_ATLAS_W)),
    ), axis=-1).reshape((-1, 3))
    return _material_reflectance_direct(body_id, normals, kind).reshape((_SURFACE_ATLAS_H, _SURFACE_ATLAS_W, -1)).astype(np.float32)


@lru_cache(maxsize=_SURFACE_ATLAS_MAX)
def _surface_atlas(body_id: str, kind: str) -> np.ndarray:
    if world_assets.available():return world_assets.spectrum(world_assets.template_for(body_id))
    atlas = _build_surface_atlas(body_id, kind)
    atlas.setflags(write=False)
    return atlas


def surface_reflectance_spectrum(body_id: str, normals: np.ndarray, kind: str | None = None) -> np.ndarray:
    """Bilinearly sample one body's cached canonical spectral material atlas."""
    n = np.array(normals, dtype=np.float64, copy=True).reshape((-1, 3))
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    kind = str(kind or body_render_kind(body_id))
    atlas = _surface_atlas(str(body_id), kind)
    h, w = atlas.shape[:2]
    lon = np.arctan2(n[:, 1], n[:, 0])
    lat = np.arcsin(np.clip(n[:, 2], -1.0, 1.0))
    width = w-1 if world_assets.available() else w
    u = ((lon + math.pi)/(2.0*math.pi) % 1.0) * width
    v = np.clip((math.pi*0.5 - lat)/math.pi * (h - 1), 0.0, h - 1.0)
    x0 = np.floor(u).astype(np.int64) % w; x1 = (x0 + 1) % w
    y0 = np.floor(v).astype(np.int64); y1 = np.minimum(h - 1, y0 + 1)
    tx = (u - np.floor(u))[:, None]; ty = (v - np.floor(v))[:, None]
    a = atlas[y0, x0]*(1.-tx) + atlas[y0, x1]*tx
    b = atlas[y1, x0]*(1.-tx) + atlas[y1, x1]*tx
    return (a*(1.-ty) + b*ty).astype(np.float64)


def _blackbody_relative(temperature: float | np.ndarray, wavelengths_nm: np.ndarray = SPECTRAL_WAVELENGTH_NM) -> np.ndarray:
    t = np.asarray(temperature, dtype=np.float64).reshape((-1, 1))
    lam = np.asarray(wavelengths_nm, dtype=np.float64).reshape((1, -1))*1e-9
    c2 = 1.438776877e-2
    x = np.clip(c2/(lam*np.maximum(t, 1.0)), 1e-6, 700.)
    b = 1.0/(np.power(lam, 5)*np.expm1(x))
    b /= np.maximum(np.max(b, axis=1, keepdims=True), 1e-300)
    return b


# Until explicit stellar illumination is added to HabitatBody, use one stable 5800 K
# world-space key light.  It is shared by fly and human rendering, so there is no
# observer-only lighting.  Ambient term represents unresolved sky/star illumination.
_WORLD_LIGHT_DIR = np.asarray([-0.45, -0.32, 0.83], dtype=np.float64)
_WORLD_LIGHT_DIR /= np.linalg.norm(_WORLD_LIGHT_DIR)
_WORLD_ILLUMINANT = _blackbody_relative(5800.0)[0]


def surface_spectral_radiance(body_id: str, normals: np.ndarray, albedo: float,
                              kind: str | None = None) -> np.ndarray:
    """Shared reflected spectral radiance for physical sphere-surface hits."""
    n = np.array(normals, dtype=np.float64, copy=True).reshape((-1, 3))
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    refl = surface_reflectance_spectrum(body_id, n, kind)
    lambert = np.clip(n @ _WORLD_LIGHT_DIR, 0.0, 1.0)
    illumination = 0.16 + 0.84*lambert
    # Habitat albedo is a scalar material parameter.  Preserve it as the global
    # reflectance scale while retaining a small unresolved diffuse floor.
    a = float(np.clip(albedo, 0.0, 1.0))
    albedo_scale = 0.12 + 0.88*a
    return np.clip(refl * _WORLD_ILLUMINANT[None, :] * (illumination*albedo_scale)[:, None], 0.0, None)


def source_spectrum(source) -> np.ndarray:
    """Same spectral source model used for retinal and human point-source display."""
    sw = getattr(source, "spectral_weights", None)
    if sw is not None:
        a = np.asarray(sw, dtype=np.float64).reshape(-1)
        if len(a) == 5 and np.max(a) > 0:
            dense = np.interp(SPECTRAL_WAVELENGTH_NM, SOURCE_ANCHOR_WAVELENGTH_NM, a,
                              left=.5*a[0], right=.35*a[-1])
            return dense/max(float(np.max(dense)), 1e-12)
    return _blackbody_relative(float(getattr(source, "temperature", 5800.0)))[0]


def spectral_to_rgb(spectrum: np.ndarray, *, tone_map: bool = True) -> np.ndarray:
    """Convert the shared 10-band spectrum to a human-display RGB approximation."""
    s = np.asarray(spectrum, dtype=np.float64)
    one = (s.ndim == 1)
    if one:s = s.reshape((1, -1))
    if s.shape[-1] != len(SPECTRAL_WAVELENGTH_NM):
        raise ValueError("spectrum must use shared 10-band wavelength grid")
    # Broad human-display matching curves.  This is a presentation conversion only;
    # Yar's receptors use their own measured opsin curves against the same spectrum.
    wr = _gaussian(610., 58.) + 0.18*_gaussian(545., 42.)
    wg = _gaussian(545., 47.) + 0.10*_gaussian(500., 35.)
    wb = _gaussian(445., 42.) + 0.16*_gaussian(380., 38.)
    weights = np.vstack((wr/wr.sum(), wg/wg.sum(), wb/wb.sum())).T
    rgb = s @ weights
    if tone_map:
        rgb = 1.0 - np.exp(-3.4*np.clip(rgb, 0.0, None))
    else:
        rgb = np.clip(rgb, 0.0, 1.0)
    rgb = np.power(np.clip(rgb, 0.0, 1.0), 1.0/2.2)
    out = np.clip(np.rint(rgb*255.0), 0, 255).astype(np.uint8)
    return out[0] if one else out


def cache_info() -> dict[str, int]:
    return {"surface_atlases": _surface_atlas.cache_info().currsize, "atlas_width": 512 if world_assets.available() else _SURFACE_ATLAS_W,
            "atlas_height": 256 if world_assets.available() else _SURFACE_ATLAS_H, "spectral_bands": len(SPECTRAL_WAVELENGTH_NM)}


@lru_cache(maxsize=96)
def _projected_atlas(body_id, kind, shape, weights_bytes):
    weights = np.frombuffer(weights_bytes, dtype=np.float64).reshape(shape)
    source=world_assets.spectrum(body_id) if world_assets.available() else _surface_atlas(body_id,kind)
    projected = source.astype(np.float64) @ (_WORLD_ILLUMINANT[:, None] * weights)
    projected.setflags(write=False)
    return projected


def surface_channels(body_id, normals, albedo, weights, kind=None):
    n = np.array(normals, dtype=np.float64, copy=True).reshape(-1, 3)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    weights = np.ascontiguousarray(weights, dtype=np.float64)
    key=world_assets.template_for(body_id) if world_assets.available() else str(body_id)
    projection_kind=key if world_assets.available() else str(kind or body_render_kind(body_id))
    atlas = _projected_atlas(key, projection_kind, weights.shape, weights.tobytes())
    h, w = atlas.shape[:2]
    width=w-1 if world_assets.available() else w
    u = ((np.arctan2(n[:, 1], n[:, 0]) + math.pi) / (2*math.pi) % 1.0) * width
    v = np.clip((math.pi*.5-np.arcsin(np.clip(n[:, 2],-1,1)))/math.pi*(h-1),0,h-1)
    x = np.floor(u).astype(int); y = np.floor(v).astype(int)
    tx = (u-x)[:, None]; ty = (v-y)[:, None]
    x1 = (x+1)%w; y1 = np.minimum(y+1,h-1)
    a = atlas[y,x]*(1-tx)+atlas[y,x1]*tx
    b = atlas[y1,x]*(1-tx)+atlas[y1,x1]*tx
    illumination = .16+.84*np.clip(n @ _WORLD_LIGHT_DIR,0,1)
    return (a*(1-ty)+b*ty) * (illumination*(.12+.88*np.clip(albedo,0,1)))[:,None]


_RGB_WEIGHTS = np.column_stack((
    _gaussian(610.,58.)+.18*_gaussian(545.,42.),
    _gaussian(545.,47.)+.10*_gaussian(500.,35.),
    _gaussian(445.,42.)+.16*_gaussian(380.,38.)))
_RGB_WEIGHTS /= _RGB_WEIGHTS.sum(axis=0)


def surface_rgb(body_id, normals, albedo, kind=None):
    rgb = surface_channels(body_id,normals,albedo,_RGB_WEIGHTS,kind)
    rgb = np.power(1-np.exp(-3.4*np.clip(rgb,0,None)),1/2.2)
    return np.clip(np.rint(rgb*255),0,255).astype(np.uint8)

