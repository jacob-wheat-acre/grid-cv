"""
Shared pytest fixtures for grid_cv tests.

Creates synthetic test images (plain NumPy arrays written to temp files)
so the test suite runs without any real drone data.
"""

from __future__ import annotations

import struct
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Synthetic image builders
# ---------------------------------------------------------------------------

def make_rgb_image(width: int = 1920, height: int = 1080, seed: int = 7) -> np.ndarray:
    """Return a BGR NumPy array that looks like a vaguely sky/ground scene."""
    rng = np.random.default_rng(seed)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Sky gradient (top half)
    sky_color = np.array([200, 160, 100], dtype=np.uint8)  # BGR blue-ish
    img[:height // 2] = sky_color
    # Ground (bottom half)
    ground_color = np.array([60, 100, 60], dtype=np.uint8)  # BGR greenish
    img[height // 2 :] = ground_color
    # Random noise to make it less trivial
    noise = rng.integers(0, 30, size=img.shape, dtype=np.uint8)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def make_thermal_image(width: int = 640, height: int = 512, seed: int = 13) -> np.ndarray:
    """Return a grayscale (3-channel) pseudo-thermal image."""
    rng = np.random.default_rng(seed)
    base = rng.integers(80, 200, size=(height, width), dtype=np.uint8)
    # Hot spots
    for _ in range(5):
        cx = int(rng.integers(50, width - 50))
        cy = int(rng.integers(50, height - 50))
        cv2.circle(base, (cx, cy), int(rng.integers(10, 40)), 255, -1)
    return cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# EXIF GPS injection helper
# ---------------------------------------------------------------------------

def _rational(num: int, denom: int = 1) -> tuple[int, int]:
    return (num, denom)


def inject_exif_gps(jpeg_bytes: bytes, lat: float, lon: float, alt_m: float) -> bytes:
    """
    Inject a minimal GPS EXIF IFD into a JPEG byte string.
    Returns the modified bytes (or original if piexif unavailable).
    """
    try:
        import io
        import piexif

        def _to_dms(deg: float) -> tuple[tuple, tuple, tuple]:
            deg = abs(deg)
            d = int(deg)
            m = int((deg - d) * 60)
            s = round(((deg - d) * 60 - m) * 60 * 100)
            return (d, 1), (m, 1), (s, 100)

        gps_ifd = {
            piexif.GPSIFD.GPSLatitudeRef:  b"N" if lat >= 0 else b"S",
            piexif.GPSIFD.GPSLatitude:     _to_dms(lat),
            piexif.GPSIFD.GPSLongitudeRef: b"E" if lon >= 0 else b"W",
            piexif.GPSIFD.GPSLongitude:    _to_dms(lon),
            piexif.GPSIFD.GPSAltitudeRef:  0,
            piexif.GPSIFD.GPSAltitude:     (int(alt_m * 100), 100),
        }
        exif_dict = {"GPS": gps_ifd}
        exif_bytes_data = piexif.dump(exif_dict)
        # piexif.insert requires a 3rd output argument; use BytesIO
        output = io.BytesIO()
        piexif.insert(exif_bytes_data, jpeg_bytes, output)
        return output.getvalue()
    except Exception:
        return jpeg_bytes


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def tmp_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("grid_cv_test")


@pytest.fixture(scope="session")
def synthetic_rgb_path(tmp_dir: Path) -> Path:
    """Write a 1920×1080 synthetic RGB JPEG with injected GPS."""
    img = make_rgb_image(1920, 1080)
    path = tmp_dir / "synthetic_rgb.jpg"
    cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    # Inject GPS: somewhere over Tennessee (utility corridor country)
    raw = path.read_bytes()
    raw = inject_exif_gps(raw, lat=35.9606, lon=-83.9207, alt_m=85.0)
    path.write_bytes(raw)
    return path


@pytest.fixture(scope="session")
def synthetic_thermal_path(tmp_dir: Path) -> Path:
    """Write a 640×512 pseudo-thermal PNG."""
    img = make_thermal_image(640, 512)
    path = tmp_dir / "synthetic_thermal.png"
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture(scope="session")
def synthetic_hires_path(tmp_dir: Path) -> Path:
    """Write a 4096×3072 (4K) synthetic image to stress-test SAHI slicing."""
    img = make_rgb_image(4096, 3072, seed=99)
    path = tmp_dir / "synthetic_4k.jpg"
    cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return path


@pytest.fixture(scope="session")
def sample_image_dir(tmp_dir: Path, synthetic_rgb_path, synthetic_thermal_path) -> Path:
    """A directory with two synthetic images for batch-inspect tests."""
    d = tmp_dir / "batch_input"
    d.mkdir(exist_ok=True)
    import shutil
    shutil.copy(synthetic_rgb_path, d / synthetic_rgb_path.name)
    shutil.copy(synthetic_thermal_path, d / synthetic_thermal_path.name)
    return d
