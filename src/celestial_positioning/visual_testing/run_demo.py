"""
Architecture
------------
1. Load image (FITS / JPEG / PNG / HEIC)
2. Detect star centroids (OpenCV)
3. Plate-solve → WCS (ASTAP primary, solve-field fallback)
4. Boresight = WCS center (primary output)
5. Extract full rotation matrix from WCS (no Kabsch needed)
6. Fuse IMU gravity + magnetometer heading → observer frame
7. Combine attitude + gravity + timestamp → lat/lon

Usage
-----
  # Basic plate solve (boresight only):
  python run_pipeline_v2.py photo.jpg --fov 60

  # With camera calibration (undistortion + independent attitude check):
  python run_pipeline_v2.py photo.jpg --calib calibration.xml

  # Full positioning with IMU data:
  python run_pipeline_v2.py photo.jpg --calib calibration.xml \\
      --imu imu_data.json --mag mag_data.json --fov 60

  # Force ASTAP or solve-field:
  python run_pipeline_v2.py photo.jpg --solver astap --fov 60
  python run_pipeline_v2.py photo.jpg --solver astrometry --fov 60
"""

import sys
import os
import shutil
import subprocess
import tempfile
import math
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import cv2
import numpy as np
from astropy.wcs import WCS
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import EarthLocation
import astropy.units as u
from scipy.linalg import svd
from scipy.spatial.transform import Rotation


# ===========================================================================
# Image loading (unchanged from v1)
# ===========================================================================

def load_image(path):
    """Load image from JPEG/PNG/TIFF, HEIC/HEIF, or FITS."""
    ext = os.path.splitext(path)[1].lower()

    if ext in (".fits", ".fit", ".fts"):
        with fits.open(path) as hdul:
            data = None
            for hdu in hdul:
                if hdu.data is not None and hdu.data.ndim >= 2:
                    data = hdu.data.astype(np.float32)
                    break
            if data is None:
                raise ValueError(f"No image data found in {path}")
        if data.ndim == 3:
            data = data[0]
        lo, hi = np.percentile(data, 1), np.percentile(data, 99)
        data = np.clip((data - lo) / (hi - lo + 1e-9) * 255, 0, 255).astype(np.uint8)
        gray = data
        img_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    elif ext in (".heic", ".heif"):
        from PIL import Image as PILImage
        import pillow_heif
        pillow_heif.register_heif_opener()
        pil_img = PILImage.open(path)
        rgb = np.array(pil_img.convert("RGB"))
        img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    else:
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    return img_bgr, gray


# ===========================================================================
# Star detection (unchanged from v1)
# ===========================================================================

def detect_stars(gray, max_stars=50, sigma_thresh=3.7):
    """Detect star centroids using adaptive thresholding + connected components."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    mean_val = np.mean(blur)
    std_val = np.std(blur)
    thresh_val = mean_val + sigma_thresh * std_val
    binary = (blur > thresh_val).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)

    candidates = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 2 or area > 500:
            continue
        mask = labels == i
        ys, xs = np.where(mask)
        weights = gray[ys, xs].astype(np.float64)
        total = weights.sum()
        if total == 0:
            continue
        cx = np.sum(xs * weights) / total
        cy = np.sum(ys * weights) / total
        peak = gray[ys, xs].max()
        candidates.append((cx, cy, float(peak)))

    if len(candidates) == 0:
        return np.empty((0, 2))

    candidates.sort(key=lambda c: c[2], reverse=True)
    candidates = candidates[:max_stars]
    return np.array([(c[0], c[1]) for c in candidates])


# ===========================================================================
# Camera calibration
# ===========================================================================

def load_calibration(path):
    """
    Load camera calibration from XML (OpenCV format) or YAML (ROS format).
    Returns dict with camera_matrix, dist_coeffs, image_width, image_height.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in (".yaml", ".yml"):
        return _load_calibration_yaml(path)
    else:
        return _load_calibration_xml(path)


def _load_calibration_xml(xml_path):
    """Load OpenCV XML calibration."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    width = int(root.find("image_Width").text)
    height = int(root.find("image_Height").text)

    cm_node = root.find("Camera_Matrix")
    cm_data = list(map(float, cm_node.find("data").text.split()))
    rows = int(cm_node.find("rows").text)
    cols = int(cm_node.find("cols").text)
    camera_matrix = np.array(cm_data).reshape(rows, cols)

    dc_node = root.find("Distortion_Coefficients")
    dc_data = list(map(float, dc_node.find("data").text.split()))
    dist_coeffs = np.array(dc_data)

    return {
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "image_width": width,
        "image_height": height,
    }


def _load_calibration_yaml(yaml_path):
    """Load ROS-format YAML calibration (from camera_calibrator.py)."""
    import yaml
    with open(yaml_path) as f:
        cal = yaml.safe_load(f)

    cm = cal["camera_matrix"]
    camera_matrix = np.array(cm["data"]).reshape(cm["rows"], cm["cols"])

    dc = cal["distortion_coefficients"]
    dist_coeffs = np.array(dc["data"])

    return {
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "image_width": cal["image_width"],
        "image_height": cal["image_height"],
    }


def undistort_points(stars_px, calib):
    """Remove lens distortion from pixel coordinates."""
    pts = stars_px.reshape(-1, 1, 2).astype(np.float64)
    undist = cv2.undistortPoints(
        pts, calib["camera_matrix"], calib["dist_coeffs"],
        P=calib["camera_matrix"],
    )
    return undist.reshape(-1, 2)


# ===========================================================================
# FOV computation utilities
# ===========================================================================

def fov_from_calibration(calib, image_width):
    """Compute horizontal FOV from calibration intrinsics."""
    fx = calib["camera_matrix"][0, 0]
    scale = image_width / calib["image_width"]
    fx_scaled = fx * scale
    return 2 * math.degrees(math.atan(image_width / (2.0 * fx_scaled)))


def fov_from_fits_header(path):
    """Try to compute FOV from FITS header CDELT or CD matrix."""
    try:
        with fits.open(path) as hdul:
            for hdu in hdul:
                if hdu.data is not None and hdu.data.ndim >= 2:
                    h = hdu.header
                    w = h.get("NAXIS1", 0)
                    cdelt = h.get("CDELT1")
                    if cdelt is not None:
                        return abs(float(cdelt)) * w
                    cd11 = h.get("CD1_1")
                    cd21 = h.get("CD2_1")
                    if cd11 is not None:
                        cd21 = cd21 or 0.0
                        scale = np.sqrt(float(cd11) ** 2 + float(cd21) ** 2)
                        return scale * w
    except Exception:
        pass
    return None


def fov_from_exif(path):
    """Compute horizontal FOV from JPEG/HEIC EXIF data."""
    try:
        from PIL import Image as PILImage
        from PIL.ExifTags import TAGS, IFD

        ext = os.path.splitext(path)[1].lower()
        if ext in (".heic", ".heif"):
            try:
                import pillow_heif
                pillow_heif.register_heif_opener()
            except ImportError:
                return None, None

        img = PILImage.open(path)
        exif_obj = img.getexif()
        if not exif_obj:
            return None, None

        exif = {}
        for tag_id, value in exif_obj.items():
            exif[TAGS.get(tag_id, str(tag_id))] = value
        try:
            exif_ifd = exif_obj.get_ifd(IFD.Exif)
            for tag_id, value in exif_ifd.items():
                exif[TAGS.get(tag_id, str(tag_id))] = value
        except Exception:
            pass

        fl_35mm = exif.get("FocalLengthIn35mmFilm")
        if fl_35mm and float(fl_35mm) > 0:
            fov = 2 * math.degrees(math.atan(36.0 / (2.0 * float(fl_35mm))))
            return fov, f"EXIF 35mm-equiv = {fl_35mm}mm"

        fl = exif.get("FocalLength")
        if fl:
            if hasattr(fl, "numerator"):
                fl_mm = float(fl.numerator) / float(fl.denominator)
            elif isinstance(fl, tuple):
                fl_mm = float(fl[0]) / float(fl[1]) if fl[1] else 0
            else:
                fl_mm = float(fl)
            if fl_mm > 0:
                model = str(exif.get("Model", "")).lower()
                crop_factor = 7.0
                if "iphone 15 pro" in model:
                    crop_factor = 36.0 / 9.8
                elif "iphone" in model:
                    crop_factor = 36.0 / 7.0
                elif "pixel" in model:
                    crop_factor = 36.0 / 8.2
                fl_35mm_equiv = fl_mm * crop_factor
                fov = 2 * math.degrees(math.atan(36.0 / (2.0 * fl_35mm_equiv)))
                return fov, f"EXIF {fl_mm:.1f}mm × {crop_factor:.1f} crop"
    except Exception as e:
        print(f"  (EXIF read failed: {e})")
    return None, None


# ===========================================================================
# Plate solving — ASTAP (primary) + astrometry.net (fallback)
# ===========================================================================

def _find_astap():
    """Locate ASTAP binary."""
    # Check PATH first
    for name in ("astap", "astap_cli"):
        path = shutil.which(name)
        if path:
            return path
    # Common install locations
    candidates = [
        # macOS .app bundle
        "/Applications/ASTAP.app/Contents/MacOS/astap",
        # Linux / Pi
        "/opt/astap/astap",
        "/usr/local/bin/astap",
        "/usr/bin/astap",
        os.path.expanduser("~/astap/astap"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def plate_solve_astap(image_path, fov_hint_deg=None, ra_hint=None,
                      dec_hint=None, timeout=30, out_dir=None,
                      database=None):
    """
    Plate solve using ASTAP.

    ASTAP uses hash-based triangle matching — much faster than
    astrometry.net for narrow/medium fields.

    ASTAP CLI flags (from actual help output):
      -f  filename            (fits, tiff, png, pgm, jpg)
      -r  radius_to_search    (degrees)
      -fov height_field       (degrees) — NOTE: this is field HEIGHT
      -ra  right_ascension    (hours)
      -spd south_pole_dist    (degrees)
      -z  downsample_factor   (0=auto, 1,2,3,4)
      -D  database_abbrev     (d80, d50, d20, d05, g05, w08)
      -d  database_path
      -wcs                    (write .wcs in astrometry.net format)
      -s  max_stars           (default 500)
      -o  output_base         (base path for output files)

    Results written to: filename.ini and filename.wcs
    Star database expected at: /usr/local/opt/astap/

    Parameters
    ----------
    image_path   : path to image (FITS, JPEG, PNG, TIFF)
    fov_hint_deg : approximate FOV HEIGHT in degrees (ASTAP convention)
    ra_hint      : approximate RA in degrees (converted to hours for ASTAP)
    dec_hint     : approximate Dec in degrees (converted to SPD for ASTAP)
    timeout      : seconds before giving up
    out_dir      : directory for output files (cached)
    database     : star database abbreviation (d80/d50/d20/d05/g05/w08)

    Returns
    -------
    astropy.wcs.WCS or None
    """
    astap_bin = _find_astap()
    if astap_bin is None:
        raise FileNotFoundError(
            "ASTAP not found. Install from: "
            "https://www.hnsky.org/astap.htm\n"
            "  macOS: install astap_M1.pkg, then run:\n"
            "    xattr -cr /Applications/ASTAP.app\n"
            "    codesign --force -s - /Applications/ASTAP.app/Contents/MacOS/astap\n"
            "  Also install a star database (D50, G05, etc.)\n"
            "  Database goes in: /usr/local/opt/astap/"
        )

    persistent = out_dir is not None
    if persistent:
        os.makedirs(out_dir, exist_ok=True)
        work_dir = out_dir
    else:
        work_dir = tempfile.mkdtemp(prefix="astap_")

    # ASTAP handles FITS, TIFF, PNG, PGM, JPG natively.
    # For HEIC, convert to JPEG first.
    ext = os.path.splitext(image_path)[1].lower()
    if ext in (".heic", ".heif"):
        img_bgr, _ = load_image(image_path)
        base = os.path.splitext(os.path.basename(image_path))[0]
        converted = os.path.join(work_dir, base + ".jpg")
        cv2.imwrite(converted, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        solve_path = converted
    else:
        solve_path = os.path.abspath(image_path)

    base = os.path.splitext(os.path.basename(solve_path))[0]
    # ASTAP writes output next to input OR at -o path
    out_base = os.path.join(work_dir, base)
    wcs_path = out_base + ".wcs"
    ini_path = out_base + ".ini"

    try:
        # Check for cached solution
        if persistent and os.path.exists(wcs_path):
            print(f"  Cached ASTAP solution found: {wcs_path}")
            with fits.open(wcs_path) as hdul:
                return WCS(hdul[0].header)

        cmd = [
            astap_bin,
            "-f", solve_path,
            "-o", out_base,
            "-wcs",             # output astrometry.net-compatible .wcs
        ]

        # FOV hint — ASTAP takes field HEIGHT in degrees
        # If caller provides width, convert assuming ~3:2 aspect ratio
        if fov_hint_deg is not None:
            # Read actual image to get aspect ratio for width→height conversion
            try:
                test_img = cv2.imread(solve_path, cv2.IMREAD_UNCHANGED)
                if test_img is not None:
                    h, w = test_img.shape[:2]
                    fov_height = fov_hint_deg * (h / w)
                else:
                    fov_height = fov_hint_deg * 0.67  # assume 3:2
            except Exception:
                fov_height = fov_hint_deg * 0.67
            cmd += ["-fov", str(round(fov_height, 2))]

        # Position hint (speeds up search significantly)
        if ra_hint is not None:
            cmd += ["-ra", str(round(ra_hint / 15.0, 4))]  # degrees → hours
        if dec_hint is not None:
            cmd += ["-spd", str(round(dec_hint + 90.0, 4))]  # Dec → SPD

        # Search radius
        if ra_hint is not None:
            cmd += ["-r", "30"]
        else:
            cmd += ["-r", "180"]  # full sky

        # Star database
        if database:
            cmd += ["-D", database]

        # Downsample for speed (0=auto is usually good)
        cmd += ["-z", "0"]

        print(f"  ASTAP cmd: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

        # ASTAP uses return codes: 0=success, 1=no solution, 2=error
        # Check .ini file for solve status
        if os.path.exists(ini_path):
            with open(ini_path) as f:
                ini_content = f.read()
            if "PLTSOLVD=F" in ini_content:
                raise RuntimeError(
                    "ASTAP: no match found. Try a different database "
                    "or check your FOV hint."
                )
            # Log solve details from .ini
            for line in ini_content.strip().split('\n'):
                if line.startswith(('CRVAL', 'CDELT', 'PLTSOLVD')):
                    print(f"    {line}")

        if result.returncode not in (0, 1) and result.returncode != 0:
            stderr = result.stderr.strip() if result.stderr else ""
            stdout = result.stdout.strip() if result.stdout else ""
            raise RuntimeError(
                f"ASTAP exited {result.returncode}. "
                f"stdout: {stdout[:300]}  stderr: {stderr[:300]}"
            )

        if not os.path.exists(wcs_path):
            raise RuntimeError(
                "ASTAP produced no .wcs file. "
                "Check that a star database is installed in "
                "/usr/local/opt/astap/"
            )

        with fits.open(wcs_path) as hdul:
            return WCS(hdul[0].header)

    finally:
        if not persistent:
            shutil.rmtree(work_dir, ignore_errors=True)


def plate_solve_astrometry(image_path, fov_hint_deg=None, timeout=60, out_dir=None):
    """
    Plate solve using astrometry.net solve-field (fallback).
    Same as v1 plate_solve().
    """
    persistent = out_dir is not None
    if persistent:
        os.makedirs(out_dir, exist_ok=True)
        work_dir = out_dir
    else:
        work_dir = tempfile.mkdtemp(prefix="platesolve_")

    ext = os.path.splitext(image_path)[1].lower()
    supported = {".jpg", ".jpeg", ".png", ".tif", ".tiff",
                 ".fits", ".fit", ".fts", ".ppm", ".pgm", ".pnm"}
    if ext in supported:
        solve_path = image_path
    else:
        img_bgr, _ = load_image(image_path)
        base = os.path.splitext(os.path.basename(image_path))[0]
        solve_path = os.path.join(work_dir, base + ".jpg")
        cv2.imwrite(solve_path, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

    base = os.path.splitext(os.path.basename(solve_path))[0]
    new_path = os.path.join(work_dir, base + ".new")

    try:
        if persistent and os.path.exists(new_path):
            print(f"  Cached astrometry.net solution: {work_dir}")
            with fits.open(new_path) as hdul:
                return WCS(hdul[0].header)

        cmd = ["solve-field", "--no-plots", "--overwrite",
               "--dir", work_dir, "--downsample", "2", "--objs", "80"]

        index_dir = os.environ.get("ASTROMETRY_INDEX_DIR")
        if not index_dir:
            default = os.path.expanduser("~/astrometry-index")
            if os.path.isdir(default):
                index_dir = default
        if index_dir:
            cmd += ["--index-dir", index_dir]

        if fov_hint_deg is not None:
            cmd += ["--scale-units", "degwidth",
                    "--scale-low", str(fov_hint_deg * 0.5),
                    "--scale-high", str(fov_hint_deg * 2.0)]

        cmd.append(os.path.abspath(solve_path))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

        if result.returncode != 0:
            raise RuntimeError(f"solve-field exited {result.returncode}")
        if not os.path.exists(new_path):
            raise RuntimeError("solve-field produced no solution")

        with fits.open(new_path) as hdul:
            return WCS(hdul[0].header)

    finally:
        if not persistent:
            shutil.rmtree(work_dir, ignore_errors=True)


def plate_solve(image_path, fov_hint_deg=None, solver="auto",
                ra_hint=None, dec_hint=None, timeout=60, out_dir=None,
                database=None):
    """
    Unified plate solver: tries ASTAP first, falls back to astrometry.net.

    Parameters
    ----------
    solver   : "auto", "astap", or "astrometry"
    database : ASTAP star database abbreviation (d80/d50/d20/d05/g05/w08)
    """
    if solver == "astrometry":
        return plate_solve_astrometry(image_path, fov_hint_deg, timeout, out_dir)

    if solver in ("auto", "astap"):
        try:
            wcs = plate_solve_astap(
                image_path, fov_hint_deg=fov_hint_deg,
                ra_hint=ra_hint, dec_hint=dec_hint,
                timeout=timeout, out_dir=out_dir,
                database=database,
            )
            if wcs is not None:
                return wcs
        except FileNotFoundError as e:
            if solver == "astap":
                raise
            print(f"  ASTAP not available: {e}")
            print(f"  Falling back to astrometry.net...")
        except (RuntimeError, subprocess.TimeoutExpired) as e:
            if solver == "astap":
                raise
            print(f"  ASTAP failed: {e}")
            print(f"  Falling back to astrometry.net...")

    if solver in ("auto",):
        return plate_solve_astrometry(image_path, fov_hint_deg, timeout, out_dir)

    return None


# ===========================================================================
# Coordinate utilities
# ===========================================================================

def radec_to_vec(ra_deg, dec_deg):
    """RA/Dec (degrees) → Nx3 unit vectors in ICRS."""
    ra = np.deg2rad(np.atleast_1d(ra_deg))
    dec = np.deg2rad(np.atleast_1d(dec_deg))
    return np.column_stack([
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec),
    ])


def vec_to_radec(vec):
    """Unit 3-vector → (RA, Dec) in degrees."""
    x, y, z = vec
    dec = np.rad2deg(np.arcsin(np.clip(z, -1.0, 1.0)))
    ra = np.rad2deg(np.arctan2(y, x)) % 360.0
    return ra, dec


def angular_separation_deg(ra1, dec1, ra2, dec2):
    """Great-circle separation between two RA/Dec points (degrees)."""
    v1 = radec_to_vec(ra1, dec1)[0]
    v2 = radec_to_vec(ra2, dec2)[0]
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))


# ===========================================================================
# Attitude extraction from WCS (replaces circular Kabsch)
# ===========================================================================

def attitude_from_wcs(wcs, image_shape, calib=None):
    """
    Extract the camera-to-celestial rotation matrix directly from WCS.

    This gives you a proper 3x3 rotation matrix R such that:
        celestial_vec = R @ camera_vec

    The WCS already encodes this mapping (it IS the plate solution).
    No need for a separate Kabsch solve.

    If calib is provided, accounts for the actual principal point;
    otherwise assumes image center is optical axis.

    Parameters
    ----------
    wcs         : astropy.wcs.WCS from plate solution
    image_shape : (height, width) of the image
    calib       : optional calibration dict

    Returns
    -------
    R : 3x3 rotation matrix (camera frame → celestial ICRS frame)
    """
    h, w = image_shape[:2]

    if calib is not None:
        cx = calib["camera_matrix"][0, 2]
        cy = calib["camera_matrix"][1, 2]
    else:
        cx, cy = w / 2.0, h / 2.0

    # Camera boresight (+Z axis) maps to the sky point at (cx, cy)
    sky_center = wcs.pixel_to_world(cx, cy)
    ra0 = sky_center.ra.deg
    dec0 = sky_center.dec.deg
    z_cel = radec_to_vec(ra0, dec0)[0]  # boresight in celestial frame

    # Camera +X axis: a point 1 pixel to the right of center
    sky_right = wcs.pixel_to_world(cx + 1.0, cy)
    x_dir = radec_to_vec(sky_right.ra.deg, sky_right.dec.deg)[0] - z_cel
    x_dir -= np.dot(x_dir, z_cel) * z_cel  # project onto tangent plane
    x_cel = x_dir / np.linalg.norm(x_dir)

    # Camera +Y axis: completes right-handed frame
    y_cel = np.cross(z_cel, x_cel)
    y_cel /= np.linalg.norm(y_cel)

    # R maps camera [X,Y,Z] to celestial [x,y,z]
    # Column i of R = where camera axis i points in celestial frame
    R = np.column_stack([x_cel, y_cel, z_cel])

    # Ensure proper rotation (det = +1)
    if np.linalg.det(R) < 0:
        R[:, 1] *= -1

    return R


def boresight_from_wcs(wcs, image_shape):
    """RA/Dec of image center from WCS."""
    h, w = image_shape[:2]
    sky = wcs.pixel_to_world(w / 2.0, h / 2.0)
    return float(sky.ra.deg), float(sky.dec.deg)


def rotation_to_euler(R):
    """Convert rotation matrix to (roll, pitch, yaw) in degrees."""
    rot = Rotation.from_matrix(R)
    # ZYX convention: yaw, pitch, roll
    yaw, pitch, roll = rot.as_euler('ZYX', degrees=True)
    return roll, pitch, yaw


# ===========================================================================
# IMU + Magnetometer fusion
# ===========================================================================

def load_imu_data(json_path):
    """Load IMU data from JSON (format from imu_node.py recording)."""
    with open(json_path) as f:
        data = json.load(f)
    return data


def load_mag_data(json_path):
    """Load magnetometer data from JSON."""
    with open(json_path) as f:
        data = json.load(f)
    return data


def estimate_gravity_vector(imu_data, n_samples=200):
    """
    Estimate gravity direction in camera/IMU frame from accelerometer data.

    Uses the first n_samples readings (assumed static/near-static).
    Returns a unit vector pointing in the direction of gravity.

    Parameters
    ----------
    imu_data  : list of dicts with acc_x, acc_y, acc_z
    n_samples : number of samples to average

    Returns
    -------
    gravity_unit : 3-vector, direction of gravity in sensor frame
    gravity_mag  : scalar, measured gravity magnitude (m/s²)
    """
    samples = imu_data[:n_samples]
    acc = np.array([[s["acc_x"], s["acc_y"], s["acc_z"]] for s in samples])

    # Average to reduce noise
    gravity = np.mean(acc, axis=0)
    gravity_mag = np.linalg.norm(gravity)
    gravity_unit = gravity / gravity_mag

    # Sanity check
    if abs(gravity_mag - 9.81) > 0.5:
        print(f"  WARNING: measured gravity = {gravity_mag:.3f} m/s² "
              f"(expected ~9.81). Sensor may be moving or miscalibrated.")

    return gravity_unit, gravity_mag


def estimate_magnetic_heading(mag_data, gravity_unit, n_samples=200):
    """
    Estimate magnetic north direction in sensor frame.

    Projects the magnetic field vector onto the plane perpendicular
    to gravity (horizontal plane), giving the heading direction.

    Parameters
    ----------
    mag_data     : list of dicts with mag_x, mag_y, mag_z (Tesla)
    gravity_unit : unit vector of gravity in sensor frame
    n_samples    : number of samples to average

    Returns
    -------
    north_unit : 3-vector, direction of magnetic north in sensor frame
                 (projected onto horizontal plane)
    """
    samples = mag_data[:n_samples]
    mag = np.array([[s["mag_x"], s["mag_y"], s["mag_z"]] for s in samples])

    mag_avg = np.mean(mag, axis=0)

    # Project onto horizontal plane (perpendicular to gravity)
    mag_horiz = mag_avg - np.dot(mag_avg, gravity_unit) * gravity_unit
    norm = np.linalg.norm(mag_horiz)

    if norm < 1e-10:
        print("  WARNING: magnetic field is nearly parallel to gravity. "
              "Heading unreliable.")
        return np.array([1.0, 0.0, 0.0])

    return mag_horiz / norm


def estimate_gyro_bias(imu_data, n_samples=200):
    """Estimate gyroscope bias from static readings."""
    samples = imu_data[:n_samples]
    gyro = np.array([[s["gyro_x"], s["gyro_y"], s["gyro_z"]] for s in samples])
    bias = np.mean(gyro, axis=0)
    std = np.std(gyro, axis=0)
    return bias, std


# ===========================================================================
# Phase 2: Attitude + Gravity + Timestamp → Lat/Lon
# ===========================================================================

def compute_position(R_cam_to_cel, gravity_cam, timestamp_unix,
                     mag_declination_deg=0.0):
    """
    Compute observer latitude and longitude.

    Algorithm:
      1. Zenith in celestial frame = R @ (-gravity_cam_normalized)
         (gravity points DOWN, zenith points UP)
      2. Dec(zenith) = observer geodetic latitude
      3. RA(zenith) = Local Apparent Sidereal Time (LAST)
      4. Longitude = LAST - GAST (Greenwich Apparent Sidereal Time)

    Parameters
    ----------
    R_cam_to_cel      : 3x3 rotation, camera frame → celestial ICRS
    gravity_cam       : 3-vector, gravity direction in camera frame (unit)
    timestamp_unix    : UNIX timestamp of observation
    mag_declination_deg : magnetic declination at observer location
                          (for future refinement, currently unused in
                          the celestial computation)

    Returns
    -------
    lat_deg  : observer latitude (degrees, +N/-S)
    lon_deg  : observer longitude (degrees, +E/-W, -180 to +180)
    """
    # Zenith = opposite of gravity, transformed to celestial frame
    zenith_cam = -gravity_cam / np.linalg.norm(gravity_cam)
    zenith_cel = R_cam_to_cel @ zenith_cam

    # Convert zenith to RA/Dec
    zenith_ra, zenith_dec = vec_to_radec(zenith_cel)

    # Latitude = declination of zenith
    latitude = zenith_dec

    # RA of zenith = Local Apparent Sidereal Time
    last_deg = zenith_ra  # degrees

    # Compute Greenwich Apparent Sidereal Time
    t = Time(timestamp_unix, format='unix', scale='utc')
    # Greenwich Mean Sidereal Time in degrees
    gast_deg = t.sidereal_time('apparent', longitude=0.0).deg

    # Longitude = LAST - GAST
    longitude = last_deg - gast_deg
    # Normalize to -180..+180
    longitude = ((longitude + 180.0) % 360.0) - 180.0

    return latitude, longitude


# ===========================================================================
# Annotation
# ===========================================================================

def save_annotated(img, stars_px, output_path="data/output_annotated.jpg"):
    """Draw detected star centroids on the image and save."""
    annotated = img.copy()
    for (x, y) in stars_px:
        cv2.circle(annotated, (int(x), int(y)), 8, (0, 255, 0), 2)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, annotated)
    return output_path


# ===========================================================================
# Main pipeline
# ===========================================================================

def run_pipeline(image_path, fov_hint_deg=None, use_fits_wcs=False,
                 calib_path=None, imu_path=None, mag_path=None,
                 solver="auto", database=None, truth_ra=None, truth_dec=None,
                 truth_lat=None, truth_lon=None, timestamp=None):
    """
    Run the full celestial positioning pipeline v2.

    Returns dict with all results.
    """
    results = {}

    # --- Load image ---
    print(f"\n{'='*60}")
    print(f"  Celestial Positioning Pipeline v2")
    print(f"{'='*60}")
    print(f"\nLoading: {image_path}")
    img, gray = load_image(image_path)
    h, w = gray.shape
    print(f"  Image size: {w}x{h}")
    results["image_shape"] = (h, w)

    # --- Load calibration ---
    calib = None
    if calib_path:
        print(f"\nLoading calibration: {calib_path}")
        calib = load_calibration(calib_path)
        print(f"  Sensor: {calib['image_width']}x{calib['image_height']}")
        fx = calib["camera_matrix"][0, 0]
        fy = calib["camera_matrix"][1, 1]
        cx = calib["camera_matrix"][0, 2]
        cy = calib["camera_matrix"][1, 2]
        print(f"  Focal length: fx={fx:.1f}  fy={fy:.1f} px")
        print(f"  Principal pt: cx={cx:.1f}  cy={cy:.1f} px")
        results["calibration"] = calib

    # --- Detect stars ---
    print(f"\nDetecting stars...")
    stars = detect_stars(gray)
    print(f"  {len(stars)} stars detected")
    results["stars_px"] = stars

    if len(stars) < 4:
        print("ERROR: Too few stars for plate solving (need >= 4)")
        results["success"] = False
        return results

    # --- Undistort ---
    stars_for_solve = stars
    if calib is not None:
        print("  Undistorting star positions...")
        stars_for_solve = undistort_points(stars, calib)

    # --- Auto-detect FOV ---
    fov_source = None
    if fov_hint_deg is None:
        if calib is not None:
            fov_hint_deg = fov_from_calibration(calib, w)
            fov_source = f"calibration (fx={calib['camera_matrix'][0,0]:.0f}px)"
        if fov_hint_deg is None:
            ext = os.path.splitext(image_path)[1].lower()
            if ext in (".jpg", ".jpeg", ".heic", ".heif", ".tiff", ".tif"):
                exif_fov, exif_desc = fov_from_exif(image_path)
                if exif_fov is not None:
                    fov_hint_deg = exif_fov
                    fov_source = exif_desc
        if fov_hint_deg is None:
            ext = os.path.splitext(image_path)[1].lower()
            if ext in (".fits", ".fit", ".fts"):
                auto_fov = fov_from_fits_header(image_path)
                if auto_fov:
                    fov_hint_deg = auto_fov
                    fov_source = "FITS header"
    else:
        fov_source = "manual (--fov)"

    # --- Plate solve ---
    wcs = None

    if use_fits_wcs:
        ext = os.path.splitext(image_path)[1].lower()
        if ext in (".fits", ".fit", ".fts"):
            print("\nLoading WCS from FITS header...")
            try:
                with fits.open(image_path) as hdul:
                    wcs = WCS(hdul[0].header)
                print(f"  WCS loaded successfully")
            except Exception as e:
                print(f"  ERROR: {e}")

    if wcs is None and not use_fits_wcs:
        print(f"\nPlate solving (solver={solver})...")
        if fov_hint_deg:
            print(f"  FOV hint: {fov_hint_deg:.2f}° [{fov_source}]")
        else:
            print(f"  WARNING: No FOV hint — search will be slow")

        script_dir = os.path.dirname(os.path.abspath(__file__))
        img_base = os.path.splitext(os.path.basename(image_path))[0]
        solve_out_dir = os.path.join(script_dir, "data", "solved", img_base)

        try:
            wcs = plate_solve(
                image_path, fov_hint_deg=fov_hint_deg, solver=solver,
                database=database, timeout=60, out_dir=solve_out_dir,
            )
            if wcs:
                print(f"  Plate solve: SUCCESS")
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
        except subprocess.TimeoutExpired:
            print(f"  ERROR: solver timed out")
        except RuntimeError as e:
            print(f"  ERROR: {e}")

    if wcs is None:
        print("\nFAILED: No WCS solution. Cannot determine pointing.")
        results["success"] = False
        return results

    results["wcs"] = wcs

    # --- Boresight ---
    wcs_ra, wcs_dec = boresight_from_wcs(wcs, gray.shape)
    results["boresight_ra"] = wcs_ra
    results["boresight_dec"] = wcs_dec

    print(f"\n{'='*50}")
    print(f"  BORESIGHT (plate solution)")
    print(f"  RA  = {wcs_ra:.6f}°  ({wcs_ra/15:.4f}h)")
    print(f"  Dec = {wcs_dec:+.6f}°")
    print(f"{'='*50}")

    # After the boresight print block, add:
    if truth_lat is not None and truth_lon is not None and timestamp is not None:
        from astropy.time import Time
        import astropy.units as u
        t = Time(timestamp, format='unix', scale='utc')
        last = t.sidereal_time('apparent', longitude=truth_lon * u.deg)
        zenith_ra = last.deg
        zenith_dec = truth_lat
        elev_offset = angular_separation_deg(wcs_ra, wcs_dec, zenith_ra, zenith_dec)
        print(f"  Zenith: RA={zenith_ra:.2f}°  Dec={zenith_dec:+.2f}°")
        print(f"  Boresight offset from zenith: {elev_offset:.2f}°")
        print(f"  Elevation: {90.0 - elev_offset:.2f}°")
        

    # --- Attitude from WCS ---
    print(f"\nExtracting camera attitude from WCS...")
    R = attitude_from_wcs(wcs, gray.shape, calib)
    results["attitude_R"] = R

    roll, pitch, yaw = rotation_to_euler(R)
    results["roll_deg"] = roll
    results["pitch_deg"] = pitch
    results["yaw_deg"] = yaw

    print(f"  Rotation matrix (camera → celestial):")
    for i in range(3):
        print(f"    [{R[i,0]:+.6f}  {R[i,1]:+.6f}  {R[i,2]:+.6f}]")
    print(f"  Euler angles: roll={roll:.2f}°  pitch={pitch:.2f}°  yaw={yaw:.2f}°")

    # Verify: boresight from attitude should match WCS boresight
    boresight_check = R @ np.array([0.0, 0.0, 1.0])
    att_ra, att_dec = vec_to_radec(boresight_check)
    sep = angular_separation_deg(wcs_ra, wcs_dec, att_ra, att_dec)
    print(f"  Attitude boresight check: separation = {sep:.6f}° "
          f"({'OK' if sep < 0.01 else 'WARNING: large!'})")

    # --- IMU + Magnetometer → Position ---
    if imu_path and mag_path:
        print(f"\n--- IMU + Magnetometer Fusion ---")

        imu_data = load_imu_data(imu_path)
        mag_data = load_mag_data(mag_path)
        print(f"  IMU samples: {len(imu_data)}")
        print(f"  Mag samples: {len(mag_data)}")

        # Gyro bias estimation
        gyro_bias, gyro_std = estimate_gyro_bias(imu_data)
        print(f"  Gyro bias: [{gyro_bias[0]:.6f}, {gyro_bias[1]:.6f}, "
              f"{gyro_bias[2]:.6f}] rad/s")
        print(f"  Gyro noise: [{gyro_std[0]:.6f}, {gyro_std[1]:.6f}, "
              f"{gyro_std[2]:.6f}] rad/s (1σ)")

        # Gravity vector
        gravity_unit, gravity_mag = estimate_gravity_vector(imu_data)
        print(f"  Gravity direction: [{gravity_unit[0]:+.6f}, "
              f"{gravity_unit[1]:+.6f}, {gravity_unit[2]:+.6f}]")
        print(f"  Gravity magnitude: {gravity_mag:.4f} m/s²")
        results["gravity_vector"] = gravity_unit
        results["gravity_magnitude"] = gravity_mag

        # Magnetic heading
        mag_north = estimate_magnetic_heading(mag_data, gravity_unit)
        print(f"  Magnetic north (horiz): [{mag_north[0]:+.6f}, "
              f"{mag_north[1]:+.6f}, {mag_north[2]:+.6f}]")
        results["magnetic_north"] = mag_north

        # Determine timestamp
        if timestamp:
            obs_time = timestamp
        else:
            # Use middle of IMU recording
            obs_time = imu_data[len(imu_data) // 2]["timestamp"]
        print(f"  Observation time: {datetime.fromtimestamp(obs_time, tz=timezone.utc).isoformat()}")

        # Compute position
        lat, lon = compute_position(R, gravity_unit, obs_time)
        results["latitude"] = lat
        results["longitude"] = lon

        print(f"\n{'='*50}")
        print(f"  OBSERVER POSITION")
        print(f"  Latitude:  {lat:+.4f}° ({'N' if lat >= 0 else 'S'})")
        print(f"  Longitude: {lon:+.4f}° ({'E' if lon >= 0 else 'W'})")
        print(f"{'='*50}")

        # Position accuracy check
        if truth_lat is not None and truth_lon is not None:
            dlat = abs(lat - truth_lat)
            dlon = abs(lon - truth_lon)
            # Approximate distance
            dist_km = np.sqrt(
                (dlat * 111.0) ** 2 +
                (dlon * 111.0 * np.cos(np.deg2rad(truth_lat))) ** 2
            )
            print(f"\n--- Position Accuracy ---")
            print(f"  True:   {truth_lat:+.4f}°, {truth_lon:+.4f}°")
            print(f"  Solved: {lat:+.4f}°, {lon:+.4f}°")
            print(f"  Error:  Δlat={dlat:.4f}°  Δlon={dlon:.4f}°  "
                  f"≈ {dist_km:.1f} km")
            results["position_error_km"] = dist_km

    elif imu_path or mag_path:
        print(f"\n  NOTE: Both --imu and --mag are required for position fix.")
        print(f"  Provide both for lat/lon computation.")

    # --- Boresight accuracy ---
    if truth_ra is not None and truth_dec is not None:
        sep = angular_separation_deg(wcs_ra, wcs_dec, truth_ra, truth_dec)
        print(f"\n--- Boresight Accuracy ---")
        print(f"  True:   RA={truth_ra:.4f}°  Dec={truth_dec:+.4f}°")
        print(f"  Solved: RA={wcs_ra:.4f}°  Dec={wcs_dec:+.4f}°")
        print(f"  Error:  {sep:.4f}° = {sep*3600:.1f}\" ≈ {sep*111:.1f} km")
        results["truth_separation_deg"] = sep

    # --- Save annotated image ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, "data", "output_annotated.jpg")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    saved = save_annotated(img, stars, out_path)
    print(f"\nAnnotated image: {saved}")
    results["annotated_path"] = saved
    results["success"] = True

    return results


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Celestial positioning v2: star image → attitude → lat/lon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s photo.jpg --fov 60                    # boresight only
  %(prog)s photo.jpg --calib calib.yaml           # + attitude
  %(prog)s photo.jpg --calib calib.yaml \\
      --imu imu_data.json --mag mag_data.json     # + lat/lon
  %(prog)s photo.jpg --solver astap --fov 60      # force ASTAP
  %(prog)s image.fits --use-fits-wcs              # use embedded WCS
        """,
    )
    parser.add_argument("image", nargs="?", default="data/stock/sky.jpg")
    parser.add_argument("--fov", type=float, default=None,
                        help="FOV width hint (degrees)")
    parser.add_argument("--calib", type=str, default=None,
                        help="Camera calibration XML or YAML")
    parser.add_argument("--imu", type=str, default=None,
                        help="IMU data JSON (from imu_node.py)")
    parser.add_argument("--mag", type=str, default=None,
                        help="Magnetometer data JSON")
    parser.add_argument("--solver", choices=["auto", "astap", "astrometry"],
                        default="auto", help="Plate solver to use")
    parser.add_argument("--database", "-D", type=str, default=None,
                        choices=["d80", "d50", "d20", "d05", "g05", "w08"],
                        help="ASTAP star database (d80/d50/d20/d05/g05/w08)")
    parser.add_argument("--use-fits-wcs", action="store_true",
                        help="Use WCS from FITS header")
    parser.add_argument("--truth-ra", type=float, default=None)
    parser.add_argument("--truth-dec", type=float, default=None)
    parser.add_argument("--truth-lat", type=float, default=None)
    parser.add_argument("--truth-lon", type=float, default=None)
    parser.add_argument("--timestamp", type=float, default=None,
                        help="UNIX timestamp of observation")
    args = parser.parse_args()

    try:
        results = run_pipeline(
            image_path=args.image,
            fov_hint_deg=args.fov,
            use_fits_wcs=args.use_fits_wcs,
            calib_path=args.calib,
            imu_path=args.imu,
            mag_path=args.mag,
            solver=args.solver,
            database=args.database,
            truth_ra=args.truth_ra,
            truth_dec=args.truth_dec,
            truth_lat=args.truth_lat,
            truth_lon=args.truth_lon,
            timestamp=args.timestamp,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    if not results.get("success", False):
        sys.exit(1)