"""
Celestial Positioning Pipeline
==============================

Phase 1 (current):  star image + timestamp  →  boresight RA/Dec (sky pointing)
Phase 2 (planned):  + camera calibration + IMU  →  attitude + lat/lon

Architecture
------------
1. Load image (FITS / JPEG / PNG)
2. Detect star centroids
3. Plate-solve  →  WCS (maps every pixel to RA/Dec)
4. Boresight = WCS center  (the primary, reliable output)
5. (Optional) Kabsch attitude solve — only meaningful with an INDEPENDENT
   focal length (from camera calibration), not derived from the WCS.
6. (Phase 2) Combine attitude + gravity vector + timestamp  →  lat/lon

FOV auto-detection priority
---------------------------
1. Camera calibration XML  (--calib):  fx + image_width → FOV
2. EXIF focal length       (JPEG/HEIC): FocalLength + sensor size → FOV
3. FITS header             (.fits):     CDELT/CD matrix → FOV
4. Manual hint             (--fov):     user-provided fallback

Usage examples
--------------
  # HEIC from iPhone (FOV auto-detected from EXIF):
  python run_demo.py IMG_1234.heic

  # JPEG with EXIF stripped (need manual hint):
  python run_demo.py screenshot.jpg --fov 60

  # With camera calibration (FOV auto-detected, enables attitude):
  python run_demo.py photo.jpg --calib calibration.xml

  # FITS with embedded WCS (for testing):
  python run_demo.py nasa_180_45.fits --use-fits-wcs
"""

import sys
import os
import shutil
import subprocess
import tempfile
import math
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from astropy.wcs import WCS
from astropy.io import fits
from scipy.linalg import svd


# ===========================================================================
# Image loading
# ===========================================================================

def load_image(path):
    """
    Load an image from JPEG/PNG/TIFF, HEIC/HEIF, or FITS.

    For FITS: uses 1st–99th percentile stretch to 8-bit.
    For HEIC: requires pillow-heif (pip install pillow-heif).

    Returns
    -------
    img_bgr : np.ndarray  uint8 BGR  (for annotation / saving)
    gray    : np.ndarray  uint8 grayscale  (for star detection)
    """
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
        try:
            from PIL import Image as PILImage
            import pillow_heif
            pillow_heif.register_heif_opener()
        except ImportError:
            raise ImportError(
                "HEIC support requires pillow-heif. "
                "Install with: pip install pillow-heif"
            )
        pil_img = PILImage.open(path)
        # Convert to RGB numpy array, then to BGR for OpenCV
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
# Star detection
# ===========================================================================

def detect_stars(gray, max_stars=50, sigma_thresh=3.5):
    """
    Detect star centroids in a grayscale image.

    Uses adaptive thresholding (mean + sigma_thresh * std), connected
    component analysis, and brightness-weighted centroid refinement.

    Parameters
    ----------
    gray         : 2D uint8 array
    max_stars    : maximum number of stars to return
    sigma_thresh : detection threshold in standard deviations above mean

    Returns
    -------
    Nx2 array of (x, y) sub-pixel centroids, brightest first.
    """
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

        # Sub-pixel centroid via intensity-weighted moments
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

def load_calibration_xml(xml_path):
    """
    Load OpenCV camera calibration from XML.

    Returns
    -------
    dict with keys:
        'camera_matrix' : 3x3 np.ndarray
        'dist_coeffs'   : 1x5 np.ndarray
        'image_width'   : int
        'image_height'  : int
    """
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


def undistort_points(stars_px, calib):
    """
    Remove lens distortion from pixel coordinates using calibration.

    Parameters
    ----------
    stars_px : Nx2 array of (x, y) pixel coordinates
    calib    : dict from load_calibration_xml

    Returns
    -------
    Nx2 array of undistorted (x, y) pixel coordinates
    """
    pts = stars_px.reshape(-1, 1, 2).astype(np.float64)
    undist = cv2.undistortPoints(
        pts,
        calib["camera_matrix"],
        calib["dist_coeffs"],
        P=calib["camera_matrix"],  # re-project to pixel coords
    )
    return undist.reshape(-1, 2)


# ===========================================================================
# Plate solving
# ===========================================================================

def _convert_for_solve_field(image_path, work_dir):
    """
    Ensure image is in a format solve-field can read (JPEG/PNG/FITS/TIFF).

    HEIC and other unsupported formats are converted to a temp JPEG.
    Returns the path to use (may be the original or a converted copy).
    """
    ext = os.path.splitext(image_path)[1].lower()
    supported = {".jpg", ".jpeg", ".png", ".tif", ".tiff",
                 ".fits", ".fit", ".fts", ".ppm", ".pgm", ".pnm"}
    if ext in supported:
        return image_path

    # Convert via our load_image (handles HEIC etc.) → save as JPEG
    print(f"  Converting {ext} to JPEG for solve-field...")
    img_bgr, _ = load_image(image_path)
    base = os.path.splitext(os.path.basename(image_path))[0]
    converted = os.path.join(work_dir, base + ".jpg")
    cv2.imwrite(converted, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return converted


def plate_solve(image_path, fov_hint_deg=None, timeout=60, out_dir=None):
    """
    Run astrometry.net solve-field on image_path.

    Automatically converts HEIC and other unsupported formats to JPEG.

    Returns
    -------
    astropy.wcs.WCS or None if solving fails.
    """
    persistent = out_dir is not None
    if persistent:
        os.makedirs(out_dir, exist_ok=True)
        work_dir = out_dir
    else:
        work_dir = tempfile.mkdtemp(prefix="platesolve_")

    # Convert if needed (HEIC → JPEG etc.)
    solve_path = _convert_for_solve_field(image_path, work_dir)
    base = os.path.splitext(os.path.basename(solve_path))[0]
    new_path = os.path.join(work_dir, base + ".new")

    try:
        if persistent and os.path.exists(new_path):
            print(f"  Cached solution found: {work_dir}")
            with fits.open(new_path) as hdul:
                return WCS(hdul[0].header)

        cmd = [
            "solve-field",
            "--no-plots",
            "--overwrite",
            "--dir", work_dir,
            "--downsample", "2",
            "--objs", "80",
        ]

        index_dir = os.environ.get("ASTROMETRY_INDEX_DIR")
        if not index_dir:
            default = os.path.expanduser("~/astrometry-index")
            if os.path.isdir(default):
                index_dir = default
        if index_dir:
            cmd += ["--index-dir", index_dir]

        if fov_hint_deg is not None:
            lo = fov_hint_deg * 0.5
            hi = fov_hint_deg * 2.0
            cmd += [
                "--scale-units", "degwidth",
                "--scale-low", str(lo),
                "--scale-high", str(hi),
            ]

        cmd.append(os.path.abspath(solve_path))

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

        if result.returncode != 0:
            raise RuntimeError(
                f"solve-field exited {result.returncode}: {result.stderr.strip()}"
            )
        if not os.path.exists(new_path):
            raise RuntimeError("solve-field produced no solution")

        with fits.open(new_path) as hdul:
            return WCS(hdul[0].header)

    finally:
        if not persistent:
            shutil.rmtree(work_dir, ignore_errors=True)


def fov_from_fits_header(path):
    """
    Try to compute FOV in degrees from a FITS header (CDELT or CD matrix).
    Returns float or None.
    """
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
    """
    Compute horizontal FOV in degrees from JPEG/HEIC EXIF data.

    Uses FocalLengthIn35mmFilm if available (most reliable),
    otherwise falls back to FocalLength + sensor crop factor estimation.

    Returns (fov_deg, source_description) or (None, None).
    """
    try:
        from PIL import Image as PILImage
        from PIL.ExifTags import TAGS, IFD

        # For HEIC, register opener
        ext = os.path.splitext(path)[1].lower()
        if ext in (".heic", ".heif"):
            try:
                import pillow_heif
                pillow_heif.register_heif_opener()
            except ImportError:
                return None, None

        img = PILImage.open(path)

        # Use modern getexif() API (works for HEIC, JPEG, etc.)
        exif_obj = img.getexif()
        if not exif_obj:
            return None, None

        # Build tag name → value dict from base IFD
        exif = {}
        for tag_id, value in exif_obj.items():
            tag_name = TAGS.get(tag_id, str(tag_id))
            exif[tag_name] = value

        # Also pull from the EXIF sub-IFD (where FocalLength etc. live)
        try:
            exif_ifd = exif_obj.get_ifd(IFD.Exif)
            for tag_id, value in exif_ifd.items():
                tag_name = TAGS.get(tag_id, str(tag_id))
                exif[tag_name] = value
        except Exception:
            pass

        # Method 1: FocalLengthIn35mmFilm (most reliable)
        fl_35mm = exif.get("FocalLengthIn35mmFilm")
        if fl_35mm and float(fl_35mm) > 0:
            fov = 2 * math.degrees(math.atan(36.0 / (2.0 * float(fl_35mm))))
            return fov, f"EXIF 35mm-equiv focal length = {fl_35mm}mm"

        # Method 2: FocalLength (actual) — needs crop factor guess
        fl = exif.get("FocalLength")
        if fl:
            # Handle IFDRational, tuple, or float
            if hasattr(fl, "numerator"):
                fl_mm = float(fl.numerator) / float(fl.denominator)
            elif isinstance(fl, tuple):
                fl_mm = float(fl[0]) / float(fl[1]) if fl[1] else 0
            else:
                fl_mm = float(fl)

            if fl_mm > 0:
                model = str(exif.get("Model", "")).lower()

                # Default: assume ~7x crop factor (typical phone)
                crop_factor = 7.0
                if "iphone 15 pro" in model:
                    crop_factor = 36.0 / 9.8  # 1/1.28" sensor
                elif "iphone 14 pro" in model:
                    crop_factor = 36.0 / 9.8
                elif "iphone" in model:
                    crop_factor = 36.0 / 7.0  # 1/1.65" typical
                elif "pixel" in model:
                    crop_factor = 36.0 / 8.2

                fl_35mm_equiv = fl_mm * crop_factor
                fov = 2 * math.degrees(math.atan(36.0 / (2.0 * fl_35mm_equiv)))
                return fov, (f"EXIF focal length = {fl_mm:.1f}mm, "
                             f"crop ~{crop_factor:.1f}x → "
                             f"~{fl_35mm_equiv:.0f}mm equiv")

    except Exception as e:
        print(f"  (EXIF read failed: {e})")
    return None, None


def fov_from_calibration(calib, image_width):
    """
    Compute horizontal FOV from camera calibration and actual image width.

    Accounts for resolution mismatch between calibration and image.

    Parameters
    ----------
    calib       : dict from load_calibration_xml
    image_width : actual width of the image being solved

    Returns
    -------
    fov_deg : float
    """
    fx = calib["camera_matrix"][0, 0]
    calib_w = calib["image_width"]

    # Scale focal length if image resolution differs from calibration
    scale = image_width / calib_w
    fx_scaled = fx * scale

    fov = 2 * math.degrees(math.atan(image_width / (2.0 * fx_scaled)))
    return fov


# ===========================================================================
# Coordinate utilities
# ===========================================================================

def radec_to_vec(ra_deg, dec_deg):
    """RA/Dec (degrees) → Nx3 unit vectors."""
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


def wcs_pixel_scale_deg(wcs):
    """Pixel scales (deg/pixel) from WCS. Returns (scale_x, scale_y)."""
    try:
        cd = wcs.wcs.cd
        scale_x = np.sqrt(cd[0, 0] ** 2 + cd[1, 0] ** 2)
        scale_y = np.sqrt(cd[0, 1] ** 2 + cd[1, 1] ** 2)
        return scale_x, scale_y
    except AttributeError:
        pass
    cdelt = wcs.wcs.cdelt
    pc = wcs.wcs.get_pc()
    scale_x = abs(cdelt[0]) * np.sqrt(pc[0, 0] ** 2 + pc[1, 0] ** 2)
    scale_y = abs(cdelt[1]) * np.sqrt(pc[0, 1] ** 2 + pc[1, 1] ** 2)
    return scale_x, scale_y


# ===========================================================================
# Boresight (primary output — directly from WCS)
# ===========================================================================

def boresight_from_wcs(wcs, image_shape):
    """
    RA/Dec of the image center directly from the WCS.

    This is the primary, reliable output of the pipeline.
    It does NOT depend on a camera model — just the plate solution.
    """
    h, w = image_shape[:2]
    sky = wcs.pixel_to_world(w / 2.0, h / 2.0)
    return float(sky.ra.deg), float(sky.dec.deg)


# ===========================================================================
# Attitude solver (requires independent camera calibration)
# ===========================================================================

def pixels_to_camera_vectors(stars_px, fx, fy, cx, cy):
    """
    Convert pixel coordinates to unit vectors in the camera frame
    using a calibrated pinhole model.

    Parameters
    ----------
    stars_px : Nx2 array of (x, y) pixel coordinates
    fx, fy   : focal lengths in pixels (from calibration)
    cx, cy   : principal point in pixels (from calibration)

    Returns
    -------
    Nx3 array of unit vectors in camera frame
    """
    vecs = np.zeros((len(stars_px), 3))
    for i, (px, py) in enumerate(stars_px):
        v = np.array([(px - cx) / fx, (py - cy) / fy, 1.0])
        vecs[i] = v / np.linalg.norm(v)
    return vecs


def pixels_to_world_vectors(stars_px, wcs):
    """
    Convert pixel coordinates to unit vectors in the celestial frame
    using the plate-solved WCS.
    """
    sky = wcs.pixel_to_world(stars_px[:, 0], stars_px[:, 1])
    return radec_to_vec(sky.ra.deg, sky.dec.deg)


def solve_attitude_kabsch(cam_vecs, world_vecs):
    """
    Kabsch algorithm: find rotation R that best aligns cam_vecs to world_vecs.

    Both inputs should be Nx3 arrays of unit vectors from INDEPENDENT sources.

    Returns
    -------
    R : 3x3 rotation matrix (camera frame → world/celestial frame)
    """
    H = cam_vecs.T @ world_vecs
    U, _, Vt = svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    S = np.diag([1.0, 1.0, np.sign(d)])
    R = Vt.T @ S @ U.T
    return R


def compute_residuals(R, cam_vecs, world_vecs):
    """Angular residuals (degrees) between R @ cam_vec[i] and world_vec[i]."""
    rotated = (R @ cam_vecs.T).T  # Nx3
    dots = np.clip(np.sum(rotated * world_vecs, axis=1), -1.0, 1.0)
    return np.rad2deg(np.arccos(dots))


def boresight_from_attitude(R):
    """RA/Dec of camera boresight (+Z axis) from rotation matrix."""
    return vec_to_radec(R @ np.array([0.0, 0.0, 1.0]))


# ===========================================================================
# Phase 2 stub: attitude + gravity + timestamp → lat/lon
# ===========================================================================

def compute_position(R_cam_to_cel, gravity_cam, timestamp_utc):
    """
    (Phase 2) Compute observer latitude/longitude.

    Given:
      - R_cam_to_cel : 3x3 rotation, camera frame → celestial (ICRS) frame
      - gravity_cam  : 3-vector, direction of gravity in camera frame (from IMU)
      - timestamp_utc: datetime, observation time

    Algorithm:
      1. zenith_cel = R_cam_to_cel @ (-gravity_cam)   [zenith in celestial frame]
      2. declination of zenith = observer latitude
      3. RA of zenith = local sidereal time
      4. longitude = LST - Greenwich Sidereal Time(timestamp)

    Returns
    -------
    (latitude_deg, longitude_deg) or raises NotImplementedError
    """
    raise NotImplementedError(
        "Phase 2: requires IMU gravity vector + timestamp. "
        "See architecture notes in module docstring."
    )


# ===========================================================================
# Annotation
# ===========================================================================

def save_annotated(img, stars_px, output_path="data/output_annotated.jpg"):
    """Draw detected star centroids on the image and save."""
    annotated = img.copy()
    for (x, y) in stars_px:
        cv2.circle(annotated, (int(x), int(y)), 8, (0, 255, 0), 2)
    cv2.imwrite(output_path, annotated)
    return output_path


# ===========================================================================
# Main
# ===========================================================================

def run_pipeline(image_path, fov_hint_deg=None, use_fits_wcs=False,
                 calib_path=None, truth_ra=None, truth_dec=None):
    """
    Run the full celestial positioning pipeline.

    Parameters
    ----------
    image_path    : path to star image (FITS, JPEG, PNG)
    fov_hint_deg  : approximate FOV for plate solver (degrees).
                    For FITS, auto-detected from header if not provided.
    use_fits_wcs  : if True, use WCS embedded in FITS header (skip solve-field)
    calib_path    : path to OpenCV calibration XML (enables independent attitude)
    truth_ra/dec  : ground truth boresight for accuracy evaluation

    Returns
    -------
    dict with results
    """
    results = {}

    # --- Load image ---
    print(f"Loading: {image_path}")
    img, gray = load_image(image_path)
    h, w = gray.shape
    print(f"  Image size: {w}x{h}")
    results["image_shape"] = (h, w)

    # --- Load calibration if provided ---
    calib = None
    if calib_path:
        print(f"Loading calibration: {calib_path}")
        calib = load_calibration_xml(calib_path)
        print(f"  Sensor: {calib['image_width']}x{calib['image_height']}")
        fx = calib["camera_matrix"][0, 0]
        fy = calib["camera_matrix"][1, 1]
        print(f"  Focal length: fx={fx:.1f}  fy={fy:.1f} pixels")
        results["calibration"] = calib

    # --- Detect stars ---
    print("Detecting stars...")
    stars = detect_stars(gray)
    print(f"  {len(stars)} stars detected")
    results["stars_px"] = stars

    if len(stars) < 4:
        print("ERROR: Too few stars detected for plate solving (need >= 4)")
        return results

    # --- Undistort if calibrated ---
    stars_for_solve = stars
    if calib is not None:
        print("Undistorting star positions...")
        stars_for_solve = undistort_points(stars, calib)
        results["stars_undistorted"] = stars_for_solve

    # --- Plate solve ---
    wcs = None

    if use_fits_wcs:
        ext = os.path.splitext(image_path)[1].lower()
        if ext not in (".fits", ".fit", ".fts"):
            print("WARNING: --use-fits-wcs only works with FITS files")
        else:
            print("Loading WCS from FITS header...")
            try:
                with fits.open(image_path) as hdul:
                    wcs = WCS(hdul[0].header)
                crval = wcs.wcs.crval
                print(f"  WCS center: RA={crval[0]:.4f}°  Dec={crval[1]:+.4f}°")
                if truth_ra is None and truth_dec is None:
                    truth_ra, truth_dec = float(crval[0]), float(crval[1])
                    print(f"  (Auto-set ground truth from WCS center)")
            except Exception as e:
                print(f"  ERROR reading FITS WCS: {e}")
                wcs = None

    if wcs is None and not use_fits_wcs:
        # --- Auto-detect FOV (priority: calibration > EXIF > FITS > manual) ---
        fov_source = None
        if fov_hint_deg is None:
            # Priority 1: Camera calibration
            if calib is not None:
                fov_hint_deg = fov_from_calibration(calib, w)
                fov_source = f"calibration (fx={calib['camera_matrix'][0,0]:.0f}px)"

            # Priority 2: EXIF focal length
            if fov_hint_deg is None:
                ext = os.path.splitext(image_path)[1].lower()
                if ext in (".jpg", ".jpeg", ".heic", ".heif", ".tiff", ".tif"):
                    exif_fov, exif_desc = fov_from_exif(image_path)
                    if exif_fov is not None:
                        fov_hint_deg = exif_fov
                        fov_source = exif_desc

            # Priority 3: FITS header
            if fov_hint_deg is None:
                ext = os.path.splitext(image_path)[1].lower()
                if ext in (".fits", ".fit", ".fts"):
                    auto_fov = fov_from_fits_header(image_path)
                    if auto_fov:
                        fov_hint_deg = auto_fov
                        fov_source = "FITS header (CDELT)"
        else:
            fov_source = "manual (--fov)"

        print(f"Running plate solver (solve-field)...")
        if fov_hint_deg:
            print(f"  FOV: {fov_hint_deg:.2f}° [{fov_source}]")
            print(f"  Search range: {fov_hint_deg*0.5:.2f}°–{fov_hint_deg*2:.2f}°")
        else:
            print(f"  WARNING: No FOV hint — solve-field will search all scales (slow)")
            print(f"  Tip: provide --fov <degrees> or use original photo with EXIF")

        script_dir = os.path.dirname(os.path.abspath(__file__))
        img_base = os.path.splitext(os.path.basename(image_path))[0]
        solve_out_dir = os.path.join(script_dir, "data", "solved", img_base)

        try:
            wcs = plate_solve(image_path, fov_hint_deg=fov_hint_deg,
                              out_dir=solve_out_dir)
            print(f"  Plate solve: OK")
        except FileNotFoundError:
            print("  ERROR: solve-field not installed")
        except subprocess.TimeoutExpired:
            print("  ERROR: solve-field timed out")
        except RuntimeError as e:
            print(f"  ERROR: {e}")

    if wcs is None:
        print("\nFAILED: No WCS solution available. Cannot determine pointing.")
        print("  Possible fixes:")
        print("  - For FITS with embedded WCS: add --use-fits-wcs")
        print("  - Wrong FOV hint: try --fov <degrees>")
        print("  - Install astrometry.net index files for your image scale")
        results["success"] = False
        return results

    results["wcs"] = wcs

    # --- Primary output: boresight from WCS ---
    wcs_ra, wcs_dec = boresight_from_wcs(wcs, gray.shape)
    results["boresight_ra"] = wcs_ra
    results["boresight_dec"] = wcs_dec

    print(f"\n{'='*50}")
    print(f"  BORESIGHT (from plate solution)")
    print(f"  RA  = {wcs_ra:.4f}°")
    print(f"  Dec = {wcs_dec:+.4f}°")
    print(f"{'='*50}")

    # --- Kabsch attitude (only if independently calibrated) ---
    if calib is not None:
        print(f"\nAttitude solve (independent camera model)...")
        fx = calib["camera_matrix"][0, 0]
        fy = calib["camera_matrix"][1, 1]
        cx = calib["camera_matrix"][0, 2]
        cy = calib["camera_matrix"][1, 2]

        cam_vecs = pixels_to_camera_vectors(stars_for_solve, fx, fy, cx, cy)
        world_vecs = pixels_to_world_vectors(stars_for_solve, wcs)

        R = solve_attitude_kabsch(cam_vecs, world_vecs)
        residuals = compute_residuals(R, cam_vecs, world_vecs)
        att_ra, att_dec = boresight_from_attitude(R)

        results["attitude_R"] = R
        results["residuals_deg"] = residuals
        results["attitude_ra"] = att_ra
        results["attitude_dec"] = att_dec

        print(f"  Boresight (attitude): RA={att_ra:.4f}°  Dec={att_dec:+.4f}°")
        print(f"  Residuals: mean={np.mean(residuals):.4f}°  "
              f"max={np.max(residuals):.4f}°  "
              f"({len(residuals)} stars)")

        # Cross-check: attitude boresight vs WCS boresight
        cross = angular_separation_deg(wcs_ra, wcs_dec, att_ra, att_dec)
        print(f"  WCS vs attitude boresight: {cross:.4f}° separation")
        if cross > 0.5:
            print(f"  WARNING: large discrepancy — check calibration or "
                  f"image/calibration resolution mismatch")
    else:
        print(f"\n  (No camera calibration — Kabsch attitude solve skipped)")
        print(f"  Provide --calib <file.xml> for independent attitude + "
              f"meaningful residuals")

    # --- Accuracy evaluation ---
    if truth_ra is not None and truth_dec is not None:
        sep = angular_separation_deg(wcs_ra, wcs_dec, truth_ra, truth_dec)
        sep_arcsec = sep * 3600.0
        sep_km = sep * 111.0
        print(f"\n--- Accuracy vs ground truth ---")
        print(f"  True:    RA={truth_ra:.4f}°  Dec={truth_dec:+.4f}°")
        print(f"  Solved:  RA={wcs_ra:.4f}°  Dec={wcs_dec:+.4f}°")
        print(f"  Error:   {sep:.4f}° = {sep_arcsec:.1f}\" = ~{sep_km:.1f} km")
        results["truth_separation_deg"] = sep

    # --- Annotated output ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, "data", "output_annotated.jpg")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    saved = save_annotated(img, stars, out_path)
    print(f"\nAnnotated image: {saved}")
    results["annotated_path"] = saved
    results["success"] = True

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Celestial positioning: star image → sky pointing (+ attitude with calibration)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s photo.heic                                 # iPhone HEIC (FOV from EXIF)
  %(prog)s photo.jpg                                  # JPEG with EXIF
  %(prog)s screenshot.jpg --fov 60                    # no EXIF, manual FOV hint
  %(prog)s photo.jpg --calib calib.xml                # + independent attitude
  %(prog)s image.fits --use-fits-wcs                  # FITS with embedded WCS
        """,
    )
    parser.add_argument("image", nargs="?", default="data/stock/sky.jpg",
                        help="Path to input image")
    parser.add_argument("--truth-ra", type=float, default=None,
                        help="True boresight RA (deg) for accuracy evaluation")
    parser.add_argument("--truth-dec", type=float, default=None,
                        help="True boresight Dec (deg) for accuracy evaluation")
    parser.add_argument("--fov", type=float, default=None,
                        help="Approximate FOV width hint for plate solver (deg). "
                             "Auto-detected from FITS headers when possible.")
    parser.add_argument("--calib", type=str, default=None,
                        help="Path to OpenCV calibration XML for independent "
                             "attitude solve")
    parser.add_argument("--use-fits-wcs", action="store_true",
                        help="Use WCS embedded in FITS header (skip solve-field)")
    args = parser.parse_args()

    try:
        results = run_pipeline(
            image_path=args.image,
            fov_hint_deg=args.fov,
            use_fits_wcs=args.use_fits_wcs,
            calib_path=args.calib,
            truth_ra=args.truth_ra,
            truth_dec=args.truth_dec,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    if not results.get("success", False):
        sys.exit(1)