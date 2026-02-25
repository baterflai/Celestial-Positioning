"""
Batch accuracy evaluation for the plate solver pipeline.

Usage
-----
python test_accuracy.py

Each test case is an image with a known ground-truth boresight RA/Dec.
Sources for test images:

  1. NASA SkyView FITS  (skyview.gsfc.nasa.gov)
     - Download DSS images as FITS.  The header WCS center IS the ground truth.
     - Use fits_to_jpg() below to extract a JPEG and read back the true RA/Dec.

  2. Stellarium screenshots
     - Set date/time/location, aim at a star, note the RA/Dec, screenshot.
     - Add entry to CASES with that RA/Dec as truth.

  3. nova.astrometry.net sample images
     - Download any solved image, read RA/Dec from the solve result.
"""

import os
import sys

import cv2
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

# Add visual_testing dir so run_demo imports work
sys.path.insert(0, os.path.dirname(__file__))

from run_demo import (
    detect_stars,
    plate_solve,
    load_fallback_wcs,
    solve_attitude,
    compute_residuals,
    boresight_radec,
    angular_separation_deg,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fits_to_jpg(fits_path, jpg_path):
    """
    Convert a grayscale FITS image to JPEG and return its true boresight RA/Dec.

    The WCS CRVAL1/CRVAL2 is the reference pixel's RA/Dec — good enough as a
    boresight approximation when CRPIX is near the image center.
    """
    with fits.open(fits_path) as hdul:
        data = hdul[0].data.astype(np.float32)
        wcs = WCS(hdul[0].header)

    # Stretch to 8-bit
    lo, hi = np.percentile(data, 1), np.percentile(data, 99)
    data = np.clip((data - lo) / (hi - lo + 1e-9) * 255, 0, 255).astype(np.uint8)

    cv2.imwrite(jpg_path, data)

    # True center from CRVAL
    crval = wcs.wcs.crval
    return float(crval[0]), float(crval[1])


def run_pipeline(img_path, fov_hint_deg=2.5, fallback_wcs_path=None):
    """Run the full pipeline on one image.  Returns (ra, dec) or raises."""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    stars = detect_stars(gray)
    if len(stars) < 5:
        raise RuntimeError(f"Too few stars detected ({len(stars)})")

    wcs = None
    try:
        wcs = plate_solve(img_path, fov_hint_deg=fov_hint_deg)
    except Exception as e:
        if fallback_wcs_path:
            wcs = load_fallback_wcs(fallback_wcs_path)
        else:
            raise RuntimeError(f"Plate solve failed and no fallback: {e}") from e

    R, cam_vecs, world_vecs = solve_attitude(stars, wcs, gray.shape)
    ra, dec = boresight_radec(R)
    residuals = compute_residuals(R, cam_vecs, world_vecs)
    return ra, dec, np.mean(residuals)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
#
# Each entry: (label, image_path, true_ra_deg, true_dec_deg, fov_hint_deg)
#
# Populate this list with your downloaded test images.
# The pre-solved sky.jpg is included as a self-consistency smoke test —
# it has no external ground truth so angular_error will read 0 (same WCS).
#
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

CASES = [
    # --- smoke test: pre-solved image (fallback WCS) ----------------------
    # True RA/Dec from wcs.fits CRVAL — run fits_extract_crval() to find it.
    # Placeholder values below; replace after reading wcs.fits header.
    {
        "label": "sky.jpg (fallback WCS, smoke test)",
        "image": os.path.join(DATA_DIR, "stock", "sky.jpg"),
        "true_ra": None,    # fill in after reading wcs.fits CRVAL
        "true_dec": None,   # fill in after reading wcs.fits CRVAL
        "fov_hint": 2.5,
        "fallback_wcs": os.path.join(DATA_DIR, "solved", "wcs.fits"),
    },
    # --- add SkyView / Stellarium / other images here --------------------
    # {
    #     "label": "Orion DSS (SkyView)",
    #     "image": os.path.join(DATA_DIR, "skyview", "orion.jpg"),
    #     "true_ra":  83.82,   # RA of Orion nebula center
    #     "true_dec": -5.39,   # Dec of Orion nebula center
    #     "fov_hint": 1.0,
    #     "fallback_wcs": None,
    # },
]


# ---------------------------------------------------------------------------
# Print true RA/Dec of the fallback WCS so you can fill in CASES above
# ---------------------------------------------------------------------------

def _print_fallback_crval():
    wcs_path = os.path.join(DATA_DIR, "solved", "wcs.fits")
    try:
        with fits.open(wcs_path) as hdul:
            crval = WCS(hdul[0].header).wcs.crval
        print(f"[INFO] wcs.fits CRVAL: RA={crval[0]:.4f}°  Dec={crval[1]:+.4f}°")
        print(f"       Use these as true_ra/true_dec for the smoke test case.\n")
    except Exception as e:
        print(f"[WARN] Could not read wcs.fits: {e}\n")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    _print_fallback_crval()

    results = []
    for case in CASES:
        label = case["label"]
        img_path = case["image"]
        true_ra = case.get("true_ra")
        true_dec = case.get("true_dec")
        fov = case.get("fov_hint", 2.5)
        fallback = case.get("fallback_wcs")

        print(f"Testing: {label}")
        try:
            est_ra, est_dec, mean_resid = run_pipeline(img_path, fov, fallback)
        except Exception as e:
            print(f"  FAILED: {e}\n")
            results.append((label, None, None, None))
            continue

        print(f"  Estimated boresight: RA={est_ra:.3f}°  Dec={est_dec:+.3f}°")
        print(f"  Kabsch residual:     {mean_resid:.4f}°")

        if true_ra is not None and true_dec is not None:
            sep = angular_separation_deg(est_ra, est_dec, true_ra, true_dec)
            sep_km = sep * 111.0
            print(f"  Angular error:       {sep:.4f}°  ({sep_km:.1f} km)")
            results.append((label, sep, sep_km, mean_resid))
        else:
            print(f"  (no ground truth provided — skipping absolute error)")
            results.append((label, None, None, mean_resid))
        print()

    # Summary table
    valid = [(l, s, k) for l, s, k, _ in results if s is not None]
    if valid:
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for label, sep, sep_km in valid:
            status = "PASS" if sep * 111.0 < 10000 else "FAIL"   # <10 km goal
            print(f"  [{status}] {sep:.4f}°  {sep_km:.1f} km   {label}")
        errors_km = [k for _, _, k in valid]
        print(f"\n  Mean error: {np.mean(errors_km):.1f} km")
        print(f"  Max error:  {np.max(errors_km):.1f} km")
        print(f"  Pass (<10 km): {sum(k < 10000 for _, _, k in valid)}/{len(valid)}")


if __name__ == "__main__":
    main()
