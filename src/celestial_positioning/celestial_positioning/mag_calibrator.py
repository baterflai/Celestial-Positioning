import collections
import json
import math
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import numpy
import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import MagneticField


MAG_CALIBRATION_HTML = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Magnetometer Calibration</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #f5f6f8; color: #1a1a1a; }
  .container { max-width: 980px; margin: 0 auto; padding: 20px; }
  header { display: flex; align-items: center; justify-content: space-between;
           margin-bottom: 16px; }
  header h1 { font-size: 22px; font-weight: 700; color: #111; }
  header .badge { font-size: 12px; font-weight: 600; padding: 3px 10px;
                  border-radius: 20px; background: #e8e8e8; color: #666; }
  header .badge.live { background: #dcfce7; color: #16a34a; }
  .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;
           margin: 16px 0; }
  .stat-card { background: #fff; border-radius: 10px; padding: 14px 16px;
               box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
  .stat-card .label { font-size: 11px; font-weight: 600; text-transform: uppercase;
                      letter-spacing: 0.5px; color: #888; margin-bottom: 4px; }
  .stat-card .value { font-size: 22px; font-weight: 700; }
  .stat-card .value.green { color: #16a34a; }
  .stat-card .value.muted { color: #bbb; }
  .readings { background: #fff; border-radius: 12px; overflow: hidden;
              box-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06);
              padding: 20px; margin-bottom: 16px; }
  .readings h2 { font-size: 14px; font-weight: 600; color: #555;
                 margin-bottom: 12px; text-transform: uppercase;
                 letter-spacing: 0.5px; }
  .reading-vals { font-family: 'SF Mono', 'Fira Code', monospace; font-size: 16px;
                  font-weight: 600; line-height: 1.6; }
  .reading-vals .axis { color: #888; font-size: 12px; margin-right: 4px; }
  .scatter-wrap { background: #fff; border-radius: 12px;
                  box-shadow: 0 1px 3px rgba(0,0,0,0.08);
                  padding: 20px; margin-bottom: 16px; }
  .scatter-wrap h2 { font-size: 14px; font-weight: 600; color: #555;
                     margin-bottom: 12px; text-transform: uppercase;
                     letter-spacing: 0.5px; }
  .scatter-row { display: flex; gap: 12px; justify-content: center; }
  .scatter-panel { text-align: center; }
  .scatter-panel canvas { background: #fafafa; border: 1px solid #eee;
                          border-radius: 8px; }
  .scatter-panel .s-label { font-size: 11px; font-weight: 600; color: #888;
                            margin-top: 4px; }
  .guidance { background: #eff6ff; border-radius: 10px; padding: 14px 16px;
              margin-bottom: 16px; font-size: 13px; color: #1e40af;
              line-height: 1.5; }
  .controls { display: flex; justify-content: center; gap: 10px; margin: 16px 0;
              flex-wrap: wrap; }
  button { padding: 10px 24px; font-size: 14px; font-weight: 600; cursor: pointer;
           border: none; border-radius: 8px; transition: all 0.15s ease; }
  button:hover:not(:disabled) { transform: translateY(-1px);
                                 box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
  button:active:not(:disabled) { transform: translateY(0); }
  button:disabled { opacity: 0.35; cursor: not-allowed; }
  .btn-collect { background: #16a34a; color: #fff; }
  .btn-collect.active { background: #dc2626; }
  .btn-capture { background: #2563eb; color: #fff; }
  .btn-calibrate { background: #f59e0b; color: #fff; }
  .btn-save { background: #16a34a; color: #fff; }
  .btn-reset { background: #fff; color: #666; border: 1px solid #ddd; }
  .btn-reset:hover:not(:disabled) { background: #fee2e2; color: #dc2626;
                                     border-color: #fca5a5; }
  .log-wrap { background: #fff; border-radius: 10px; padding: 14px;
              box-shadow: 0 1px 3px rgba(0,0,0,0.06); margin-top: 16px; }
  .log-wrap .log-title { font-size: 11px; font-weight: 600; text-transform: uppercase;
                         letter-spacing: 0.5px; color: #888; margin-bottom: 8px; }
  #log { font-family: 'SF Mono', 'Fira Code', monospace; font-size: 12px;
         line-height: 1.6; white-space: pre-wrap; color: #555; height: 120px;
         overflow-y: auto; }
  #log .ok { color: #16a34a; }
  #log .err { color: #dc2626; }
  #log .info { color: #2563eb; }
  @media (max-width: 640px) {
    .stats { grid-template-columns: repeat(2, 1fr); }
    .scatter-row { flex-direction: column; align-items: center; }
  }
</style></head><body>
<div class="container">
<header>
  <h1>Magnetometer Calibration</h1>
  <span class="badge" id="liveBadge">Connecting...</span>
</header>
<div class="stats">
  <div class="stat-card">
    <div class="label">Samples</div>
    <div class="value" id="sampleCount">0</div>
  </div>
  <div class="stat-card">
    <div class="label">Coverage</div>
    <div class="value muted" id="coverage">--</div>
  </div>
  <div class="stat-card">
    <div class="label">Collecting</div>
    <div class="value muted" id="collecting">Stopped</div>
  </div>
  <div class="stat-card">
    <div class="label">Fit Error</div>
    <div class="value muted" id="fitError">--</div>
  </div>
</div>
<div class="readings">
  <h2>Live Magnetometer Reading (T)</h2>
  <div class="reading-vals" id="mag">
    <span class="axis">X:</span> --<br>
    <span class="axis">Y:</span> --<br>
    <span class="axis">Z:</span> --
  </div>
</div>
<div class="scatter-wrap">
  <h2>Sample Distribution</h2>
  <div class="scatter-row">
    <div class="scatter-panel">
      <canvas id="cXY" width="220" height="220"></canvas>
      <div class="s-label">X vs Y</div>
    </div>
    <div class="scatter-panel">
      <canvas id="cXZ" width="220" height="220"></canvas>
      <div class="s-label">X vs Z</div>
    </div>
    <div class="scatter-panel">
      <canvas id="cYZ" width="220" height="220"></canvas>
      <div class="s-label">Y vs Z</div>
    </div>
  </div>
</div>
<div class="guidance">
  Slowly rotate the sensor in all directions, covering as many orientations
  as possible. Click <strong>Start Collecting</strong> to begin auto-capture,
  then move the sensor. Aim for &gt;80% angular coverage. The scatter plots
  should show a dense ellipsoidal cloud.
</div>
<div class="controls">
  <button class="btn-collect" id="btnCollect" onclick="toggleCollect()">
    Start Collecting</button>
  <button class="btn-capture" id="btnCaptureSingle" onclick="doCapture()">
    Capture Single</button>
  <button class="btn-calibrate" id="btnCalibrate" onclick="doCalibrate()" disabled>
    Calibrate</button>
  <button class="btn-save" id="btnSave" onclick="doSave()" disabled>
    Save YAML</button>
  <button class="btn-reset" id="btnReset" onclick="doReset()">Reset All</button>
</div>
<div class="log-wrap">
  <div class="log-title">Activity Log</div>
  <div id="log">Waiting for magnetometer data...</div>
</div>
</div>
<script>
const logEl = document.getElementById('log');
let isCollecting = false;
let rawSamples = [];
let correctedSamples = [];

function log(msg, cls) {
  const span = document.createElement('span');
  if (cls) span.className = cls;
  span.textContent = '\\n' + msg;
  logEl.appendChild(span);
  logEl.scrollTop = logEl.scrollHeight;
}
function post(url) {
  return fetch(url, {method:'POST'}).then(r => r.json());
}
function toggleCollect() {
  const btn = document.getElementById('btnCollect');
  if (isCollecting) {
    post('/stop_collect').then(() => {
      isCollecting = false;
      btn.textContent = 'Start Collecting';
      btn.classList.remove('active');
      log('Collection stopped', 'info');
    }).catch(() => log('Request failed', 'err'));
  } else {
    post('/start_collect').then(() => {
      isCollecting = true;
      btn.textContent = 'Stop Collecting';
      btn.classList.add('active');
      log('Collection started', 'ok');
    }).catch(() => log('Request failed', 'err'));
  }
}
function doCapture() {
  post('/capture').then(d => {
    if (d.error) log(d.error, 'err');
    else log('Captured sample ' + d.captured, 'ok');
  }).catch(() => log('Request failed', 'err'));
}
function doCalibrate() {
  log('Running ellipsoid fit...', 'info');
  document.getElementById('btnCalibrate').disabled = true;
  post('/calibrate').then(d => {
    if (d.error) log(d.error, 'err');
    else {
      log('Calibrated! Fit error: ' + (d.fit_error * 100).toFixed(2) +
          '%, Hard iron: [' +
          d.hard_iron.map(v => v.toExponential(4)).join(', ') + ']', 'ok');
      if (d.corrected_samples) correctedSamples = d.corrected_samples;
    }
  }).catch(() => log('Request failed', 'err'));
}
function doSave() {
  post('/save').then(d => {
    if (d.error) log(d.error, 'err');
    else log('Saved to ' + d.saved_to, 'ok');
  }).catch(() => log('Request failed', 'err'));
}
function doReset() {
  if (!confirm('Discard all collected samples and calibration results?')) return;
  post('/reset').then(() => {
    rawSamples = [];
    correctedSamples = [];
    isCollecting = false;
    const btn = document.getElementById('btnCollect');
    btn.textContent = 'Start Collecting';
    btn.classList.remove('active');
    log('Reset complete', 'info');
  }).catch(() => log('Request failed', 'err'));
}

function drawScatter(canvasId, pts, corrPts, idxA, idxB, labelA, labelB) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (pts.length === 0) return;
  // Compute range from raw samples
  let minA = Infinity, maxA = -Infinity, minB = Infinity, maxB = -Infinity;
  for (const p of pts) {
    if (p[idxA] < minA) minA = p[idxA];
    if (p[idxA] > maxA) maxA = p[idxA];
    if (p[idxB] < minB) minB = p[idxB];
    if (p[idxB] > maxB) maxB = p[idxB];
  }
  for (const p of corrPts) {
    if (p[idxA] < minA) minA = p[idxA];
    if (p[idxA] > maxA) maxA = p[idxA];
    if (p[idxB] < minB) minB = p[idxB];
    if (p[idxB] > maxB) maxB = p[idxB];
  }
  const pad = 15;
  const rangeA = (maxA - minA) || 1e-6;
  const rangeB = (maxB - minB) || 1e-6;
  const range = Math.max(rangeA, rangeB) * 1.1;
  const cenA = (minA + maxA) / 2, cenB = (minB + maxB) / 2;
  function toX(v) { return pad + (v - cenA + range/2) / range * (w - 2*pad); }
  function toY(v) { return h - pad - (v - cenB + range/2) / range * (h - 2*pad); }
  // Grid
  ctx.strokeStyle = '#eee'; ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(toX(cenA), pad); ctx.lineTo(toX(cenA), h - pad);
  ctx.moveTo(pad, toY(cenB)); ctx.lineTo(w - pad, toY(cenB));
  ctx.stroke();
  // Raw points
  ctx.fillStyle = 'rgba(37, 99, 235, 0.4)';
  for (const p of pts) {
    ctx.beginPath();
    ctx.arc(toX(p[idxA]), toY(p[idxB]), 1.5, 0, Math.PI * 2);
    ctx.fill();
  }
  // Corrected points
  if (corrPts.length > 0) {
    ctx.fillStyle = 'rgba(22, 163, 74, 0.5)';
    for (const p of corrPts) {
      ctx.beginPath();
      ctx.arc(toX(p[idxA]), toY(p[idxB]), 1.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

// Fetch samples for scatter plots every 2 seconds
setInterval(() => {
  fetch('/samples').then(r => r.json()).then(d => {
    rawSamples = d.raw || [];
    if (d.corrected) correctedSamples = d.corrected;
    drawScatter('cXY', rawSamples, correctedSamples, 0, 1, 'X', 'Y');
    drawScatter('cXZ', rawSamples, correctedSamples, 0, 2, 'X', 'Z');
    drawScatter('cYZ', rawSamples, correctedSamples, 1, 2, 'Y', 'Z');
  }).catch(() => {});
}, 2000);

// Status polling
setInterval(() => {
  fetch('/status').then(r => r.json()).then(d => {
    document.getElementById('sampleCount').textContent = d.capture_count;
    const covEl = document.getElementById('coverage');
    covEl.textContent = d.coverage_pct.toFixed(0) + '%';
    covEl.className = 'value' + (d.coverage_pct >= 80 ? ' green' : '');
    const colEl = document.getElementById('collecting');
    colEl.textContent = d.collecting ? 'Active' : 'Stopped';
    colEl.className = 'value' + (d.collecting ? ' green' : ' muted');
    const errEl = document.getElementById('fitError');
    if (d.fit_error !== null) {
      errEl.textContent = (d.fit_error * 100).toFixed(2) + '%';
      errEl.className = 'value' + (d.fit_error < 0.02 ? ' green' : '');
    } else { errEl.textContent = '--'; errEl.className = 'value muted'; }
    if (d.latest_mag)
      document.getElementById('mag').innerHTML =
        '<span class="axis">X:</span> ' + d.latest_mag[0].toExponential(4) + '<br>' +
        '<span class="axis">Y:</span> ' + d.latest_mag[1].toExponential(4) + '<br>' +
        '<span class="axis">Z:</span> ' + d.latest_mag[2].toExponential(4);
    document.getElementById('btnCalibrate').disabled = d.capture_count < 200;
    document.getElementById('btnSave').disabled = !d.calibrated;
    const badge = document.getElementById('liveBadge');
    if (d.sample_count > 0) { badge.textContent = 'Live';
      badge.className = 'badge live'; }
  }).catch(() => {});
}, 500);
</script></body></html>"""


class MagCalibratorState:
    """Thread-safe shared state for magnetometer calibration."""

    def __init__(self):
        self.lock = threading.Lock()
        # Live data
        self.latest_mag = [0.0, 0.0, 0.0]
        self.sample_count = 0
        # Collected samples
        self.mag_samples = []
        self.capture_count = 0
        self.collecting = False
        self.last_collect_time = 0.0
        # Spatial coverage tracking (quantized spherical bins)
        self.coverage_bins = set()
        self.coverage_pct = 0.0
        # Calibration results
        self.calibrated = False
        self.hard_iron = None
        self.soft_iron = None
        self.fit_error = None
        self.corrected_samples = None
        # Config
        self.min_samples = 200
        self.collect_interval = 0.05
        self.save_path = os.path.expanduser('~/imu_calibration.yaml')
        # Coverage grid: 6 theta bins x 12 phi bins = 72 bins
        self.n_theta_bins = 6
        self.n_phi_bins = 12
        self.total_bins = 6 * 12


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class MagCalibrationHTTPHandler(BaseHTTPRequestHandler):
    state = None
    node_logger = None

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == '/':
            self._serve_html()
        elif self.path == '/status':
            self._serve_status()
        elif self.path == '/samples':
            self._serve_samples()
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == '/start_collect':
            self._handle_start_collect()
        elif self.path == '/stop_collect':
            self._handle_stop_collect()
        elif self.path == '/capture':
            self._handle_capture()
        elif self.path == '/calibrate':
            self._handle_calibrate()
        elif self.path == '/save':
            self._handle_save()
        elif self.path == '/reset':
            self._handle_reset()
        else:
            self.send_error(404)

    def _serve_html(self):
        content = MAG_CALIBRATION_HTML.encode()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.send_header('Content-Length', str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _serve_status(self):
        with self.state.lock:
            data = {
                'latest_mag': list(self.state.latest_mag),
                'sample_count': self.state.sample_count,
                'capture_count': self.state.capture_count,
                'collecting': self.state.collecting,
                'coverage_pct': self.state.coverage_pct,
                'calibrated': self.state.calibrated,
                'fit_error': self.state.fit_error,
            }
        self._json_response(200, data)

    def _serve_samples(self):
        with self.state.lock:
            # Downsample for the scatter plot if too many
            samples = list(self.state.mag_samples)
            corrected = None
            if self.state.corrected_samples is not None:
                corrected = self.state.corrected_samples.tolist()
        if len(samples) > 2000:
            step = len(samples) // 2000
            samples = samples[::step]
            if corrected:
                corrected = corrected[::step]
        data = {'raw': [s.tolist() for s in samples]}
        if corrected:
            data['corrected'] = corrected
        self._json_response(200, data)

    def _handle_start_collect(self):
        with self.state.lock:
            self.state.collecting = True
        self._json_response(200, {'collecting': True})

    def _handle_stop_collect(self):
        with self.state.lock:
            self.state.collecting = False
        self._json_response(200, {'collecting': False})

    def _handle_capture(self):
        with self.state.lock:
            sample = numpy.array(self.state.latest_mag)
            self.state.mag_samples.append(sample)
            self.state.capture_count += 1
            _update_coverage(self.state, sample)
            count = self.state.capture_count
        self._json_response(200, {'captured': count})

    def _handle_calibrate(self):
        with self.state.lock:
            if self.state.capture_count < self.state.min_samples:
                self._json_response(400, {
                    'error': f'Need at least {self.state.min_samples} '
                             f'samples, have {self.state.capture_count}'})
                return
            samples = numpy.array(self.state.mag_samples)

        if self.node_logger:
            self.node_logger.info(
                f'Running ellipsoid fit with {len(samples)} samples...')

        hard_iron, soft_iron, fit_error = fit_ellipsoid(samples)

        # Compute corrected samples for visualization
        corrected = (soft_iron @ (samples.T - hard_iron.reshape(3, 1))).T

        with self.state.lock:
            self.state.calibrated = True
            self.state.hard_iron = hard_iron
            self.state.soft_iron = soft_iron
            self.state.fit_error = fit_error
            self.state.corrected_samples = corrected

        if self.node_logger:
            self.node_logger.info(
                f'Ellipsoid fit done. Relative error: '
                f'{fit_error * 100:.2f}%')

        self._json_response(200, {
            'fit_error': fit_error,
            'hard_iron': hard_iron.tolist(),
            'soft_iron': soft_iron.tolist(),
            'corrected_samples': corrected.tolist(),
        })

    def _handle_save(self):
        with self.state.lock:
            if not self.state.calibrated:
                self._json_response(400, {'error': 'Not calibrated yet'})
                return
            h = self.state.hard_iron.copy()
            S = self.state.soft_iron.copy()
            save_path = self.state.save_path

        # Merge with existing file if present
        calibration = {}
        if os.path.exists(save_path):
            with open(save_path) as f:
                existing = yaml.safe_load(f)
                if isinstance(existing, dict):
                    calibration = existing

        calibration['magnetometer'] = {
            'hard_iron': h.tolist(),
            'soft_iron': S.tolist(),
        }

        with open(save_path, 'w') as f:
            yaml.dump(calibration, f, default_flow_style=False)

        if self.node_logger:
            self.node_logger.info(
                f'Magnetometer calibration saved to {save_path}')
        self._json_response(200, {'saved_to': save_path})

    def _handle_reset(self):
        with self.state.lock:
            self.state.mag_samples.clear()
            self.state.capture_count = 0
            self.state.collecting = False
            self.state.coverage_bins.clear()
            self.state.coverage_pct = 0.0
            self.state.calibrated = False
            self.state.hard_iron = None
            self.state.soft_iron = None
            self.state.fit_error = None
            self.state.corrected_samples = None

        if self.node_logger:
            self.node_logger.info('Magnetometer calibration data reset')
        self._json_response(200, {'status': 'reset'})

    def _json_response(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _update_coverage(state, sample):
    """Update spherical coverage bins. Call with lock held."""
    samples = state.mag_samples
    if len(samples) < 2:
        return
    # Estimate centroid from all samples so far
    arr = numpy.array(samples)
    centroid = arr.mean(axis=0)
    v = sample - centroid
    norm = numpy.linalg.norm(v)
    if norm < 1e-12:
        return
    v = v / norm
    # Spherical coordinates
    theta = math.acos(max(-1.0, min(1.0, v[2])))  # 0 to pi
    phi = math.atan2(v[1], v[0])  # -pi to pi
    # Quantize
    t_bin = int(theta / math.pi * state.n_theta_bins)
    t_bin = min(t_bin, state.n_theta_bins - 1)
    p_bin = int((phi + math.pi) / (2 * math.pi) * state.n_phi_bins)
    p_bin = min(p_bin, state.n_phi_bins - 1)
    state.coverage_bins.add((t_bin, p_bin))
    state.coverage_pct = len(state.coverage_bins) / state.total_bins * 100.0


def fit_ellipsoid(samples):
    """Fit an ellipsoid to magnetometer samples using algebraic method.

    Fits: Ax^2 + By^2 + Cz^2 + 2Dxy + 2Exz + 2Fyz + 2Gx + 2Hy + 2Iz = 1
    Returns: (hard_iron, soft_iron, fit_error)
      hard_iron: (3,) center offset
      soft_iron: (3,3) correction matrix such that
                 m_corrected = soft_iron @ (m_raw - hard_iron)
      fit_error: relative radius standard deviation (lower is better)
    """
    s = samples.T  # (3, N)

    # Design matrix (N, 9)
    D = numpy.array([
        s[0] ** 2, s[1] ** 2, s[2] ** 2,
        2 * s[0] * s[1], 2 * s[0] * s[2], 2 * s[1] * s[2],
        2 * s[0], 2 * s[1], 2 * s[2],
    ]).T

    # Solve D @ v = 1 in least-squares sense
    ones = numpy.ones(len(samples))
    v, _, _, _ = numpy.linalg.lstsq(D, ones, rcond=None)

    # Reconstruct the quadratic form matrix
    A_mat = numpy.array([
        [v[0], v[3], v[4]],
        [v[3], v[1], v[5]],
        [v[4], v[5], v[2]],
    ])
    b_vec = numpy.array([v[6], v[7], v[8]])

    # Hard iron = center of the ellipsoid
    hard_iron = -numpy.linalg.solve(A_mat, b_vec)

    # Translate to center: the value at center
    val = 1.0 + b_vec @ hard_iron

    # Normalize so the centered ellipsoid equation is x^T A_norm x = 1
    A_norm = A_mat / val

    # Eigendecompose to get the square-root (soft iron correction)
    eigenvalues, eigenvectors = numpy.linalg.eigh(A_norm)

    # Ensure positive eigenvalues (valid ellipsoid)
    eigenvalues = numpy.abs(eigenvalues)

    # Soft iron matrix: maps ellipsoid to unit sphere
    soft_iron = (eigenvectors
                 @ numpy.diag(numpy.sqrt(eigenvalues))
                 @ eigenvectors.T)

    # Compute fit quality: corrected samples should lie on a sphere
    corrected = (soft_iron @ (samples.T - hard_iron.reshape(3, 1))).T
    radii = numpy.linalg.norm(corrected, axis=1)
    mean_radius = numpy.mean(radii)
    fit_error = float(numpy.std(radii) / mean_radius)

    # Normalize soft_iron so corrected data has the mean field magnitude
    soft_iron = soft_iron / mean_radius

    return hard_iron, soft_iron, fit_error


class MagCalibratorNode(Node):

    def __init__(self):
        super().__init__('mag_calibrator_node')

        self.declare_parameter('web_port', 8082)
        self.declare_parameter('min_samples', 200)
        self.declare_parameter('collect_interval', 0.05)
        self.declare_parameter('save_path', '~/imu_calibration.yaml')

        self.state = MagCalibratorState()
        self.state.min_samples = self.get_parameter('min_samples').value
        self.state.collect_interval = self.get_parameter(
            'collect_interval').value
        self.state.save_path = os.path.expanduser(
            self.get_parameter('save_path').value)

        mag_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.subscription = self.create_subscription(
            MagneticField, '/imu/mag',
            self._mag_callback, mag_qos)

        port = self.get_parameter('web_port').value
        MagCalibrationHTTPHandler.state = self.state
        MagCalibrationHTTPHandler.node_logger = self.get_logger()
        self._http_server = ThreadedHTTPServer(
            ('0.0.0.0', port), MagCalibrationHTTPHandler)
        self._http_thread = threading.Thread(
            target=self._http_server.serve_forever, daemon=True)
        self._http_thread.start()

        self.get_logger().info(
            f'Magnetometer Calibration UI at http://0.0.0.0:{port}')

    def _mag_callback(self, msg):
        mag = [
            msg.magnetic_field.x,
            msg.magnetic_field.y,
            msg.magnetic_field.z,
        ]
        now = time.monotonic()

        with self.state.lock:
            self.state.latest_mag = mag
            self.state.sample_count += 1
            if (self.state.collecting
                    and now - self.state.last_collect_time
                    >= self.state.collect_interval):
                sample = numpy.array(mag)
                self.state.mag_samples.append(sample)
                self.state.capture_count += 1
                self.state.last_collect_time = now
                _update_coverage(self.state, sample)

    def destroy_node(self):
        self._http_server.shutdown()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MagCalibratorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
