import collections
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import numpy
import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from scipy.optimize import least_squares
from sensor_msgs.msg import Imu


GRAVITY = 9.80665

IMU_CALIBRATION_HTML = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>IMU Calibration</title>
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
  .reading-row { display: flex; gap: 24px; margin-bottom: 12px; }
  .reading-group { flex: 1; }
  .reading-group .rg-title { font-size: 12px; font-weight: 600; color: #888;
                             margin-bottom: 6px; }
  .reading-vals { font-family: 'SF Mono', 'Fira Code', monospace; font-size: 16px;
                  font-weight: 600; line-height: 1.6; }
  .reading-vals .axis { color: #888; font-size: 12px; margin-right: 4px; }
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
    .reading-row { flex-direction: column; gap: 12px; }
  }
</style></head><body>
<div class="container">
<header>
  <h1>IMU Accel/Gyro Calibration</h1>
  <span class="badge" id="liveBadge">Connecting...</span>
</header>
<div class="stats">
  <div class="stat-card">
    <div class="label">Stationary</div>
    <div class="value muted" id="stationary">--</div>
  </div>
  <div class="stat-card">
    <div class="label">Accel Variance</div>
    <div class="value muted" id="variance">--</div>
  </div>
  <div class="stat-card">
    <div class="label">Samples Captured</div>
    <div class="value" id="count">0</div>
  </div>
  <div class="stat-card">
    <div class="label">Residual Error</div>
    <div class="value muted" id="error">--</div>
  </div>
</div>
<div class="readings">
  <h2>Live Sensor Readings</h2>
  <div class="reading-row">
    <div class="reading-group">
      <div class="rg-title">Accelerometer (m/s&sup2;)</div>
      <div class="reading-vals" id="accel">
        <span class="axis">X:</span> --<br>
        <span class="axis">Y:</span> --<br>
        <span class="axis">Z:</span> --
      </div>
    </div>
    <div class="reading-group">
      <div class="rg-title">Gyroscope (rad/s)</div>
      <div class="reading-vals" id="gyro">
        <span class="axis">X:</span> --<br>
        <span class="axis">Y:</span> --<br>
        <span class="axis">Z:</span> --
      </div>
    </div>
  </div>
</div>
<div class="guidance">
  Place the IMU in various stationary orientations. Hold still until the
  <strong>Stationary</strong> indicator turns green, then click
  <strong>Capture Orientation</strong>. Capture at least 6 different
  orientations (12+ recommended) covering all axes in both directions.
</div>
<div class="controls">
  <button class="btn-capture" id="btnCapture" onclick="doCapture()" disabled>
    Capture Orientation</button>
  <button class="btn-calibrate" id="btnCalibrate" onclick="doCalibrate()" disabled>
    Calibrate</button>
  <button class="btn-save" id="btnSave" onclick="doSave()" disabled>
    Save YAML</button>
  <button class="btn-reset" id="btnReset" onclick="doReset()">Reset All</button>
</div>
<div class="log-wrap">
  <div class="log-title">Activity Log</div>
  <div id="log">Waiting for IMU data...</div>
</div>
</div>
<script>
const logEl = document.getElementById('log');
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
function fmtV(v) {
  return '<span class="axis">X:</span> ' + v[0].toFixed(4) + '<br>' +
         '<span class="axis">Y:</span> ' + v[1].toFixed(4) + '<br>' +
         '<span class="axis">Z:</span> ' + v[2].toFixed(4);
}
function doCapture() {
  post('/capture').then(d => {
    if (d.error) log(d.error, 'err');
    else log('Captured orientation ' + d.captured, 'ok');
  }).catch(() => log('Request failed', 'err'));
}
function doCalibrate() {
  log('Running calibration...', 'info');
  document.getElementById('btnCalibrate').disabled = true;
  post('/calibrate').then(d => {
    if (d.error) log(d.error, 'err');
    else log('Calibrated! Residual: ' + d.residual_error.toFixed(6) +
             ' m/s\\u00B2, Gyro bias: [' +
             d.gyro_bias.map(v => v.toFixed(6)).join(', ') + '] rad/s', 'ok');
  }).catch(() => log('Request failed', 'err'));
}
function doSave() {
  post('/save').then(d => {
    if (d.error) log(d.error, 'err');
    else log('Saved to ' + d.saved_to, 'ok');
  }).catch(() => log('Request failed', 'err'));
}
function doReset() {
  if (!confirm('Discard all captured data and calibration results?')) return;
  post('/reset').then(() => log('Reset complete', 'info'))
    .catch(() => log('Request failed', 'err'));
}
setInterval(() => {
  fetch('/status').then(r => r.json()).then(d => {
    const statEl = document.getElementById('stationary');
    statEl.textContent = d.is_stationary ? 'Yes' : 'No';
    statEl.className = 'value' + (d.is_stationary ? ' green' : ' muted');
    const varEl = document.getElementById('variance');
    varEl.textContent = d.accel_variance.toFixed(6);
    varEl.className = 'value' + (d.is_stationary ? ' green' : '');
    document.getElementById('count').textContent = d.capture_count;
    const errEl = document.getElementById('error');
    if (d.residual_error !== null) {
      errEl.textContent = d.residual_error.toFixed(6) + ' m/s\\u00B2';
      errEl.className = 'value' + (d.residual_error < 0.05 ? ' green' : '');
    } else { errEl.textContent = '--'; errEl.className = 'value muted'; }
    if (d.latest_accel)
      document.getElementById('accel').innerHTML = fmtV(d.latest_accel);
    if (d.latest_gyro)
      document.getElementById('gyro').innerHTML = fmtV(d.latest_gyro);
    document.getElementById('btnCapture').disabled = !d.is_stationary;
    document.getElementById('btnCalibrate').disabled = d.capture_count < 6;
    document.getElementById('btnSave').disabled = !d.calibrated;
    const badge = document.getElementById('liveBadge');
    if (d.sample_count > 0) { badge.textContent = 'Live';
      badge.className = 'badge live'; }
  }).catch(() => {});
}, 500);
</script></body></html>"""


class ImuCalibratorState:
    """Thread-safe shared state for IMU calibration."""

    def __init__(self):
        self.lock = threading.Lock()
        # Live data
        self.latest_accel = [0.0, 0.0, 0.0]
        self.latest_gyro = [0.0, 0.0, 0.0]
        self.sample_count = 0
        # Rolling windows for stationarity detection
        self.accel_window = collections.deque(maxlen=100)
        self.gyro_window = collections.deque(maxlen=100)
        self.is_stationary = False
        self.accel_variance = 1.0
        # Captured orientation samples
        self.accel_samples = []
        self.gyro_samples = []
        self.capture_count = 0
        # Calibration results
        self.calibrated = False
        self.accel_matrix = None
        self.accel_bias = None
        self.gyro_bias = None
        self.residual_error = None
        # Config
        self.stationarity_threshold = 0.005
        self.save_path = os.path.expanduser('~/imu_calibration.yaml')


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class ImuCalibrationHTTPHandler(BaseHTTPRequestHandler):
    state = None
    node_logger = None

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == '/':
            self._serve_html()
        elif self.path == '/status':
            self._serve_status()
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == '/capture':
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
        content = IMU_CALIBRATION_HTML.encode()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.send_header('Content-Length', str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _serve_status(self):
        with self.state.lock:
            data = {
                'is_stationary': self.state.is_stationary,
                'accel_variance': self.state.accel_variance,
                'latest_accel': list(self.state.latest_accel),
                'latest_gyro': list(self.state.latest_gyro),
                'capture_count': self.state.capture_count,
                'calibrated': self.state.calibrated,
                'residual_error': self.state.residual_error,
                'sample_count': self.state.sample_count,
            }
        self._json_response(200, data)

    def _handle_capture(self):
        with self.state.lock:
            if not self.state.is_stationary:
                self._json_response(
                    400, {'error': 'Sensor is not stationary'})
                return
            if len(self.state.accel_window) < 20:
                self._json_response(
                    400, {'error': 'Not enough data in window yet'})
                return
            accel_mean = numpy.mean(
                list(self.state.accel_window), axis=0)
            gyro_mean = numpy.mean(
                list(self.state.gyro_window), axis=0)
            self.state.accel_samples.append(accel_mean)
            self.state.gyro_samples.append(gyro_mean)
            self.state.capture_count += 1
            count = self.state.capture_count

        if self.node_logger:
            self.node_logger.info(
                f'Captured orientation {count}: '
                f'accel=[{accel_mean[0]:.4f}, {accel_mean[1]:.4f}, '
                f'{accel_mean[2]:.4f}]')
        self._json_response(200, {'captured': count})

    def _handle_calibrate(self):
        with self.state.lock:
            if self.state.capture_count < 6:
                self._json_response(400, {
                    'error': f'Need at least 6 captures, have '
                             f'{self.state.capture_count}'})
                return
            accel_samples = numpy.array(self.state.accel_samples)
            gyro_samples = numpy.array(self.state.gyro_samples)

        if self.node_logger:
            self.node_logger.info(
                f'Running 12-param accel calibration with '
                f'{len(accel_samples)} samples...')

        accel_matrix, accel_bias, residual = _calibrate_accel_12param(
            accel_samples)
        gyro_bias = numpy.mean(gyro_samples, axis=0)

        with self.state.lock:
            self.state.calibrated = True
            self.state.accel_matrix = accel_matrix
            self.state.accel_bias = accel_bias
            self.state.gyro_bias = gyro_bias
            self.state.residual_error = residual

        if self.node_logger:
            self.node_logger.info(
                f'Calibration done. Residual: {residual:.6f} m/s^2')

        self._json_response(200, {
            'residual_error': residual,
            'accel_matrix': accel_matrix.tolist(),
            'accel_bias': accel_bias.tolist(),
            'gyro_bias': gyro_bias.tolist(),
        })

    def _handle_save(self):
        with self.state.lock:
            if not self.state.calibrated:
                self._json_response(400, {'error': 'Not calibrated yet'})
                return
            M = self.state.accel_matrix.copy()
            b = self.state.accel_bias.copy()
            gb = self.state.gyro_bias.copy()
            save_path = self.state.save_path

        # Merge with existing file if present
        calibration = {}
        if os.path.exists(save_path):
            with open(save_path) as f:
                existing = yaml.safe_load(f)
                if isinstance(existing, dict):
                    calibration = existing

        calibration['accelerometer'] = {
            'bias': b.tolist(),
            'matrix': M.tolist(),
            'scale': numpy.diag(M).tolist(),
        }
        calibration['gyroscope'] = {
            'bias': gb.tolist(),
        }

        with open(save_path, 'w') as f:
            yaml.dump(calibration, f, default_flow_style=False)

        if self.node_logger:
            self.node_logger.info(f'IMU calibration saved to {save_path}')
        self._json_response(200, {'saved_to': save_path})

    def _handle_reset(self):
        with self.state.lock:
            self.state.accel_samples.clear()
            self.state.gyro_samples.clear()
            self.state.capture_count = 0
            self.state.calibrated = False
            self.state.accel_matrix = None
            self.state.accel_bias = None
            self.state.gyro_bias = None
            self.state.residual_error = None

        if self.node_logger:
            self.node_logger.info('IMU calibration data reset')
        self._json_response(200, {'status': 'reset'})

    def _json_response(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _calibrate_accel_12param(samples):
    """Fit a 12-parameter accelerometer model.

    Model: a_corrected = M @ (a_raw - b)
    where M is a 3x3 matrix (scale, cross-axis, misalignment)
    and b is a 3-vector bias.

    For stationary readings: ||a_corrected|| should equal g.
    """
    N = len(samples)

    def residuals(params):
        M = params[:9].reshape(3, 3)
        b = params[9:12]
        res = numpy.empty(N)
        for i in range(N):
            corrected = M @ (samples[i] - b)
            res[i] = numpy.linalg.norm(corrected) - GRAVITY
        return res

    # Initial guess: identity matrix, zero bias
    x0 = numpy.zeros(12)
    x0[:9] = numpy.eye(3).flatten()

    result = least_squares(residuals, x0, method='lm')

    M = result.x[:9].reshape(3, 3)
    b = result.x[9:12]
    residual_rms = numpy.sqrt(numpy.mean(result.fun ** 2))

    return M, b, float(residual_rms)


class ImuCalibratorNode(Node):

    def __init__(self):
        super().__init__('imu_calibrator_node')

        self.declare_parameter('web_port', 8081)
        self.declare_parameter('stationarity_threshold', 0.005)
        self.declare_parameter('min_samples', 6)
        self.declare_parameter('save_path', '~/imu_calibration.yaml')

        self.state = ImuCalibratorState()
        self.state.stationarity_threshold = self.get_parameter(
            'stationarity_threshold').value
        self.state.save_path = os.path.expanduser(
            self.get_parameter('save_path').value)

        imu_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.subscription = self.create_subscription(
            Imu, '/imu/data_raw',
            self._imu_callback, imu_qos)

        port = self.get_parameter('web_port').value
        ImuCalibrationHTTPHandler.state = self.state
        ImuCalibrationHTTPHandler.node_logger = self.get_logger()
        self._http_server = ThreadedHTTPServer(
            ('0.0.0.0', port), ImuCalibrationHTTPHandler)
        self._http_thread = threading.Thread(
            target=self._http_server.serve_forever, daemon=True)
        self._http_thread.start()

        self.get_logger().info(
            f'IMU Calibration UI at http://0.0.0.0:{port}')

    def _imu_callback(self, msg):
        accel = [
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
        ]
        gyro = [
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
        ]
        mag = numpy.sqrt(accel[0] ** 2 + accel[1] ** 2 + accel[2] ** 2)

        with self.state.lock:
            self.state.latest_accel = accel
            self.state.latest_gyro = gyro
            self.state.sample_count += 1
            self.state.accel_window.append(accel)
            self.state.gyro_window.append(gyro)
            # Compute accel magnitude variance over window
            if len(self.state.accel_window) >= 20:
                mags = [
                    (a[0] ** 2 + a[1] ** 2 + a[2] ** 2)
                    for a in self.state.accel_window
                ]
                self.state.accel_variance = float(numpy.var(mags))
                self.state.is_stationary = (
                    self.state.accel_variance
                    < self.state.stationarity_threshold)
            else:
                self.state.accel_variance = 1.0
                self.state.is_stationary = False

    def destroy_node(self):
        self._http_server.shutdown()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ImuCalibratorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
