import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import cv2
import numpy
import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image


CALIBRATION_HTML = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Camera Calibration</title>
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
  .stream-wrap { background: #fff; border-radius: 12px; overflow: hidden;
                 box-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06); }
  #stream { width: 100%; display: block; }
  .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;
           margin: 16px 0; }
  .stat-card { background: #fff; border-radius: 10px; padding: 14px 16px;
               box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
  .stat-card .label { font-size: 11px; font-weight: 600; text-transform: uppercase;
                      letter-spacing: 0.5px; color: #888; margin-bottom: 4px; }
  .stat-card .value { font-size: 22px; font-weight: 700; }
  .stat-card .value.green { color: #16a34a; }
  .stat-card .value.muted { color: #bbb; }
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
  }
</style></head><body>
<div class="container">
<header>
  <h1>Camera Calibration</h1>
  <span class="badge" id="liveBadge">Connecting...</span>
</header>
<div class="stream-wrap">
  <img id="stream" src="/stream" alt="Loading camera feed..." />
</div>
<div class="stats">
  <div class="stat-card">
    <div class="label">Corners Detected</div>
    <div class="value muted" id="detected">--</div>
  </div>
  <div class="stat-card">
    <div class="label">Frames Captured</div>
    <div class="value" id="count">0</div>
  </div>
  <div class="stat-card">
    <div class="label">RMS Error</div>
    <div class="value muted" id="error">--</div>
  </div>
  <div class="stat-card">
    <div class="label">Camera FPS</div>
    <div class="value muted" id="fps">--</div>
  </div>
</div>
<div class="controls">
  <button class="btn-capture" id="btnCapture" onclick="doCapture()" disabled>
    Capture Frame</button>
  <button class="btn-calibrate" id="btnCalibrate" onclick="doCalibrate()" disabled>
    Calibrate</button>
  <button class="btn-save" id="btnSave" onclick="doSave()" disabled>
    Save YAML</button>
  <button class="btn-reset" id="btnReset" onclick="doReset()">Reset All</button>
</div>
<div class="log-wrap">
  <div class="log-title">Activity Log</div>
  <div id="log">Waiting for camera frames...</div>
</div>
</div>
<script>
const logEl = document.getElementById('log');
let lastFrameCount = 0, lastTime = Date.now();
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
function doCapture() {
  post('/capture').then(d => {
    if (d.error) log(d.error, 'err');
    else log('Captured frame ' + d.captured, 'ok');
  }).catch(() => log('Request failed', 'err'));
}
function doCalibrate() {
  log('Running calibration...', 'info');
  document.getElementById('btnCalibrate').disabled = true;
  post('/calibrate').then(d => {
    if (d.error) log(d.error, 'err');
    else log('Calibrated! RMS error: ' + d.reprojection_error.toFixed(4) + ' px', 'ok');
  }).catch(() => log('Request failed', 'err'));
}
function doSave() {
  post('/save').then(d => {
    if (d.error) log(d.error, 'err');
    else log('Saved to ' + d.saved_to, 'ok');
  }).catch(() => log('Request failed', 'err'));
}
function doReset() {
  if (!confirm('Discard all captured frames and calibration data?')) return;
  post('/reset').then(() => log('Reset complete', 'info'))
    .catch(() => log('Request failed', 'err'));
}
setInterval(() => {
  fetch('/status').then(r => r.json()).then(d => {
    const det = document.getElementById('detected');
    det.textContent = d.corners_found ? 'Yes' : 'No';
    det.className = 'value' + (d.corners_found ? ' green' : ' muted');
    document.getElementById('count').textContent = d.capture_count;
    const errEl = document.getElementById('error');
    if (d.reprojection_error !== null) {
      errEl.textContent = d.reprojection_error.toFixed(4) + ' px';
      errEl.className = 'value' + (d.reprojection_error < 0.5 ? ' green' : '');
    } else { errEl.textContent = '--'; errEl.className = 'value muted'; }
    const now = Date.now();
    const dt = (now - lastTime) / 1000;
    if (dt > 0.4) {
      const fps = (d.frame_count - lastFrameCount) / dt;
      document.getElementById('fps').textContent = fps > 0 ? fps.toFixed(1) : '--';
      document.getElementById('fps').className = 'value' + (fps > 0 ? '' : ' muted');
      lastFrameCount = d.frame_count; lastTime = now;
    }
    document.getElementById('btnCapture').disabled = !d.corners_found;
    document.getElementById('btnCalibrate').disabled = d.capture_count < 5;
    document.getElementById('btnSave').disabled = !d.calibrated;
    const badge = document.getElementById('liveBadge');
    if (d.frame_count > 0) { badge.textContent = 'Live';
      badge.className = 'badge live'; }
  }).catch(() => {});
}, 500);
</script></body></html>"""


class CalibratorState:
    """Thread-safe shared state for calibration."""

    def __init__(self):
        self.lock = threading.Lock()
        self.frame_condition = threading.Condition(self.lock)
        # Raw BGR frame from camera (set by callback)
        self.bgr_frame = None
        self.frame_count = 0
        # Pre-encoded JPEG for the stream (set by detection thread)
        self.stream_jpeg = None
        self.stream_seq = 0
        # The MJPEG stream reads whichever is freshest
        self.corners_found = False
        self.current_corners = None
        self.image_width = 1280
        self.image_height = 720
        # Calibration captures
        self.captured_image_points = []
        self.captured_object_points = []
        self.capture_count = 0
        # Calibration results
        self.calibrated = False
        self.camera_matrix = None
        self.dist_coeffs = None
        self.reprojection_error = None
        # Config
        self.pattern_cols = 7
        self.pattern_rows = 9
        self.square_size = 0.020
        self.detection_scale = 0.25
        self.jpeg_quality = 70
        self.save_path = os.path.expanduser('~/camera_calibration.yaml')


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class CalibrationHTTPHandler(BaseHTTPRequestHandler):
    state = None
    node_logger = None

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == '/':
            self._serve_html()
        elif self.path == '/stream':
            self._serve_mjpeg_stream()
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
        content = CALIBRATION_HTML.encode()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.send_header('Content-Length', str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _serve_mjpeg_stream(self):
        self.send_response(200)
        self.send_header('Content-Type',
                         'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.connection.settimeout(5.0)
        last_seq = -1
        try:
            while True:
                with self.state.frame_condition:
                    # Wait up to 500ms for a new frame instead of polling
                    self.state.frame_condition.wait_for(
                        lambda: self.state.stream_seq != last_seq
                        or self.state.stream_jpeg is not None,
                        timeout=0.5)
                    jpeg = self.state.stream_jpeg
                    last_seq = self.state.stream_seq
                if jpeg is None:
                    continue
                self.wfile.write(b'--frame\r\n')
                self.wfile.write(b'Content-Type: image/jpeg\r\n')
                self.wfile.write(
                    f'Content-Length: {len(jpeg)}\r\n'.encode())
                self.wfile.write(b'\r\n')
                self.wfile.write(jpeg)
                self.wfile.write(b'\r\n')
        except (BrokenPipeError, ConnectionResetError,
                ConnectionAbortedError, OSError, TimeoutError):
            pass

    def _serve_status(self):
        with self.state.lock:
            data = {
                'corners_found': self.state.corners_found,
                'capture_count': self.state.capture_count,
                'frame_count': self.state.frame_count,
                'calibrated': self.state.calibrated,
                'reprojection_error': self.state.reprojection_error,
            }
        self._json_response(200, data)

    def _handle_capture(self):
        with self.state.lock:
            if not self.state.corners_found or self.state.current_corners is None:
                self._json_response(400, {'error': 'No checkerboard detected'})
                return
            corners = self.state.current_corners.copy()
            pcols = self.state.pattern_cols
            prows = self.state.pattern_rows
            sq = self.state.square_size

        objp = numpy.zeros((prows * pcols, 3), numpy.float32)
        objp[:, :2] = numpy.mgrid[
            0:pcols, 0:prows].T.reshape(-1, 2) * sq

        with self.state.lock:
            self.state.captured_object_points.append(objp)
            self.state.captured_image_points.append(corners)
            self.state.capture_count += 1
            count = self.state.capture_count

        if self.node_logger:
            self.node_logger.info(f'Captured calibration frame {count}')
        self._json_response(200, {'captured': count})

    def _handle_calibrate(self):
        with self.state.lock:
            if self.state.capture_count < 5:
                self._json_response(400, {
                    'error': f'Need at least 5 captures, have '
                             f'{self.state.capture_count}'})
                return
            obj_pts = list(self.state.captured_object_points)
            img_pts = list(self.state.captured_image_points)
            img_size = (self.state.image_width, self.state.image_height)

        if self.node_logger:
            self.node_logger.info(
                f'Running calibration with {len(obj_pts)} frames...')

        ret, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
            obj_pts, img_pts, img_size, None, None)

        with self.state.lock:
            self.state.calibrated = True
            self.state.camera_matrix = camera_matrix
            self.state.dist_coeffs = dist_coeffs
            self.state.reprojection_error = ret

        if self.node_logger:
            self.node_logger.info(
                f'Calibration done. RMS error: {ret:.4f} px')

        self._json_response(200, {
            'reprojection_error': ret,
            'camera_matrix': camera_matrix.tolist(),
            'dist_coeffs': dist_coeffs.flatten().tolist(),
        })

    def _handle_save(self):
        with self.state.lock:
            if not self.state.calibrated:
                self._json_response(400, {'error': 'Not calibrated yet'})
                return
            K = self.state.camera_matrix.copy()
            D = self.state.dist_coeffs.copy()
            w = self.state.image_width
            h = self.state.image_height
            save_path = self.state.save_path

        P = numpy.zeros((3, 4))
        P[:3, :3] = K

        calibration = {
            'image_width': w,
            'image_height': h,
            'camera_name': 'imx477',
            'camera_matrix': {
                'rows': 3, 'cols': 3,
                'data': K.flatten().tolist(),
            },
            'distortion_model': 'plumb_bob',
            'distortion_coefficients': {
                'rows': 1, 'cols': 5,
                'data': D.flatten().tolist(),
            },
            'rectification_matrix': {
                'rows': 3, 'cols': 3,
                'data': numpy.eye(3).flatten().tolist(),
            },
            'projection_matrix': {
                'rows': 3, 'cols': 4,
                'data': P.flatten().tolist(),
            },
        }

        with open(save_path, 'w') as f:
            yaml.dump(calibration, f, default_flow_style=False)

        if self.node_logger:
            self.node_logger.info(f'Calibration saved to {save_path}')
        self._json_response(200, {'saved_to': save_path})

    def _handle_reset(self):
        with self.state.lock:
            self.state.captured_image_points.clear()
            self.state.captured_object_points.clear()
            self.state.capture_count = 0
            self.state.calibrated = False
            self.state.camera_matrix = None
            self.state.dist_coeffs = None
            self.state.reprojection_error = None

        if self.node_logger:
            self.node_logger.info('Calibration data reset')
        self._json_response(200, {'status': 'reset'})

    def _json_response(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class CameraCalibratorNode(Node):

    def __init__(self):
        super().__init__('camera_calibrator_node')

        self.declare_parameter('web_port', 8080)
        self.declare_parameter('pattern_columns', 7)
        self.declare_parameter('pattern_rows', 9)
        self.declare_parameter('square_size', 0.020)
        self.declare_parameter('detection_scale', 0.5)
        self.declare_parameter('jpeg_quality', 70)
        self.declare_parameter('save_path', '~/camera_calibration.yaml')

        self.state = CalibratorState()
        self.state.pattern_cols = self.get_parameter(
            'pattern_columns').value
        self.state.pattern_rows = self.get_parameter(
            'pattern_rows').value
        self.state.square_size = self.get_parameter(
            'square_size').value
        self.state.detection_scale = self.get_parameter(
            'detection_scale').value
        self.state.jpeg_quality = self.get_parameter(
            'jpeg_quality').value
        self.state.save_path = os.path.expanduser(
            self.get_parameter('save_path').value)

        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.subscription = self.create_subscription(
            Image, '/camera/image_raw',
            self._image_callback, image_qos)

        # Single detection thread — also produces overlay JPEG
        self._detection_thread = threading.Thread(
            target=self._detection_loop, daemon=True)
        self._detection_thread.start()

        port = self.get_parameter('web_port').value
        CalibrationHTTPHandler.state = self.state
        CalibrationHTTPHandler.node_logger = self.get_logger()
        self._http_server = ThreadedHTTPServer(
            ('0.0.0.0', port), CalibrationHTTPHandler)
        self._http_thread = threading.Thread(
            target=self._http_server.serve_forever, daemon=True)
        self._http_thread.start()

        self.get_logger().info(
            f'Calibration UI at http://0.0.0.0:{port}')
        self.get_logger().info(
            f'Pattern: {self.state.pattern_cols}x{self.state.pattern_rows}'
            f' inner corners, {self.state.square_size * 1000:.0f}mm squares')

    def _image_callback(self, msg):
        # Convert raw ROS Image to BGR, encode JPEG for stream, share frame
        if self.state.frame_count == 0:
            self.get_logger().info(
                f'First frame: encoding={msg.encoding}, '
                f'size={msg.width}x{msg.height}, '
                f'step={msg.step}, data_len={len(msg.data)}')
        img = numpy.frombuffer(msg.data, dtype=numpy.uint8).reshape(
            msg.height, msg.width, 3)
        if msg.encoding in ('bgr8', 'BGR888'):
            bgr = img
        else:
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode(
            '.jpg', bgr,
            [cv2.IMWRITE_JPEG_QUALITY, self.state.jpeg_quality])
        jpeg = buf.tobytes()
        with self.state.frame_condition:
            self.state.bgr_frame = bgr
            self.state.frame_count += 1
            # Only update stream JPEG if detection hasn't provided an
            # overlay for this frame (keeps stream running at camera rate)
            if not self.state.corners_found:
                self.state.stream_jpeg = jpeg
                self.state.stream_seq += 1
            self.state.frame_condition.notify_all()

    def _detection_loop(self):
        """Detection loop that waits for new frames, runs corner
        detection, and produces a JPEG for the stream."""
        pattern = (self.state.pattern_cols, self.state.pattern_rows)
        flags = (cv2.CALIB_CB_ADAPTIVE_THRESH
                 | cv2.CALIB_CB_NORMALIZE_IMAGE)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    30, 0.001)
        scale = self.state.detection_scale
        quality = self.state.jpeg_quality
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        last_frame_id = -1

        while rclpy.ok():
            try:
                # Wait for a new frame instead of polling
                with self.state.frame_condition:
                    self.state.frame_condition.wait_for(
                        lambda: self.state.frame_count != last_frame_id,
                        timeout=1.0)
                    frame_bgr = self.state.bgr_frame
                    fid = self.state.frame_count
                if frame_bgr is None or fid == last_frame_id:
                    continue
                last_frame_id = fid

                h, w = frame_bgr.shape[:2]

                # Detect on downscaled grayscale
                small = cv2.resize(frame_bgr, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_LINEAR)
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

                found, corners_small = cv2.findChessboardCorners(
                    gray, pattern, flags)

                if fid % 30 == 0:
                    self.get_logger().info(
                        f'Detection: frame={fid}, '
                        f'input={w}x{h}, '
                        f'scaled={gray.shape[1]}x{gray.shape[0]}, '
                        f'pattern={pattern}, '
                        f'found={found}')

                corners_full = None
                if found:
                    corners_full = corners_small / scale
                    gray_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                    corners_full = cv2.cornerSubPix(
                        gray_full, corners_full, (11, 11), (-1, -1),
                        criteria)
                    # Draw overlay on the frame
                    annotated = frame_bgr.copy()
                    cv2.drawChessboardCorners(
                        annotated, pattern, corners_full, True)
                    pts = corners_full.reshape(-1, 2)
                    x0, y0 = pts.min(axis=0).astype(int)
                    x1, y1 = pts.max(axis=0).astype(int)
                    pad = 12
                    cv2.rectangle(annotated, (x0 - pad, y0 - pad),
                                  (x1 + pad, y1 + pad), (0, 200, 0), 2)
                    _, buf = cv2.imencode('.jpg', annotated, encode_params)
                    overlay_jpeg = buf.tobytes()

                with self.state.frame_condition:
                    self.state.corners_found = found
                    self.state.current_corners = corners_full
                    self.state.image_width = w
                    self.state.image_height = h
                    # Only push annotated overlay to the stream;
                    # plain frames are handled by _image_callback
                    if found:
                        self.state.stream_jpeg = overlay_jpeg
                        self.state.stream_seq += 1
                    self.state.frame_condition.notify_all()

            except Exception as e:
                self.get_logger().error(
                    f'Detection error: {e}', throttle_duration_sec=5)
                time.sleep(0.5)

    def destroy_node(self):
        self._http_server.shutdown()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraCalibratorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
