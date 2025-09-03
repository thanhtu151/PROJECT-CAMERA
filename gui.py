# three_cam_gui.py — Multi-cam GUI (YOLO + DeepSORT + InsightFace) with JSON output
import os, sys, time, threading, queue, argparse, json
from datetime import datetime
import numpy as np
import cv2

from PySide6.QtCore import Qt, QObject, Signal, QThread
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QCheckBox, QGroupBox, QFormLayout, QLineEdit, QSpinBox,
    QMessageBox, QTextEdit
)

# Reduce OpenCV thread contention on CPU
try:
    cv2.setNumThreads(1)
except Exception:
    pass

from insightface.app import FaceAnalysis
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

FACE_INDEX = "dataset/face_index.npz"
DEFAULT_CFG = "sources.txt"
# đầu file (gần OUT_JSON)
OUT_JSON = "logs/active_tracks.json"
OUT_EVENTS = "logs/track_events.jsonl"

OUT_JSON = "logs/active_tracks.json"

# ---------------------- Utils ----------------------
def load_index(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path} (run build_face_index.py first)")
    z = np.load(path, allow_pickle=True)
    E = z["embeddings"].astype(np.float32)
    n = np.linalg.norm(E, axis=1, keepdims=True); n[n==0] = 1.0
    E = E / n
    NAMES  = z["names"].tolist()
    IDENTS = z["idents"].tolist()
    MSSVS  = z["mssvs"].tolist()
    PHONES = z["phones"].tolist()
    return E, NAMES, IDENTS, MSSVS, PHONES

def make_face_app(cpu=False, det_size=512):
    providers = ['CPUExecutionProvider'] if cpu else ['CUDAExecutionProvider','CPUExecutionProvider']
    app = FaceAnalysis(providers=providers)
    app.prepare(ctx_id=(-1 if cpu else 0), det_size=(det_size, det_size))
    return app

def load_sources_from_file(path):
    srcs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"): continue
            srcs.append(s)
    return srcs

def resolve_sources(cfg_path, extra_sources):
    sources = []
    cfg = cfg_path or (DEFAULT_CFG if os.path.exists(DEFAULT_CFG) else None)
    if cfg:
        if not os.path.exists(cfg):
            raise FileNotFoundError(f"Config not found: {cfg}")
        sources += load_sources_from_file(cfg)
    if extra_sources:
        sources += extra_sources
    return sources

def set_rtsp_env(use_udp=False):
    transport = "udp" if use_udp else "tcp"
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        f"rtsp_transport;{transport}|"
        "stimeout;5000000|max_delay;0|buffer_size;102400|"
        "reorder_queue_size;0|fflags;nobuffer|flags;low_delay"
    )

def open_cap(src, use_udp=False):
    set_rtsp_env(use_udp)
    if isinstance(src, str) and src.startswith(("rtsp://","rtmp://","http://","https://")):
        return cv2.VideoCapture(src, cv2.CAP_FFMPEG)
    return cv2.VideoCapture(int(src))

def put_label(img, text, xy, color=(0,255,255), scale=0.7, thickness=2):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def make_qpix_from_bgr(img):
    if img is None or img.size == 0:
        return QPixmap()
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, w, h, 3*w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)

# ---------------------- Worker Thread ----------------------
class MultiCamWorker(QThread):
    frame_ready = Signal(int, object)   # cam_idx, BGR frame
    status_msg  = Signal(str)
    active_json = Signal(str)           # JSON string of active tracks
    event_line  = Signal(str)      # từng dòng JSON sự kiện (append UI)

    def __init__(self, sources, threshold=0.40, det_size=512, cpu=False,
                 rtsp_udp=False, sticky_id=False, ttl=3.0, relock_margin=0.10,
                 max_age=60, window_title="MultiCam Follow",
                 face_every=4, render_fps=25, high_quality=False):
        super().__init__()
        self.sources = sources
        self.threshold = float(threshold)
        self.det_size = int(det_size)
        self.cpu = bool(cpu)
        self.rtsp_udp = bool(rtsp_udp)
        self.sticky_id = bool(sticky_id)
        self.ttl = float(ttl)
        self.relock_margin = float(relock_margin)
        self.max_age = int(max_age)
        self.window_title = window_title

        # Perf knobs
        self.face_every = max(1, int(face_every))
        self._face_counter = [0]*len(self.sources)
        self.render_interval = 1.0 / max(5, int(render_fps))
        self.last_emit = [0.0]*len(self.sources)
        self.high_quality = bool(high_quality)

        self.stop_flag = False
        self.qs = [queue.Queue(maxsize=1) for _ in sources]
        self.threads = []

    def run(self):
        try:
            os.makedirs(os.path.dirname(OUT_JSON) or ".", exist_ok=True)

            # Load index & face app
            E, NAMES, IDENTS, MSSVS, PHONES = load_index(FACE_INDEX)
            LABELS = [f"{n} [{m}]" for n, m in zip(NAMES, MSSVS)]
            self.status_msg.emit(f"[INDEX] {E.shape[0]} identities loaded")

            face_app = make_face_app(cpu=self.cpu, det_size=self.det_size)
            yolo = YOLO("yolov8s.pt" if self.high_quality else "yolov8n.pt")

            trackers = [
                DeepSort(
                    max_age=max(90, self.max_age),
                    n_init=1,
                    nms_max_overlap=1.0,
                    max_cosine_distance=0.35
                )
                for _ in self.sources
            ]

            # Reader threads
            def reader_thread(cam_id, src, q, use_udp):
                cap = open_cap(src, use_udp=use_udp)
                last_ok = time.time()
                while not self.stop_flag:
                    ok, frame = cap.read()
                    if not ok:
                        if isinstance(src, str) and src.startswith("rtsp://"):
                            if time.time() - last_ok > 1.0:
                                self.status_msg.emit(f"[Cam{cam_id}] reconnecting ...")
                                cap.release(); time.sleep(0.5)
                                cap = open_cap(src, use_udp=use_udp)
                                last_ok = time.time()
                        time.sleep(0.01)
                        continue
                    last_ok = time.time()
                    while not q.empty():
                        try: q.get_nowait()
                        except: break
                    q.put(frame)
                cap.release()

            for i, src in enumerate(self.sources):
                t = threading.Thread(target=reader_thread, args=(i, src, self.qs[i], self.rtsp_udp), daemon=True)
                t.start(); self.threads.append(t)
                self.status_msg.emit(f"[Cam{i}] reader started")

            last_fps_t = time.time()
            frames = [0]*len(self.sources)
            fps = [0.0]*len(self.sources)

            # state dicts
            self.track2id = {}
            self.track_start = {}

            def overlap_face_ratio(tb, fb):
                tx1,ty1,tx2,ty2 = tb
                fx1,fy1,fx2,fy2 = fb
                ix1,iy1 = max(tx1,fx1), max(ty1,fy1)
                ix2,iy2 = min(tx2,fx2), min(ty2,fy2)
                iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
                inter = iw*ih
                farea = max(1, (fx2-fx1)*(fy2-fy1))
                return inter / farea

            def face_center_in_track(tb, fb):
                tx1,ty1,tx2,ty2 = tb
                fx1,fy1,fx2,fy2 = fb
                cx = (fx1 + fx2) / 2.0
                cy = (fy1 + fy2) / 2.0
                return (tx1 <= cx <= tx2) and (ty1 <= cy <= ty2)

            while not self.stop_flag:
                now = time.time()
                # ------- Build an overall list for JSON output each loop -------
                json_items = []

                for ci, src in enumerate(self.sources):
                    frame = None
                    if not self.qs[ci].empty():
                        frame = self.qs[ci].get()
                    if frame is None:
                        if (now - self.last_emit[ci]) >= self.render_interval:
                            self.last_emit[ci] = now
                            blank = np.zeros((360,640,3), dtype=np.uint8)
                            self.frame_ready.emit(ci, blank)
                        continue

                    frames[ci] += 1
                    if now - last_fps_t >= 1.0:
                        fps[ci] = frames[ci] / (now - last_fps_t)
                        frames[ci] = 0

                    H,W = frame.shape[:2]

                    # 1) YOLO person
                    imgsz = 960 if self.high_quality else 640
                    conf  = 0.25
                    iou_nms = 0.45

                    dets_person = []
                    res = yolo.predict(frame, imgsz=imgsz, conf=conf, iou=iou_nms, verbose=False)[0]
                    if len(res.boxes):
                        for b in res.boxes:
                            if int(b.cls[0].item()) != 0:
                                continue
                            x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                            confv = float(b.conf[0].item())
                            dets_person.append(([x1, y1, x2, y2], confv, "person"))

                    # 2) Tracking
                    tracks = trackers[ci].update_tracks(dets_person, frame=frame)

                    # 3) Faces — throttled
                    face_infos = []
                    has_track = any(trk.is_confirmed() and trk.time_since_update == 0 for trk in tracks)
                    do_face = False
                    if has_track:
                        self._face_counter[ci] = (self._face_counter[ci] + 1) % self.face_every
                        do_face = (self._face_counter[ci] == 0)

                    if do_face:
                        faces = face_app.get(frame)
                        for f in faces:
                            fx1,fy1,fx2,fy2 = map(int, f.bbox)
                            emb = getattr(f, "normed_embedding", None)
                            if emb is None:
                                continue
                            sims = E @ emb.astype(np.float32)
                            j = int(np.argmax(sims)); simv = float(sims[j])
                            face_infos.append((fx1,fy1,fx2,fy2, j, simv))

                    # 4) Map face->track (with sticky)
                    for trk in tracks:
                        if not trk.is_confirmed() or trk.time_since_update > 0:
                            continue
                        tx1,ty1,tx2,ty2 = map(int, trk.to_tlbr())
                        tid = trk.track_id
                        tbox = (tx1,ty1,tx2,ty2)
                        key = (ci, tid)

                        # track start time
                        if key not in self.track_start:
                            self.track_start[key] = now

                        cands = []
                        for (fx1,fy1,fx2,fy2,j,simv) in face_infos:
                            fbox = (fx1,fy1,fx2,fy2)
                            of = overlap_face_ratio(tbox, fbox)
                            inside = face_center_in_track(tbox, fbox)
                            if simv >= self.threshold and (inside or of >= 0.5):
                                cands.append((simv, of, j))

                        if cands:
                            cands.sort(key=lambda x: (x[0], x[1]), reverse=True)
                            best_sim, best_of, best_j = cands[0]
                            if key in self.track2id and self.track2id[key].get("locked", False):
                                cur_j = self.track2id[key]["idx"]
                                if best_j != cur_j and best_sim >= (self.threshold + self.relock_margin):
                                    self.track2id[key] = {"idx": best_j, "last_seen": now, "sim": best_sim, "locked": True}
                                else:
                                    self.track2id[key]["last_seen"] = now
                                    self.track2id[key]["sim"] = max(self.track2id[key].get("sim", 0.0), best_sim)
                            else:
                                self.track2id[key] = {"idx": best_j, "last_seen": now, "sim": best_sim, "locked": bool(self.sticky_id)}
                        else:
                            if key in self.track2id:
                                if self.sticky_id:
                                    self.track2id[key]["last_seen"] = now
                                else:
                                    if (now - self.track2id[key]["last_seen"]) > self.ttl:
                                        self.track2id.pop(key, None)

                    # cleanup lost tracks for this cam
                    active_keys = {(ci, trk.track_id) for trk in tracks if trk.is_confirmed()}
                    for k in list(self.track2id.keys()):
                        if k[0] == ci and k not in active_keys:
                            self.track2id.pop(k, None)
                    for k in list(self.track_start.keys()):
                        if k[0] == ci and k not in active_keys:
                            self.track_start.pop(k, None)

                    # 5) Draw + also build json_items
                    for trk in tracks:
                        if not trk.is_confirmed() or trk.time_since_update > 0:
                            continue
                        x1,y1,x2,y2 = map(int, trk.to_tlbr())
                        tid = trk.track_id
                        key = (ci, tid)
                        duration = now - self.track_start.get(key, now)

                        if key in self.track2id:
                            j = self.track2id[key]["idx"]
                            disp = f"{LABELS[j]}" if 0 <= j < len(LABELS) else f"idx{j}"
                            color = (0,255,0) if self.track2id[key].get("locked", False) else (0,200,200)
                            label = f"{disp}"
                            info  = f"ID:{tid} | idx:{j}"
                            simv  = float(self.track2id[key].get("sim", 0.0))
                            locked = bool(self.track2id[key].get("locked", False))
                        else:
                            color = (0,0,255); label = "Unknown"; info = f"ID:{tid}"
                            disp = "Unknown"; simv = 0.0; locked = False

                        # draw
                        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                        put_label(frame, label, (x1, max(0,y1-8)), color=color)
                        put_label(frame, info,  (x1, min(H-5, y2+20)), color=(255,255,255), scale=0.6)

                        # collect json for this track
                        json_items.append({
                            "id": int(tid),
                            "name": disp,
                            "zone": f"Cam{ci}",
                            "duration_sec": round(duration, 2),
                            "idx": int(self.track2id[key]["idx"]) if key in self.track2id else None,
                            "locked": locked,
                            "similarity": round(simv, 3)
                        })

                    # 6) FPS
                    if now - last_fps_t >= 1.0:
                        last_fps_t = now
                    put_label(frame, f"FPS: {fps[ci]:.1f}", (10,25), color=(255,255,0), scale=0.8)

                    # Emit image
                    if (now - self.last_emit[ci]) >= self.render_interval:
                        self.last_emit[ci] = now
                        self.frame_ready.emit(ci, frame)

                # ------- Emit JSON (all cams combined) -------
                payload = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "active_tracks": json_items
                }
                js = json.dumps(payload, ensure_ascii=False, indent=2)
                try:
                    with open(OUT_JSON, "w", encoding="utf-8") as f:
                        f.write(js)
                except Exception as _:
                    pass
                self.active_json.emit(js)

            # end while
        except Exception as e:
            self.status_msg.emit(f"[ERROR] {e!r}")

    def stop(self):
        self.stop_flag = True
        for t in self.threads:
            try: t.join(timeout=1.0)
            except: pass

# ---------------------- GUI ----------------------
class MainWindow(QWidget):
    def __init__(self, args):
        super().__init__()
        self.setWindowTitle("Three-Cam GUI | YOLO + DeepSORT + InsightFace (Optimized)")
        self.resize(1600, 940)

        # Left grid: 2x2 (3 cams + 1 empty)
        self.labels = []
        grid = QGridLayout()
        for i in range(4):
            lbl = QLabel("No Camera")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("background:#202020; color:#aaaaaa; border:1px solid #444;")
            lbl.setMinimumSize(360, 240)
            self.labels.append(lbl)
            grid.addWidget(lbl, i//2, i%2)

        # Controls
        ctrl = QGroupBox("Controls")
        form = QFormLayout()

        self.edt_cfg = QLineEdit(args.config or (DEFAULT_CFG if os.path.exists(DEFAULT_CFG) else ""))
        self.edt_src = QLineEdit("")  # optional extra sources, comma-separated

        self.spin_thr = QLineEdit(str(args.threshold))
        self.spin_det = QSpinBox(); self.spin_det.setRange(256, 1536); self.spin_det.setSingleStep(64)
        self.spin_det.setValue(args.det_size)

        self.chk_cpu  = QCheckBox("Force CPU (InsightFace)"); self.chk_cpu.setChecked(args.cpu)
        self.chk_udp  = QCheckBox("RTSP over UDP");           self.chk_udp.setChecked(args.rtsp_udp)
        self.chk_sticky = QCheckBox("Sticky ID until track lost"); self.chk_sticky.setChecked(args.sticky_id)
        self.chk_hq   = QCheckBox("High quality detect (v8s + imgsz 960)"); self.chk_hq.setChecked(False)

        self.spin_ttl = QLineEdit(str(args.ttl))
        self.spin_relock = QLineEdit(str(args.relock_margin))
        self.spin_maxage = QLineEdit(str(args.max_age))

        self.spin_face_every = QSpinBox(); self.spin_face_every.setRange(1, 10); self.spin_face_every.setValue(4)
        self.spin_render_fps = QSpinBox(); self.spin_render_fps.setRange(5, 60); self.spin_render_fps.setValue(25)

        self.btn_start = QPushButton("Start"); self.btn_stop = QPushButton("Stop")
        self.btn_start.clicked.connect(self.on_start)
        self.btn_stop.clicked.connect(self.on_stop)

        form.addRow("Config file:", self.edt_cfg)
        form.addRow("Extra sources (comma):", self.edt_src)
        form.addRow("Face threshold:", self.spin_thr)
        form.addRow("Face det size:", self.spin_det)
        form.addRow(self.chk_cpu, self.chk_udp)
        form.addRow(self.chk_sticky)
        form.addRow(self.chk_hq)
        form.addRow("TTL (s):", self.spin_ttl)
        form.addRow("Relock margin:", self.spin_relock)
        form.addRow("DeepSORT max_age:", self.spin_maxage)
        form.addRow("Face scan every N frames:", self.spin_face_every)
        form.addRow("Render FPS:", self.spin_render_fps)
        form.addRow(self.btn_start, self.btn_stop)
        ctrl.setLayout(form)

        # JSON Output panel (read-only)
        self.txt_output = QTextEdit()
        self.txt_output.setReadOnly(True)
        self.txt_output.setStyleSheet("background:#0e0e0e; color:#d0ffd0; font-family: Consolas, monospace;")
        self.txt_output.setPlaceholderText("Active tracks JSON will appear here...")

        right = QVBoxLayout()
        right.addWidget(ctrl)
        self.lbl_status = QLabel("")
        right.addWidget(self.lbl_status)
        right.addWidget(self.txt_output, 1)  # Output sits under status

        root = QHBoxLayout(self)
        root.addLayout(grid, 2)
        root.addLayout(right, 1)

        self.worker: MultiCamWorker | None = None

    # ---------- Controls ----------
    def on_start(self):
        if self.worker is not None:
            QMessageBox.information(self, "Info", "Already running.")
            return

        cfg = self.edt_cfg.text().strip() or None
        extra = []
        if self.edt_src.text().strip():
            extra = [s.strip() for s in self.edt_src.text().split(",") if s.strip()]
        try:
            thr = float(self.spin_thr.text().strip())
        except:
            thr = 0.40
        det = int(self.spin_det.value())
        cpu = self.chk_cpu.isChecked()
        udp = self.chk_udp.isChecked()
        sticky = self.chk_sticky.isChecked()
        hq = self.chk_hq.isChecked()
        try: ttl = float(self.spin_ttl.text().strip())
        except: ttl = 3.0
        try: relock = float(self.spin_relock.text().strip())
        except: relock = 0.10
        try: max_age = int(self.spin_maxage.text().strip())
        except: max_age = 60
        face_every = int(self.spin_face_every.value())
        render_fps = int(self.spin_render_fps.value())

        try:
            sources = resolve_sources(cfg, extra)
        except Exception as e:
            QMessageBox.critical(self, "Sources", f"Failed to load sources: {e}")
            return
        if not sources:
            QMessageBox.critical(self, "Sources", "No camera sources")
            return

        self.worker = MultiCamWorker(
            sources=sources, threshold=thr, det_size=det, cpu=cpu,
            rtsp_udp=udp, sticky_id=sticky, ttl=ttl, relock_margin=relock,
            max_age=max_age, window_title="MultiCam Follow",
            face_every=face_every, render_fps=render_fps, high_quality=hq
        )
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.status_msg.connect(self.on_status)
        self.worker.active_json.connect(self.on_json)
        self.worker.start()
        self.lbl_status.setText("[RUNNING]")

    def on_stop(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait(1000)
            self.worker = None
        for lbl in self.labels:
            lbl.setText("No Camera")
            lbl.setPixmap(QPixmap())
        self.lbl_status.setText("[STOPPED]")
        self.txt_output.clear()

    def on_frame(self, cam_idx: int, frame_bgr):
        if cam_idx >= len(self.labels):
            return
        pix = make_qpix_from_bgr(frame_bgr)
        if not pix.isNull():
            self.labels[cam_idx].setPixmap(
                pix.scaled(self.labels[cam_idx].size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        else:
            self.labels[cam_idx].setText("No Signal")

    def on_status(self, msg: str):
        self.lbl_status.setText(msg)

    def on_json(self, js: str):
        # Show latest JSON content in the output panel
        self.txt_output.setPlainText(js)
        self.txt_output.verticalScrollBar().setValue(self.txt_output.verticalScrollBar().maximum())

    def closeEvent(self, ev):
        self.on_stop()
        super().closeEvent(ev)

# ---------------------- CLI wrapper ----------------------
def parse_args():
    ap = argparse.ArgumentParser("Three-Cam GUI (optimized)")
    ap.add_argument("--config", help=f"RTSP/Webcam list file (default '{DEFAULT_CFG}' if exists)")
    ap.add_argument("--threshold", type=float, default=0.40)
    ap.add_argument("--det-size", type=int, default=512)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--rtsp-udp", action="store_true")
    ap.add_argument("--sticky-id", action="store_true")
    ap.add_argument("--ttl", type=float, default=3.0)
    ap.add_argument("--relock-margin", type=float, default=0.10)
    ap.add_argument("--max-age", type=int, default=60)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    app = QApplication(sys.argv)
    w = MainWindow(args)
    w.show()
    sys.exit(app.exec())
