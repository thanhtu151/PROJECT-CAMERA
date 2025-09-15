# three_cam_gui.py — Multi-cam GUI (YOLO + DeepSORT + InsightFace + Manual Label)
# Version: per-camera workers (1 camera = 1 QThread) to utilize multi-core CPU (12 threads)
# - 8 inline RTSP sources (edit DEFAULT_SOURCES)
# - Each camera has its own worker: capture + YOLO + DeepSORT + optional InsightFace
# - Supervisor aggregates JSON from all workers; GUI shows 2×4 grid
# - Cam6 (index 5) auto-rotates 180° on display

import os, sys, time, json, argparse, signal
from datetime import datetime
from typing import List, Optional
import numpy as np
import cv2

from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QCheckBox, QGroupBox, QFormLayout, QLineEdit, QSpinBox,
    QMessageBox, QTextEdit
)

def sigint_handler(sig, frame):
    print("Stopping...")
    app.quit()   # hoặc QApplication.quit()

signal.signal(signal.SIGINT, sigint_handler)

# Optional GPU libs
try:
    import torch
    HAVE_TORCH = True
except Exception:
    torch = None; HAVE_TORCH = False

try:
    import onnxruntime as ort
    HAVE_ORT = True
except Exception:
    ort = None; HAVE_ORT = False

def gpu_info_text(gpu_id:int):
    parts=[]
    if HAVE_TORCH:
        parts.append(f"torch.cuda.is_available={torch.cuda.is_available()}")
        if torch.cuda.is_available() and gpu_id>=0:
            try:
                parts.append(f"torch.device={torch.cuda.get_device_name(gpu_id)}")
            except Exception:
                parts.append("torch.device=?")
    if HAVE_ORT:
        try:
            sessopt = ort.get_device()
            parts.append(f"ort_device={sessopt}")
        except Exception:
            parts.append("ort_device=?")
    return " | ".join(parts) if parts else "no torch/onnxruntime"

# Limit OpenCV CPU threads (avoid oversubscription)
try:
    cv2.setNumThreads(0)
except Exception:
    pass

# -------------- Inline sources (EDIT ME) --------------
DEFAULT_SOURCES: List[str] = [
    "rtsp://admin:123456Tu@10.50.197.21:554/stream2",
    "rtsp://admin:123456Tu@10.50.197.24:554/stream2",
    "rtsp://admin:123456Tu@10.50.197.25:554/stream2",
    "rtsp://admin:123456Tu@10.50.197.29:554/stream2",
    "rtsp://admin:123456Tu@10.50.197.32:554/stream2",
    "rtsp://admin:123456Tu@10.50.197.33:554/stream2",
    "rtsp://admin:123456Tu@10.50.197.40:554/stream2",
    "rtsp://admin:123456Tu@10.50.197.60:554/stream2",
]
# ------------------------------------------------------

OUT_JSON = "logs/active_tracks.json"
FACE_INDEX = "dataset/face_index.npz"

# Optional deps (fail-soft)
try:
    from ultralytics import YOLO
    HAVE_YOLO = True
except Exception as e:
    YOLO = None; HAVE_YOLO = False
    print(f"[WARN] YOLO not available: {e}")
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    HAVE_DEEPSORT = True
except Exception as e:
    DeepSort = None; HAVE_DEEPSORT = False
    print(f"[WARN] DeepSORT not available: {e}")
try:
    from insightface.app import FaceAnalysis
    HAVE_FACE = True
except Exception as e:
    FaceAnalysis = None; HAVE_FACE = False
    print(f"[WARN] InsightFace not available: {e}")

# ---------------- Utils ----------------
def set_rtsp_env(use_udp=False):
    transport = "udp" if use_udp else "tcp"
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        f"rtsp_transport;{transport}|"
        "rtsp_flags;prefer_tcp|"
        "stimeout;5000000|"
        "max_delay;500000|"
        "buffer_size;262144|"
        "reorder_queue_size;0|"
        "probesize;65536|"
        "analyzeduration;0|"
        "fflags;discardcorrupt"
    )

def open_cap(src, use_udp=False):
    set_rtsp_env(use_udp)
    if isinstance(src, str) and src.startswith(("rtsp://","rtmp://","http://","https://")):
        return cv2.VideoCapture(src, cv2.CAP_FFMPEG)
    return cv2.VideoCapture(int(src))

def to_pix(frame_bgr):
    if frame_bgr is None or frame_bgr.size == 0:
        return QPixmap()
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, w, h, 3*w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)

def put_label(img, text, xy, color=(0,255,255), scale=0.7, thickness=2):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

# Face index
def load_index(path):
    if not HAVE_FACE:
        return None
    if not os.path.exists(path):
        print(f"[WARN] face index not found: {path}")
        return None
    z = np.load(path, allow_pickle=True)
    E = z["embeddings"].astype(np.float32)
    n = np.linalg.norm(E, axis=1, keepdims=True); n[n==0]=1.0
    E = E / n
    names = z["names"].tolist()
    mssv  = z["mssvs"].tolist() if "mssvs" in z else [""]*len(names)
    labels = [f"{n} [{m}]".strip() for n,m in zip(names, mssv)]
    return {"E":E, "labels":labels}

# ---------------- Per-camera worker ----------------
class CamWorker(QThread):
    frame_ready = Signal(int, object)  # cam_idx, frame_bgr
    json_ready  = Signal(int, object)  # cam_idx, list_of_tracks
    status      = Signal(int, str)

    def __init__(self, cam_idx:int, src:str, threshold:float, det_size:int, cpu_face:bool,
                 rtsp_udp:bool, sticky:bool, ttl:float, relock_margin:float, max_age:int,
                 high_quality:bool, face_every:int, render_fps:int, face_db:Optional[dict], gpu_id:int, use_half:bool):
        super().__init__()
        from queue import Queue
        self.q = Queue(maxsize=1)
        self.manual_labels = {}   # default, sẽ được gán từ MainWindow

        self.gpu_id   = gpu_id
        self.use_half = use_half and HAVE_TORCH
        self.device   = ("cuda:"+str(gpu_id)) if (HAVE_TORCH and torch.cuda.is_available() and gpu_id>=0) else "cpu"
        self.cam_idx = cam_idx
        self.src = src
        self.threshold = threshold
        self.det_size = det_size
        self.cpu_face = cpu_face
        self.rtsp_udp = rtsp_udp
        self.sticky = sticky
        self.ttl = ttl
        self.relock_margin = relock_margin
        self.max_age = max_age
        self.high_quality = high_quality
        self.face_every = max(1, face_every)
        self.render_interval = 1.0 / max(5, render_fps)
        self.face_db = face_db
        self.stop_flag = False

        self._last_emit = 0.0
        self._face_tick = 0

        # Per-cam states
        self.track2id = {}
        self.track_start = {}

        # Lazy init heavy modules
        self.model = None
        self.tracker = None
        self.face_app = None
        self._kv_idx = 0  # (tuỳ chọn nếu sau muốn quay vòng)

    def reader_loop(self, cap):
        while not self.stop_flag:
            ok, frame = cap.read()
            # Nếu cam_idx == 3 (cam lắp ngược) thì xoay
            if self.cam_idx == 5:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            if not ok:
                self.status.emit(self.cam_idx, "reconnecting…")
                try: cap.release()
                except: pass
                time.sleep(0.3)
                cap = open_cap(self.src, use_udp=self.rtsp_udp)
                continue
            if not self.q.empty():
                try: self.q.get_nowait()
                except: pass
            try:
                self.q.put(frame, block=False)
            except:
                pass

    def run(self):
    # 1) Open RTSP (1 lần) + reader thread bơm vào self.q
        cap = open_cap(self.src, use_udp=self.rtsp_udp)
        if not cap or not cap.isOpened():
            self.status.emit(self.cam_idx, "failed to open stream")
            return

        import threading
        t = threading.Thread(target=self.reader_loop, args=(cap,), daemon=True)
        t.start()
        self.status.emit(self.cam_idx, "reader started")

        # 2) Khởi tạo mô hình (ngoài vòng while)
        if HAVE_YOLO:
            try:
                self.model = YOLO("yolov8s.pt" if self.high_quality else "yolov8n.pt")
                self.status.emit(self.cam_idx, f"YOLO ready on {self.device} | half={self.use_half}")
            except Exception as e:
                self.status.emit(self.cam_idx, f"YOLO init failed: {e}")
                self.model = None

        if HAVE_DEEPSORT:
            try:
                self.tracker = DeepSort(
                    max_age=max(90, self.max_age),
                    n_init=1,
                    nms_max_overlap=1.0,
                    max_cosine_distance=0.35
                )
            except Exception as e:
                self.status.emit(self.cam_idx, f"DeepSORT init failed: {e}")
                self.tracker = None

        if HAVE_FACE and self.face_db is not None:
            try:
                force_cpu = self.cpu_face
                if not force_cpu and HAVE_ORT:
                    providers = ['CUDAExecutionProvider','CPUExecutionProvider']
                    ctx_id = (self.gpu_id if (self.gpu_id>=0) else 0)
                else:
                    providers = ['CPUExecutionProvider']; ctx_id = -1
                self.face_app = FaceAnalysis(providers=providers)
                self.face_app.prepare(ctx_id=ctx_id, det_size=(self.det_size, self.det_size))
                self.status.emit(self.cam_idx, f"Face ready providers={providers} (ctx_id={ctx_id}) | {gpu_info_text(self.gpu_id)}")
            except Exception as e:
                self.status.emit(self.cam_idx, f"Face init failed: {e}")
                self.face_app = None

        # 3) Main loop: lấy frame mới nhất từ queue, xử lý theo nhịp
        last_fps_t = time.time(); frame_cnt = 0; fps = 0.0
        last_det_t = 0.0
        det_interval = max(1 / 12.0, 1.0 / (15 if self.high_quality else 20))  # detect ~12–20 FPS
        self._last_emit = 0.0

        labels = (self.face_db or {}).get("labels", [])
        while not self.stop_flag:
            # luôn lấy khung MỚI nhất; nếu không có → chờ nhẹ rồi tiếp
            try:
                frame = self.q.get(timeout=0.3)
            except:
                self.status.emit(self.cam_idx, "waiting frame…")
                continue

            if frame is None or frame.size == 0:
                continue

            now = time.time()
            frame_cnt += 1
            if now - last_fps_t >= 1.0:
                fps = frame_cnt / (now - last_fps_t)
                last_fps_t = now
                frame_cnt = 0

            H, W = frame.shape[:2]

            # 3.1 Detect người: chỉ chạy theo nhịp để giảm tải
            dets_person = []
            do_detect = (self.model is not None) and ((now - last_det_t) >= det_interval)
            if do_detect:
                try:
                    res = self.model.predict(
                        frame,
                        imgsz=(960 if self.high_quality else 640),
                        conf=0.15, iou=0.45, verbose=False,
                        device=self.device, half=self.use_half
                    )[0]
                    for b in res.boxes:
                        if int(b.cls[0].item()) != 0:  # chỉ class 'person'
                            continue
                        x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                        confv = float(b.conf[0].item())
                        dets_person.append(([x1,y1,x2,y2], confv, "person"))
                    last_det_t = now
                except Exception as e:
                    self.status.emit(self.cam_idx, f"YOLO err: {e}")

            # 3.2 Tracking (có thể cập nhật dù không detect mỗi khung)
            tracks = []
            if self.tracker is not None:
                try:
                    # Nếu không có dets ở khung này, truyền [] để DeepSORT duy trì track theo max_age
                    tracks = self.tracker.update_tracks(dets_person, frame=frame)
                except Exception as e:
                    self.status.emit(self.cam_idx, f"DeepSORT err: {e}")

            # 3.3 Face scan (thưa hơn)
            face_infos = []
            has_track = any(t.is_confirmed() and t.time_since_update==0 for t in tracks)
            do_face = False
            if has_track and self.face_app is not None and self.face_db is not None:
                self._face_tick = (self._face_tick + 1) % self.face_every
                do_face = (self._face_tick == 0)
            if do_face:
                try:
                    faces = self.face_app.get(frame)
                    E = self.face_db["E"]
                    for f in faces:
                        fx1,fy1,fx2,fy2 = map(int, f.bbox)
                        emb = getattr(f, "normed_embedding", None)
                        if emb is None: continue
                        sims = E @ emb.astype(np.float32)
                        j = int(np.argmax(sims)); simv = float(sims[j])
                        face_infos.append((fx1,fy1,fx2,fy2, j, simv))
                except Exception as e:
                    self.status.emit(self.cam_idx, f"Face err: {e}")

            # 3.4 Gán face→track + sticky (giữ nguyên logic của bạn)
            def overlap_face_ratio(tb, fb):
                tx1,ty1,tx2,ty2 = tb; fx1,fy1,fx2,fy2 = fb
                ix1,iy1 = max(tx1,fx1), max(ty1,fy1)
                ix2,iy2 = min(tx2,fx2), min(ty2,fy2)
                iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
                inter = iw*ih; farea = max(1,(fx2-fx1)*(fy2-fy1))
                return inter/farea

            def face_center_in_track(tb, fb):
                tx1,ty1,tx2,ty2 = tb; fx1,fy1,fx2,fy2 = fb
                cx, cy = (fx1+fx2)/2.0, (fy1+fy2)/2.0
                return (tx1 <= cx <= tx2) and (ty1 <= cy <= ty2)

            for trk in tracks:
                if not trk.is_confirmed() or trk.time_since_update>0: continue
                x1,y1,x2,y2 = map(int, trk.to_tlbr())
                tid = trk.track_id; key=(self.cam_idx, tid)
                if key not in self.track_start: self.track_start[key] = now
                cands = []
                for (fx1,fy1,fx2,fy2,j,simv) in face_infos:
                    fbox=(fx1,fy1,fx2,fy2); tbox=(x1,y1,x2,y2)
                    if simv>=self.threshold and (face_center_in_track(tbox,fbox) or overlap_face_ratio(tbox,fbox)>=0.5):
                        cands.append((simv,j))
                if cands:
                    cands.sort(reverse=True)
                    best_sim, best_j = cands[0]
                    if key in self.track2id and self.track2id[key].get("locked",False):
                        cur=self.track2id[key]["idx"]
                        if best_j!=cur and best_sim >= (self.threshold + self.relock_margin):
                            self.track2id[key] = {"idx":best_j, "last_seen":now, "sim":best_sim, "locked":True}
                        else:
                             if key in self.track2id:
                                self.track2id[key]["last_seen"] = now
                                self.track2id[key]["sim"]=max(self.track2id[key].get("sim",0.0), best_sim)
                    else:
                        self.track2id[key] = {"idx":best_j, "last_seen":now, "sim":best_sim, "locked":bool(self.sticky)}
                else:
                    if key in self.track2id:
                        if self.sticky:
                            self.track2id[key]["last_seen"]=now
                        else:
                            if (now - self.track2id[key]["last_seen"])>self.ttl:
                                self.track2id.pop(key,None)

            active_keys = {(self.cam_idx, t.track_id) for t in tracks if t.is_confirmed()}
            for k in list(self.track2id.keys()):
                if k[0] == self.cam_idx and k not in active_keys:
                    # chỉ xóa nếu sticky=False và ttl đã hết
                    if not self.sticky and (now - self.track2id[k]["last_seen"]) > self.ttl:
                        self.track2id.pop(k, None)

            for k in list(self.track_start.keys()):
                if k[0]==self.cam_idx and k not in active_keys:
                    self.track_start.pop(k,None)

            # 3.5 Vẽ + JSON
            json_items=[]
            for t_ in tracks:
                if not t_.is_confirmed() or t_.time_since_update>0: continue
                x1,y1,x2,y2 = map(int, t_.to_tlbr()); tid=t_.track_id
                key=(self.cam_idx,tid); duration= now - self.track_start.get(key, now)

                # ---- Ưu tiên manual label ----
                if key in self.manual_labels:
                    disp = self.manual_labels[key]
                    color = (255, 200, 0)   # vàng cam để dễ nhận biết
                    label = f"[Manual] {disp}"
                    info  = f"ID:{tid} (manual)"
                elif key in self.track2id:
                    j=self.track2id[key]["idx"]
                    disp = labels[j] if 0<=j<len(labels) else f"idx{j}"
                    color=(0,255,0) if self.track2id[key].get("locked",False) else (0,200,200)
                    label=disp; info=f"ID:{tid} | idx:{j}"
                else:
                    color=(0,0,255); label="Unknown"; info=f"ID:{tid}"; disp="Unknown"

                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                put_label(frame,label,(x1,max(0,y1-8)),color=color)
                put_label(frame,info,(x1,min(H-5,y2+20)),color=(255,255,255),scale=0.6)

                json_items.append({
                    "id":int(tid),
                    "name":disp,
                    "zone":f"Cam{self.cam_idx}",
                    "duration_sec":round(duration,2)
                })


            put_label(frame, f"FPS:{fps:.1f}", (10,25), color=(255,255,0), scale=0.8)

            # 3.6 Emit (giữ nhịp GUI)
            if (now - self._last_emit) >= self.render_interval:
                self._last_emit = now
                self.frame_ready.emit(self.cam_idx, frame)
            self.json_ready.emit(self.cam_idx, json_items)

        # 4) Cleanup
        try:
            cap.release()
        except Exception:
            pass


    def stop(self):
        self.stop_flag = True

# ---------------- GUI / Supervisor ----------------
class MainWindow(QWidget):
    def __init__(self, args):
        super().__init__()
        self.setWindowTitle("Eight-Cam GUI | YOLO + DeepSORT + InsightFace (per-cam threads)")
        self.resize(1700, 980)

        self.manual_labels = {}  # (cam_idx, track_id) -> name

        # Grid 2×4
        self.labels=[]
        grid=QGridLayout()
        for i in range(8):
            lbl=QLabel("No Camera"); lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("background:#202020;color:#aaa;border:1px solid #444;")
            lbl.setMinimumSize(360,240); self.labels.append(lbl)
            grid.addWidget(lbl, i//4, i%4)

        # Controls
        ctrl=QGroupBox("Controls"); form=QFormLayout()
        self.edt_cfg=QLineEdit(args.config or "")
        self.edt_src=QLineEdit("")
        self.edt_thr=QLineEdit(str(args.threshold))
        self.spin_det=QSpinBox(); self.spin_det.setRange(256,1536); self.spin_det.setSingleStep(64); self.spin_det.setValue(args.det_size)
        self.chk_cpu=QCheckBox("Force CPU (InsightFace)"); self.chk_cpu.setChecked(args.cpu)
        self.chk_udp=QCheckBox("RTSP over UDP"); self.chk_udp.setChecked(args.rtsp_udp)
        self.chk_sticky=QCheckBox("Sticky ID until track lost"); self.chk_sticky.setChecked(args.sticky_id)
        self.chk_hq=QCheckBox("High quality detect (v8s + imgsz 960)")
        self.spin_ttl=QLineEdit(str(args.ttl)); self.spin_relock=QLineEdit(str(args.relock_margin)); self.spin_maxage=QLineEdit(str(args.max_age))
        self.spin_face_every=QSpinBox(); self.spin_face_every.setRange(1,10); self.spin_face_every.setValue(4)
        self.spin_render=QSpinBox(); self.spin_render.setRange(5,60); self.spin_render.setValue(25)

        # Manual label inputs
        self.input_cam = QLineEdit(); self.input_track = QLineEdit(); self.input_name = QLineEdit()
        self.btn_assign=QPushButton("Assign Manual Label")
        self.btn_assign.clicked.connect(self.on_assign_label)

        self.btn_start=QPushButton("Start"); self.btn_stop=QPushButton("Stop")
        self.btn_start.clicked.connect(self.on_start); self.btn_stop.clicked.connect(self.on_stop)

        for w in [
            ("Config file (optional):", self.edt_cfg),
            ("Extra sources (comma):", self.edt_src),
            ("Face threshold:", self.edt_thr),
            ("Face det size:", self.spin_det),
        ]:
            form.addRow(*w)
        form.addRow(self.chk_cpu, self.chk_udp)
        form.addRow(self.chk_sticky)
        form.addRow(self.chk_hq)
        form.addRow("TTL (s):", self.spin_ttl)
        form.addRow("Relock margin:", self.spin_relock)
        form.addRow("DeepSORT max_age:", self.spin_maxage)
        form.addRow("Face scan every N frames:", self.spin_face_every)
        form.addRow("Render FPS:", self.spin_render)
        form.addRow("Cam idx:", self.input_cam)
        form.addRow("Track id:", self.input_track)
        form.addRow("Name:", self.input_name)
        form.addRow(self.btn_assign)
        form.addRow(self.btn_start, self.btn_stop)
        ctrl.setLayout(form)

        self.txt=QTextEdit(); self.txt.setReadOnly(True)
        self.txt.setStyleSheet("background:#0e0e0e;color:#d0ffd0;font-family:Consolas,monospace;")
        self.lbl_status=QLabel("")
        self.kv = QLabel("ID:\nName:\nZone:\nDuration_sec:")
        self.kv.setStyleSheet(
            "background:#0e0e0e; color:#d0ffd0; font-family:Consolas,monospace; "
            "padding:10px; border:1px solid #333;"
        )
        self.kv.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.kv.setMinimumHeight(140)  # just tall enough for 4 lines

        self.lbl_status = QLabel("")

        right = QVBoxLayout()
        right.addWidget(ctrl)
        right.addWidget(self.lbl_status)
        right.addWidget(self.kv, 1)

        root = QHBoxLayout(self)
        root.addLayout(grid, 2)
        root.addLayout(right, 1)
        ctrl.hide()


        self.workers: List = []
        self.latest_tracks = {}
        self.face_db = load_index(FACE_INDEX)
        self.lbl_status.setText(f"[INIT] {gpu_info_text(args.gpu)}")

        # Periodic JSON aggregator
        self.timer = QTimer(self); self.timer.setInterval(300)
        self.timer.timeout.connect(self.flush_json)
        QTimer.singleShot(0, self.on_start)
    def on_assign_label(self):
        try:
            cam_idx=int(self.input_cam.text().strip())
            track_id=int(self.input_track.text().strip())
            name=self.input_name.text().strip()
            self.manual_labels[(cam_idx, track_id)] = name
            QMessageBox.information(self,"Manual Label", f"Cam{cam_idx} Track{track_id} -> {name}")
        except Exception as e:
            QMessageBox.critical(self,"Error",f"Invalid input: {e}")
    def resolve_sources(self, cfg: Optional[str], extra: List[str]):
        srcs: List[str] = []
        if cfg:
            if not os.path.exists(cfg):
                raise FileNotFoundError(f"Config not found: {cfg}")
            with open(cfg,'r',encoding='utf-8') as f:
                for ln in f:
                    s=ln.strip();
                    if not s or s.startswith('#'): continue
                    srcs.append(s)
        if extra: srcs += extra
        if not srcs: srcs = DEFAULT_SOURCES.copy()
        return srcs[:8]

    def on_start(self):
        if self.workers:
            QMessageBox.information(self,"Info","Already running."); return
        cfg = self.edt_cfg.text().strip() or None
        extra = [s.strip() for s in self.edt_src.text().split(',') if s.strip()]
        try:
            srcs = self.resolve_sources(cfg, extra)
        except Exception as e:
            QMessageBox.critical(self,"Sources",str(e)); return
        if not srcs:
            QMessageBox.critical(self,"Sources","No camera sources"); return

        thr = float(self.edt_thr.text() or 0.4)
        det = int(self.spin_det.value())
        cpu = self.chk_cpu.isChecked(); udp = self.chk_udp.isChecked(); sticky = self.chk_sticky.isChecked()
        ttl = float(self.spin_ttl.text() or 3.0); relock = float(self.spin_relock.text() or 0.10); max_age = int(self.spin_maxage.text() or 60)
        face_every = int(self.spin_face_every.value()); render_fps = int(self.spin_render.value())
        hq = self.chk_hq.isChecked()
        gpu_id = args.gpu
        use_half = bool(args.half)
        self.latest_tracks = {}
        for i, src in enumerate(srcs):
            w = CamWorker(i, src, thr, det, cpu, udp, sticky, ttl, relock, max_age, hq, face_every, render_fps, self.face_db, gpu_id, use_half)
            w.manual_labels = self.manual_labels 
            w.frame_ready.connect(self.on_frame)
            w.json_ready.connect(self.on_json)
            w.status.connect(self.on_status)
            self.workers.append(w)
            w.start()
        self.timer.start(); self.lbl_status.setText(f"[RUNNING] {len(self.workers)} workers")

    def on_stop(self):
        for w in self.workers:
            try: w.stop()
            except: pass
        for w in self.workers:
            try: w.wait(1000)
            except: pass
        self.workers.clear(); self.timer.stop(); self.lbl_status.setText("[STOPPED]")
        for lbl in self.labels:
            lbl.setText("No Camera"); lbl.setPixmap(QPixmap())
        self.txt.clear()

    def on_frame(self, cam_idx:int, frame_bgr):
        if cam_idx >= len(self.labels): return
        # Rotate cam6 (index 5)

        pix = to_pix(frame_bgr)
        if not pix.isNull():
            self.labels[cam_idx].setPixmap(pix.scaled(self.labels[cam_idx].size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.labels[cam_idx].setText("No Signal")

    def on_json(self, cam_idx:int, items:list):
        # ưu tiên manual_labels + đóng dấu thời gian cập nhật
        stamped = []
        now_t = time.time()
        for it in items:
            key = (cam_idx, it["id"])
            if key in self.manual_labels:
                it = dict(it)  # copy để không đụng shared ref
                it["name"] = self.manual_labels[key]
            # đóng dấu thời gian cập nhật (quan trọng!)
            if "_t" not in it:
                it = dict(it)
            it["_t"] = now_t
            stamped.append(it)
        self.latest_tracks[cam_idx] = stamped


    def on_status(self, cam_idx:int, msg:str):
        self.lbl_status.setText(f"Cam{cam_idx}: {msg}")

    def flush_json(self):
        # Gom tất cả track mới nhất từ các cam
        all_items = []
        for ci in sorted(self.latest_tracks.keys()):
            all_items.extend(self.latest_tracks[ci])
        # Payload JSON (giữ nguyên cấu trúc và vẫn lưu file)
        payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "active_tracks": all_items
        }
        js = json.dumps(payload, ensure_ascii=False, indent=2)
        try:
            os.makedirs(os.path.dirname(OUT_JSON) or ".", exist_ok=True)
            with open(OUT_JSON, "w", encoding="utf-8") as f:
                f.write(js)
        except Exception:
            pass
        # --- HIỂN THỊ: chọn track được cập nhật GẦN NHẤT ---
        show = max(all_items, key=lambda it: it.get("_t", 0.0)) if all_items else None
        if show is None:
            text = "ID:\nName:\nZone:\nDuration_sec:"
        else:
            text = (
                f"ID: {show.get('id','')}\n"
                f"Name: {show.get('name','')}\n"
                f"Zone: {show.get('zone','')}\n"
                f"Duration_sec: {show.get('duration_sec','')}"
            )
        if hasattr(self, "kv"):
            self.kv.setText(text)


    def closeEvent(self, ev):
        self.on_stop(); super().closeEvent(ev)

# ---------------- CLI ----------------

def parse_args():
    ap = argparse.ArgumentParser("Eight-Cam GUI (per-camera threads)")
    ap.add_argument("--config")
    ap.add_argument("--threshold", type=float, default=0.40)
    ap.add_argument("--det-size", type=int, default=512)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--rtsp-udp", action="store_true")
    ap.add_argument("--sticky-id", action="store_true")
    ap.add_argument("--ttl", type=float, default=3.0)
    ap.add_argument("--relock-margin", type=float, default=0.10)
    ap.add_argument("--max-age", type=int, default=60)
    ap.add_argument("--gpu", type=int, default=0, help="GPU device id, -1 = CPU")
    ap.add_argument("--half", action="store_true", help="Use FP16 for YOLO if GPU available")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    app = QApplication(sys.argv)
    w = MainWindow(args)
    w.show()
    sys.exit(app.exec())