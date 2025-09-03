# enroll_capture.py
import argparse, os, time, csv, unicodedata, re, threading, queue
import cv2
import numpy as np
from datetime import datetime
from insightface.app import FaceAnalysis

# ---------------------- Helpers ----------------------
def slugify(v: str) -> str:
    v = unicodedata.normalize('NFKD', v).encode('ascii', 'ignore').decode('ascii')
    return re.sub(r'[^a-zA-Z0-9]+', '-', v).strip('-').lower()

def open_cap(src):
    if isinstance(src, str) and src.startswith("rtsp://"):
        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(int(src))
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 4000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 4000)
    except Exception:
        pass
    return cap

def ensure_dirs(base, ident):
    faces_dir = os.path.join(base, "faces", ident)
    os.makedirs(faces_dir, exist_ok=True)
    return faces_dir

def save_metadata(base, row):
    meta = os.path.join(base, "metadata.csv")
    headers = ["mssv","name","phone","faces_dir","samples","timestamp"]
    exists = os.path.exists(meta)
    with open(meta, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(headers)
        w.writerow(row)

def crop_safe(img, xyxy):
    if xyxy is None: return None
    h, w = img.shape[:2]
    x1,y1,x2,y2 = [int(v) for v in xyxy]
    x1 = max(0, min(x1, w-1)); x2 = max(0, min(x2, w-1))
    y1 = max(0, min(y1, h-1)); y2 = max(0, min(y2, h-1))
    if x2 <= x1 or y2 <= y1: return None
    return img[y1:y2, x1:x2]

def capture_thread(source, q: queue.Queue, stop_flag: dict):
    cap = open_cap(source)
    last_ok = time.time()
    while not stop_flag["stop"]:
        # lấy frame mới nhất, không dồn queue
        if not cap.grab():
            if time.time() - last_ok > 3:
                print(f"[{source}] reconnecting RTSP...")
                cap.release(); cap = open_cap(source); last_ok = time.time()
            continue
        ret, frame = cap.retrieve()
        if not ret:
            if time.time() - last_ok > 3:
                print(f"[{source}] reconnecting RTSP...")
                cap.release(); cap = open_cap(source); last_ok = time.time()
            continue
        last_ok = time.time()
        if not q.empty():
            try: q.get_nowait()
            except: pass
        q.put(frame)
    cap.release()

# ---------------------- Main ----------------------
def main():
    ap = argparse.ArgumentParser(description="Enroll khuôn mặt từ RTSP/Webcam (low-latency + quality check)")
    ap.add_argument("--source", required=True, help="RTSP hoặc webcam index (vd: 0)")
    ap.add_argument("--name", required=True, help="Họ tên người cần đăng ký")
    ap.add_argument("--mssv", required=True)
    ap.add_argument("--phone", required=True)
    ap.add_argument("--out", default="dataset", help="Thư mục gốc lưu dataset")
    ap.add_argument("--samples", type=int, default=30, help="Số ảnh cần lưu")
    ap.add_argument("--show", action="store_true", help="Hiển thị preview")
    ap.add_argument("--rtsp-udp", action="store_true", help="Dùng UDP cho RTSP (mặc định TCP)")
    ap.add_argument("--det-size", type=int, default=512, help="Kích thước detect (ví dụ 512/640/800/1024)")
    ap.add_argument("--min-conf", type=float, default=0.60, help="Ngưỡng confidence tối thiểu để lưu")
    ap.add_argument("--min-sharp", type=float, default=50.0, help="Ngưỡng độ nét (variance of Laplacian)")
    ap.add_argument("--cpu", action="store_true", help="Bắt buộc dùng CPU (nếu không set thì ưu tiên CUDA)")
    args = ap.parse_args()

    # RTSP options
    transport = "udp" if args.rtsp_udp else "tcp"
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        f"rtsp_transport;{transport}|"
        "stimeout;5000000|max_delay;0|buffer_size;102400|"
        "reorder_queue_size;0|fflags;nobuffer|flags;low_delay"
    )

    ident = f"{args.mssv}_{slugify(args.name)}"
    faces_dir = ensure_dirs(args.out, ident)

    # InsightFace providers + prepare
    providers = ['CPUExecutionProvider'] if args.cpu else ['CUDAExecutionProvider','CPUExecutionProvider']
    face_app = FaceAnalysis(providers=providers)
    # ctx_id: 0 = GPU0, -1 = CPU
    ctx_id = -1 if args.cpu else 0
    face_app.prepare(ctx_id=ctx_id, det_size=(args.det_size, args.det_size))

    # Thread capture (không dồn frame)
    stop_flag = {"stop": False}
    q = queue.Queue(maxsize=1)
    t = threading.Thread(target=capture_thread, args=(args.source, q, stop_flag), daemon=True)
    t.start()

    # Chờ frame đầu
    frame = None
    timeout = time.time() + 10
    while frame is None and time.time() < timeout:
        if not q.empty():
            frame = q.get()
        else:
            time.sleep(0.01)
    if frame is None:
        stop_flag["stop"] = True
        t.join(timeout=1.0)
        raise RuntimeError("Không nhận được khung hình đầu tiên. Kiểm tra RTSP/Webcam.")

    H, W = frame.shape[:2]
    CENTER_BOX_RATIO = 0.4
    cx1 = int(W*(1-CENTER_BOX_RATIO)/2); cx2 = int(W*(1+(CENTER_BOX_RATIO))/2)
    cy1 = int(H*(1-CENTER_BOX_RATIO)/2); cy2 = int(H*(1+(CENTER_BOX_RATIO))/2)
    center_box = (cx1, cy1, cx2, cy2)

    captured = 0
    last_save = 0.0
    min_interval = 0.12  # 120ms
    auto = True

    if args.show:
        cv2.namedWindow("Enroll (Low-Latency)", cv2.WINDOW_NORMAL)
        view_w = min(960, W)
        cv2.resizeWindow("Enroll (Low-Latency)", view_w, int(H * view_w / W))

    print("[Hướng dẫn] Nhìn vào cam, quay trái/phải/ngửa/cúi nhẹ. SPACE: bật/tắt auto | C: chụp tay | ESC: thoát")
    MIN_CONF = args.min_conf
    MIN_SHARPNESS = args.min_sharp

    try:
        while True:
            # lấy frame mới nhất
            if not q.empty():
                frame = q.get()

            work = frame.copy()

            # optional brighten nhẹ nếu tối
            gray_mean = work.mean()
            if gray_mean < 80:
                work = cv2.convertScaleAbs(work, alpha=1.2, beta=15)

            # detect
            faces = face_app.get(work)
            face_xyxy, face_conf, sharp = None, 0.0, 0.0
            disp = frame.copy()

            # vẽ vùng trung tâm
            cv2.rectangle(disp, (cx1,cy1), (cx2,cy2), (255,255,0), 1)

            if len(faces) > 0:
                best = max(faces, key=lambda f: float(getattr(f, "det_score", 0.0)))
                x1,y1,x2,y2 = map(int, best.bbox)
                face_xyxy = [x1,y1,x2,y2]
                face_conf = float(getattr(best, "det_score", 0.0))

                face_crop_dbg = crop_safe(work, face_xyxy)
                if face_crop_dbg is not None:
                    g = cv2.cvtColor(face_crop_dbg, cv2.COLOR_BGR2GRAY)
                    sharp = cv2.Laplacian(g, cv2.CV_64F).var()

                cv2.rectangle(disp, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(disp, f"conf {face_conf:.2f} | sharp {sharp:.0f}",
                            (x1, max(0,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # quyết định lưu
            now = time.time()
            should_save = False
            if auto and face_xyxy is not None and (now - last_save) > min_interval:
                fx1,fy1,fx2,fy2 = face_xyxy
                inter_x1 = max(fx1, cx1); inter_y1 = max(fy1, cy1)
                inter_x2 = min(fx2, cx2); inter_y2 = min(fy2, cy2)
                inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                farea = max(1, (fx2-fx1)*(fy2-fy1))
                overlap = inter / farea
                if overlap >= 0.5:
                    should_save = True

            if args.show:
                msg = f"{args.name} | MSSV {args.mssv} | {captured}/{args.samples} | Auto:{'ON' if auto else 'OFF'}"
                cv2.putText(disp, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                cv2.imshow("Enroll (Low-Latency)", disp)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):   # chụp tay
                    should_save = True
                elif key == 32:       # SPACE: toggle auto
                    auto = not auto
                    print("Auto-capture:", "ON" if auto else "OFF")
                elif key == 27:       # ESC
                    break

            # ... trong khối:
            # if should_save and face_xyxy is not None and face_conf >= MIN_CONF and sharp >= MIN_SHARPNESS:

            if should_save and face_xyxy is not None and face_conf >= MIN_CONF and sharp >= MIN_SHARPNESS:
                x1,y1,x2,y2 = face_xyxy
                # nới biên 20% theo mỗi chiều để crop rộng, detect lại luôn chắc chắn
                w = x2 - x1
                h = y2 - y1
                pad_x = int(0.2 * w)
                pad_y = int(0.2 * h)
                x1p = max(0, x1 - pad_x); y1p = max(0, y1 - pad_y)
                x2p = min(frame.shape[1]-1, x2 + pad_x)
                y2p = min(frame.shape[0]-1, y2 + pad_y)

                face_crop = frame[y1p:y2p, x1p:x2p]  # lưu từ frame gốc (không brighten)
                if face_crop is not None and face_crop.size:
                    captured += 1
                    cv2.imwrite(os.path.join(faces_dir, f"{captured:04d}.jpg"), face_crop)
                    last_save = now


            if captured >= args.samples:
                break

    finally:
        stop_flag["stop"] = True
        t.join(timeout=1.0)
        if args.show:
            cv2.destroyAllWindows()

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_metadata(args.out, [args.mssv, args.name, args.phone, faces_dir, captured, ts])
    print(f"✅ Done. Lưu {captured} ảnh vào: {faces_dir}\n→ metadata: {os.path.join(args.out,'metadata.csv')}")

if __name__ == "__main__":
    main()
