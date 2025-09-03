# build_face_index.py (rescue)
import os, csv, glob
import numpy as np
import cv2
from insightface.app import FaceAnalysis

DATASET_DIR = "dataset"
META_CSV = os.path.join(DATASET_DIR, "metadata.csv")
OUT_NPZ  = os.path.join(DATASET_DIR, "face_index.npz")
MAX_PER_ID = 80
DET_SIZE = (1024, 1024)
MIN_CONF = 0.45
MIN_SHARP = 25.0
EXTS = ("*.jpg","*.jpeg","*.png","*.bmp")

def make_face_app(cpu=False):
    providers = ['CPUExecutionProvider'] if cpu else ['CUDAExecutionProvider','CPUExecutionProvider']
    app = FaceAnalysis(providers=providers)
    app.prepare(ctx_id=(-1 if cpu else 0), det_size=DET_SIZE)
    return app

def read_meta():
    if not os.path.exists(META_CSV):
        raise FileNotFoundError(f"Not found: {META_CSV}")
    rows = []
    with open(META_CSV, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows

def sharpness(img):
    if img is None or img.size == 0: return 0.0
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(g, cv2.CV_64F).var()

def pad_image(img, pad=0.4):
    h,w = img.shape[:2]
    bh, bw = int(h*pad), int(w*pad)
    return cv2.copyMakeBorder(img, bh,bh,bw,bw, cv2.BORDER_REPLICATE)

def upsize_min(img, min_side=600):
    h,w = img.shape[:2]
    s = min(h,w)
    if s >= min_side: return img
    scale = float(min_side)/max(1,s)
    return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)

def try_detect(app, img):
    """Trả về embedding tốt nhất hoặc None (thử nhiều chiến lược)."""
    strategies = [
        lambda x: x,
        lambda x: pad_image(x, 0.4),
        lambda x: upsize_min(pad_image(x, 0.8), 600),
    ]
    for st in strategies:
        cur = st(img)
        faces = app.get(cur)
        if not faces: 
            continue
        f = max(faces, key=lambda x: float(getattr(x, "det_score", 0.0)))
        conf = float(getattr(f, "det_score", 0.0))
        x1,y1,x2,y2 = map(int, f.bbox)
        crop = cur[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
        shp = sharpness(crop)
        if conf >= MIN_CONF and shp >= MIN_SHARP:
            return f.normed_embedding
    return None

def main():
    app = make_face_app(cpu=False)

    names, idents, mssvs, phones, embs = [], [], [], [], []
    metas = read_meta()
    print(f"[INDEX] identities in metadata: {len(metas)}")

    for r in metas:
        faces_dir = r["faces_dir"]
        person = f'{r["mssv"]}_{r["name"]}'.strip()
        if not os.path.isdir(faces_dir):
            print(f"[SKIP] no dir: {person}")
            continue

        img_paths = []
        for e in EXTS:
            img_paths += glob.glob(os.path.join(faces_dir, e))
        img_paths = sorted(img_paths)[:MAX_PER_ID]

        if not img_paths:
            print(f"[SKIP] no images: {person}")
            continue

        vecs = []
        used = 0
        for p in img_paths:
            img = cv2.imread(p)
            if img is None: 
                continue
            emb = try_detect(app, img)
            if emb is None:
                continue
            vecs.append(emb.astype(np.float32))
            used += 1

        if not vecs:
            print(f"[{person}] vectors: 0 (skip)")
            continue

        mean_vec = np.mean(vecs, axis=0).astype(np.float32)
        mean_vec /= (np.linalg.norm(mean_vec) + 1e-9)

        names.append(r["name"])
        idents.append(str(r["mssv"]))
        mssvs.append(r["mssv"])
        phones.append(r["phone"])
        embs.append(mean_vec)
        print(f"[{person}] vectors used: {used} → mean saved")

    if not embs:
        print("[INDEX] Nothing to save.")
        return

    arr = np.vstack(embs).astype(np.float32)
    np.savez_compressed(
        OUT_NPZ,
        embeddings=arr,
        names=np.array(names, dtype=object),
        idents=np.array(idents, dtype=object),
        mssvs=np.array(mssvs, dtype=object),
        phones=np.array(phones, dtype=object),
    )
    print(f"✅ Saved {arr.shape[0]} identities → {OUT_NPZ}")

if __name__ == "__main__":
    main()
