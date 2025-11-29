#!/usr/bin/env python3
"""
Simple, modular COLMAP TXT -> affordance PLY converter with OpenSeg + CLIP support.
Defaults openseg_savedmodel_dir to: /home/harsh/openscene/openseg_tf/exported_model/

Keep helpers small and the main flow compact.
"""

import os
import argparse
import numpy as np
from PIL import Image
import open3d as o3d

# optional libs (import errors handled in helpers)
try:
    import tensorflow as tf
    TRY_TF = True
except Exception:
    tf = None
    TRY_TF = False

try:
    import torch
    import clip
    TRY_CLIP = True
except Exception:
    torch = None
    clip = None
    TRY_CLIP = False

# -----------------------
# Helpers (small, focused)
# -----------------------

def read_cameras_txt(path):
    cams = {}
    with open(path, 'r') as f:
        for line in f:
            if not line.strip() or line.startswith('#'): continue
            p = line.split()
            cid = int(p[0]); model = p[1]; w = int(p[2]); h = int(p[3])
            params = np.array([float(x) for x in p[4:]], dtype=np.float64)
            cams[cid] = {'model': model, 'w': w, 'h': h, 'params': params}
    return cams

def read_images_txt(path):
    imgs = {}
    with open(path, 'r') as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        iid = int(parts[0])
        qvec = np.array(list(map(float, parts[1:5])), dtype=np.float64)
        tvec = np.array(list(map(float, parts[5:8])), dtype=np.float64)
        cam_id = int(parts[8]); name = parts[9]
        imgs[iid] = {'qvec': qvec, 'tvec': tvec, 'cam_id': cam_id, 'name': name}
        i += 2
    return imgs

def qvec2rotmat(q):
    qw, qx, qy, qz = q
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)]
    ], dtype=np.float64)

def project_points(pts, R, t, params):
    # Pinhole projection
    Xc = (R @ pts.T).T + t.reshape(1,3)
    Z = Xc[:,2]
    eps = 1e-9
    x = Xc[:,0] / (Z + eps); y = Xc[:,1] / (Z + eps)
    fx, fy, cx, cy = params[:4]
    u = fx * x + cx; v = fy * y + cy
    return u, v, Z

def bilinear_sample(feat, us, vs):
    H, W, C = feat.shape
    us_f = np.asarray(us); vs_f = np.asarray(vs)
    i = np.floor(us_f).astype(int); j = np.floor(vs_f).astype(int)
    i0 = np.clip(i, 0, W-1); j0 = np.clip(j, 0, H-1)
    i1 = np.clip(i0+1, 0, W-1); j1 = np.clip(j0+1, 0, H-1)
    a = (us_f - i)[:,None]; b = (vs_f - j)[:,None]
    f00 = feat[j0, i0]; f01 = feat[j0, i1]; f10 = feat[j1, i0]; f11 = feat[j1, i1]
    return (1-a)*(1-b)*f00 + a*(1-b)*f01 + (1-a)*b*f10 + a*b*f11

def extract_rgb(path, down=1):
    img = Image.open(path).convert('RGB')
    if down > 1:
        w,h = img.size; img = img.resize((w//down, h//down), Image.BILINEAR)
    return np.asarray(img).astype(np.float32) / 255.0

def simple_text_emb(text, dim):
    if len(text)==0: return np.zeros(dim, dtype=np.float32)
    vals = np.array([ord(c) for c in text], dtype=np.float32)
    rng = np.linspace(0, 2*np.pi, dim)
    v = np.cos(rng * (vals.mean()%10 + 0.1)).astype(np.float32)
    return v / (np.linalg.norm(v)+1e-9)

# -----------------------
# OpenSeg + CLIP helpers
# -----------------------

def load_openseg_model(saved_model_dir):
    if not TRY_TF:
        raise RuntimeError("TensorFlow not installed; cannot load OpenSeg model.")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass
    model = tf.saved_model.load(saved_model_dir)
    return model

def load_clip_model(device):
    if not TRY_CLIP:
        return None
    model, _ = clip.load("ViT-L/14@336px", device=device)
    model.eval()
    return model

# build token-level and pooled CLIP embeddings
def build_clip_text_embs(text, clip_model, device):
    """
    Returns (token_emb_np, pooled_emb_np)
    - token_emb_np: shape (1, T, D) float32 (per-token embedding)
    - pooled_emb_np: shape (1, D) float32 (pooled text embedding)
    """
    if clip_model is None:
        return None, None
    with torch.no_grad():
        token = clip.tokenize([text]).to(device)            # (1, T)
        # token embeddings (before transformer) -> shape (1, T, D)
        token_emb = clip_model.token_embedding(token)       # torch.Tensor (1,T,D)
        token_emb = token_emb.cpu().numpy().astype(np.float32)

        # pooled embedding (original encode_text)
        pooled = clip_model.encode_text(token)              # (1, D)
        pooled = pooled.cpu().numpy().astype(np.float32)
        pooled = pooled / (np.linalg.norm(pooled, axis=1, keepdims=True) + 1e-9)

    return token_emb, pooled

def extract_openseg_feat_np(
        img_path,
        openseg_model,
        token_emb_np,      # (1,T,D) or None
        pooled_emb_np,     # (1,D)
        img_size=None,
        cache_dir=None):

    # ---------- cache ----------
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        key_name = os.path.splitext(os.path.basename(img_path))[0]
        size_tag = f"{img_size[0]}x{img_size[1]}" if img_size else "orig"
        cache_p = os.path.join(cache_dir, f"{key_name}_openseg_{size_tag}.npy")
        if os.path.exists(cache_p):
            return np.load(cache_p)

    if not TRY_TF:
        raise RuntimeError("TensorFlow not available for OpenSeg extraction.")

    raw_bytes = tf.io.read_file(img_path).numpy()

    # ---------- nested call ----------
    def call_sig(text_np):
        return openseg_model.signatures['serving_default'](
            inp_image_bytes=tf.convert_to_tensor(raw_bytes),
            inp_text_emb=tf.convert_to_tensor(text_np, dtype=tf.float32)
        )

    # ---------- try token-level then pooled ----------
    results = None
    if token_emb_np is not None:
        try:
            results = call_sig(token_emb_np)          # (1,T,D) → rank 3
        except Exception as e_token:
            if pooled_emb_np is None:
                raise RuntimeError(f"Token embedding failed: {e_token}")
            try:
                results = call_sig(pooled_emb_np)     # fallback rank 2
            except Exception as e_pool:
                raise RuntimeError(
                    f"OpenSeg failed for BOTH token ({e_token}) "
                    f"and pooled ({e_pool}) embeddings."
                )
    else:
        # no CLIP, only pooled fallback
        results = call_sig(pooled_emb_np)

    # ---------- process output ----------
    img_info = results['image_info']
    crop_sz = [
        int(img_info[0,0] * img_info[2,0]),
        int(img_info[0,1] * img_info[2,1])
    ]

    feat = results['ppixel_ave_feat'][:, :crop_sz[0], :crop_sz[1]]

    if img_size is not None:
        feat = tf.image.resize(
            feat, img_size, method='nearest'
        )[0].numpy().astype(np.float32)
    else:
        feat = feat[0].numpy().astype(np.float32)

    # ---------- normalize ----------
    feat = feat.astype(np.float32)
    feat /= (np.linalg.norm(feat, axis=2, keepdims=True) + 1e-9)

    if cache_dir:
        np.save(cache_p, feat)

    return feat


# -----------------------
# Small fusion helpers
# -----------------------

def zbuffer_filter(u, v, z, W, H, z_eps=0.01):
    # returns mask of kept indices (relative to input indices array)
    valid = (z > 0) & (u >= 0) & (u < (W-1)) & (v >= 0) & (v < (H-1))
    if not np.any(valid):
        return np.zeros_like(valid, dtype=bool)
    idxs = np.nonzero(valid)[0]
    us_pix = np.floor(u[idxs]).astype(int)
    vs_pix = np.floor(v[idxs]).astype(int)
    flat_idx = vs_pix * W + us_pix
    zbuf_flat = np.full(H * W, np.inf, dtype=np.float32)
    np.minimum.at(zbuf_flat, flat_idx, z[idxs].astype(np.float32))
    z_at_pix = zbuf_flat.reshape(H, W)[vs_pix, us_pix]
    tol = np.maximum(z_eps, 0.01 * z[idxs])
    keep_mask = (z[idxs] <= (z_at_pix + tol))
    mask = np.zeros_like(valid, dtype=bool)
    mask[idxs[keep_mask]] = True
    return mask

def compute_weights(R, t, pts_idx):
    cam_center = - R.T.dot(t)
    vec = cam_center.reshape(1,3) - pts_idx
    dists = np.linalg.norm(vec, axis=1) + 1e-9
    view_dir = vec / dists[:,None]
    camera_forward_world = R.T.dot(np.array([0.0,0.0,1.0], dtype=np.float64))
    cos_theta = np.clip(view_dir.dot(camera_forward_world), 0.0, 1.0)
    w = cos_theta / (dists**2 + 1e-6)  # weighted fusion formula
    return w

# -----------------------
# Compact main flow
# -----------------------

def main(args):
    # prepare models / embeddings
    clip_model = None
    device = "cpu"
    if args.use_clip and TRY_CLIP:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model = load_clip_model(device)

    openseg_model = None
    if args.use_openseg:
        if args.openseg_savedmodel_dir is None:
            args.openseg_savedmodel_dir = "openscene/exported_model"
        openseg_model = load_openseg_model(args.openseg_savedmodel_dir)

    pcd = o3d.io.read_point_cloud(args.ply)
    pts = np.asarray(pcd.points)
    cams = read_cameras_txt(os.path.join(args.colmap_txt_dir, 'cameras.txt'))
    imgs = read_images_txt(os.path.join(args.colmap_txt_dir, 'images.txt'))
    img_items = list(imgs.items())[:args.max_images] if args.max_images else list(imgs.items())

    N = pts.shape[0]
    feat_sum = None
    counts = np.zeros(N, dtype=np.float32)
    feat_cache = {}

    # precompute text embedding for OpenSeg (token + pooled) if needed
    token_emb_np, pooled_emb_np = None, None
    if args.use_openseg:
        if clip_model is not None:
            token_emb_np, pooled_emb_np = build_clip_text_embs(args.text, clip_model, device)  # (1,T,D), (1,D)
        else:
            # fallback simple vector: make pooled shape (1,D)
            pooled_emb_np = np.expand_dims(simple_text_emb(args.text, 512), 0)

    for _, info in img_items:
        img_path = os.path.join(args.images_dir, info['name'])
        if not os.path.exists(img_path):
            continue
        cam = cams[info['cam_id']]
        key = (img_path, args.downsample, args.use_openseg)
        if key in feat_cache:
            feat = feat_cache[key]
        else:
            if args.use_openseg:
                # choose small spatial size for features when downsampling
                try:
                    im = Image.open(img_path)
                    H0, W0 = im.size[1], im.size[0]
                    img_size = [max(32, H0//args.downsample), max(32, W0//args.downsample)]
                except Exception:
                    img_size = None
                # pass both token and pooled embeddings
                feat = extract_openseg_feat_np(img_path, openseg_model, token_emb_np, pooled_emb_np, img_size=img_size, cache_dir=args.cache_dir)
            else:
                feat = extract_rgb(img_path, args.downsample)
            feat_cache[key] = feat

        H, W, C = feat.shape
        R = qvec2rotmat(info['qvec']); t = info['tvec']

        # project and z-buffer filter
        u, v, z = project_points(pts, R, t, cam['params'])
        if args.downsample > 1:
            u = u / args.downsample; v = v / args.downsample
        keep_mask = zbuffer_filter(u, v, z, W, H, z_eps=args.z_eps)
        if not np.any(keep_mask):
            continue
        idxs = np.nonzero(keep_mask)[0]
        us_keep = u[idxs]; vs_keep = v[idxs]

        # sample features and compute weights
        sampled = bilinear_sample(feat, us_keep, vs_keep)   # (M,C)
        weights = compute_weights(R, t, pts[idxs])         # (M,)

        if feat_sum is None:
            feat_sum = np.zeros((N, sampled.shape[1]), dtype=np.float32)

        feat_sum[idxs] += sampled * weights[:,None]
        counts[idxs] += weights

    if feat_sum is None:
        raise RuntimeError("No features sampled. Check inputs or downsample factor.")

    counts_safe = np.maximum(counts, 1e-9)[:,None]
    pfeats = feat_sum / counts_safe
    pnorm = pfeats / (np.linalg.norm(pfeats, axis=1, keepdims=True) + 1e-9)

    # text embedding for similarity (use pooled CLIP embedding if available)
    if args.use_clip and TRY_CLIP and token_emb_np is not None and pooled_emb_np is not None:
        # pooled_emb_np is (1, D)
        t = pooled_emb_np.reshape(-1)
        if t.shape[0] != pnorm.shape[1]:
            # tile/trim
            d = pnorm.shape[1]
            tmp = np.zeros(d, dtype=np.float32)
            for i in range(d):
                tmp[i] = t[i % t.shape[0]]
            t = tmp / (np.linalg.norm(tmp) + 1e-9)
        text_emb = t
    else:
        text_emb = simple_text_emb(args.text, pnorm.shape[1])

    text_emb = text_emb / (np.linalg.norm(text_emb) + 1e-9)
    scores = pnorm @ text_emb
    smin, smax = float(scores.min()), float(scores.max())
    scores_norm = (scores - smin) / (smax - smin + 1e-9)

    colors = np.stack([
        np.minimum(1, 2*scores_norm),
        np.minimum(1, 2*(1 - np.abs(scores_norm - 0.5))),
        np.minimum(1, 2*(1 - scores_norm))
    ], axis=1).astype(np.float32)

    out_pc = o3d.geometry.PointCloud()
    out_pc.points = o3d.utility.Vector3dVector(pts)
    out_pc.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(args.out, out_pc)
    print("Wrote:", args.out, "score_range:", smin, smax)

# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--colmap_txt_dir", required=True)
    p.add_argument("--images_dir", required=True)
    p.add_argument("--ply", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--downsample", type=int, default=2, help="Use >=2 for 8GB VRAM")
    p.add_argument("--use_clip", action="store_true", help="Use CLIP text embedding")
    p.add_argument("--use_openseg", action="store_true", help="Use OpenSeg per-pixel features")
    p.add_argument("--openseg_savedmodel_dir", type=str,
                   default="openscene/exported_model",
                   help="Path to OpenSeg savedmodel (if required)")
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--z_eps", type=float, default=0.01)
    p.add_argument("--cache_dir", type=str, default="feat_cache")
    args = p.parse_args()
    main(args)
