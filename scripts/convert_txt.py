#!/usr/bin/env python3
"""
Simple COLMAP TXT -> affordance PLY converter (patched).
Implements: z-buffer occlusion, weighted fusion (view-angle & distance), and feature cache.
"""

import os
import argparse
import numpy as np
from PIL import Image
import open3d as o3d
from tqdm import tqdm

# --- I/O helpers ---
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

# --- Math helpers ---
def qvec2rotmat(q):
    qw, qx, qy, qz = q
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)]
    ], dtype=np.float64)

def project_points(pts, R, t, params):
    # Pinhole projection (u = fx * Xc/Zc + cx ; v = fy * Yc/Zc + cy)
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

# --- Feature & text encoders ---
def extract_rgb(path, down=1):
    img = Image.open(path).convert('RGB')
    if down > 1:
        w,h = img.size; img = img.resize((w//down, h//down), Image.BILINEAR)
    return np.asarray(img).astype(np.float32) / 255.0  # H,W,3

def simple_text_emb(text, dim):
    if len(text)==0: return np.zeros(dim, dtype=np.float32)
    vals = np.array([ord(c) for c in text], dtype=np.float32)
    rng = np.linspace(0, 2*np.pi, dim)
    v = np.cos(rng * (vals.mean()%10 + 0.1)).astype(np.float32)
    return v / (np.linalg.norm(v)+1e-9)

# --- Patched pipeline ---
def run(colmap_txt_dir, images_dir, ply_path, text, out_ply,
        down=1, max_images=None, z_eps=0.01):
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points)                      # (N,3)
    cams = read_cameras_txt(os.path.join(colmap_txt_dir,'cameras.txt'))
    imgs = read_images_txt(os.path.join(colmap_txt_dir,'images.txt'))
    img_items = list(imgs.items())[:max_images] if max_images else list(imgs.items())

    N = pts.shape[0]
    feat_sum = None
    counts = np.zeros(N, dtype=np.float32)           # float counts for weighted fusion

    feat_cache = {}  # (img_path, down) -> feat_map

    for _, info in tqdm(img_items, desc="images"):
        img_path = os.path.join(images_dir, info['name'])
        if not os.path.exists(img_path): continue

        cam = cams[info['cam_id']]
        key = (img_path, down)
        if key in feat_cache:
            feat = feat_cache[key]
        else:
            feat = extract_rgb(img_path, down)
            feat_cache[key] = feat

        H, W, C = feat.shape
        R = qvec2rotmat(info['qvec']); t = info['tvec']

        # project points
        u, v, z = project_points(pts, R, t, cam['params'])
        if down > 1:
            u = u / down; v = v / down

        # valid points in front and inside image
        valid = (z > 0) & (u >= 0) & (u < (W-1)) & (v >= 0) & (v < (H-1))
        if not np.any(valid): continue
        idxs = np.nonzero(valid)[0]

        # z-buffer occlusion test (compute min depth per pixel)
        us_pix = np.floor(u[idxs]).astype(int)
        vs_pix = np.floor(v[idxs]).astype(int)
        flat_idx = vs_pix * W + us_pix
        zbuf_flat = np.full(H * W, np.inf, dtype=np.float32)
        # vectorized min: np.minimum.at
        np.minimum.at(zbuf_flat, flat_idx, z[idxs].astype(np.float32))
        zbuf = zbuf_flat.reshape(H, W)
        # keep points close to z-buffer (z-buffer test)
        z_at_pix = zbuf[vs_pix, us_pix]
        tol = np.maximum(z_eps, 0.01 * z[idxs])
        keep_mask = (z[idxs] <= (z_at_pix + tol))
        if not np.any(keep_mask): continue
        idxs = idxs[keep_mask]
        us_keep = u[idxs]; vs_keep = v[idxs]

        # sample features
        sampled = bilinear_sample(feat, us_keep, vs_keep)  # (M, C)

        # view-angle & distance weighting
        cam_center = - R.T.dot(t)           # camera center in world coords (C = -R^T t)
        vec = cam_center.reshape(1,3) - pts[idxs]   # (M,3) vector from point -> camera
        dists = np.linalg.norm(vec, axis=1) + 1e-9
        view_dir = vec / dists[:,None]     # normalized
        camera_forward_world = R.T.dot(np.array([0.0,0.0,1.0], dtype=np.float64))
        cos_theta = np.clip(view_dir.dot(camera_forward_world), 0.0, 1.0)  # clamp negative
        w = cos_theta / (dists**2 + 1e-6)   # weight formula: cos / (distance^2 + eps)

        # accumulate weighted features
        if feat_sum is None:
            feat_sum = np.zeros((N, sampled.shape[1]), dtype=np.float32)
        feat_sum[idxs] += sampled * w[:,None]
        counts[idxs] += w

    if feat_sum is None:
        raise RuntimeError("No features sampled. Check paths/intrinsics or downsample factor.")

    counts_safe = np.maximum(counts, 1e-9)[:,None]   # avoid divide by zero
    pfeats = feat_sum / counts_safe                  # weighted fusion: F = sum(w*f)/sum(w)
    norms = np.linalg.norm(pfeats, axis=1, keepdims=True) + 1e-9
    pnorm = pfeats / norms

    text_emb = simple_text_emb(text, pnorm.shape[1])
    text_emb = text_emb / (np.linalg.norm(text_emb)+1e-9)
    scores = pnorm @ text_emb
    smin, smax = float(scores.min()), float(scores.max())
    scores_norm = (scores - smin) / (smax - smin + 1e-9)

    # color map: blue->red
    colors = np.stack([np.minimum(1,2*scores_norm),
                       np.minimum(1,2*(1-np.abs(scores_norm-0.5))),
                       np.minimum(1,2*(1-scores_norm))], axis=1).astype(np.float32)

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts)
    out.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(out_ply, out)
    print("Wrote:", out_ply, "score_range:", smin, smax)

# --- CLI ---
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--colmap_txt_dir", required=True)
    p.add_argument("--images_dir", required=True)
    p.add_argument("--ply", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--downsample", type=int, default=1)
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--z_eps", type=float, default=0.01, help="z-buffer tolerance (abs or relative)")
    args = p.parse_args()
    run(args.colmap_txt_dir, args.images_dir, args.ply, args.text, args.out,
        down=args.downsample, max_images=args.max_images, z_eps=args.z_eps)
