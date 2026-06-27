"""Programmatic generator for the chihuahua-vs-muffin dataset.

IMPORTANT — what this data actually is
--------------------------------------
This is **NOT** the real internet meme photo set. It is a controlled,
procedurally rendered 2-class image dataset built with numpy + OpenCV that
deliberately recreates the *fine-grained* challenge the meme is famous for:

  * Both classes are a round, brown/tan blob on a soft background.
  * Both classes carry small dark dots in the same colour range.
  * The ONLY reliable signal is structure:
        - "chihuahua"  -> two ears (triangles) on top + a face arrangement
                          of two eyes and a nose/snout, plus fur speckle.
        - "muffin"     -> a fluted muffin-top dome + scattered blueberry
                          specks (no ears, no symmetric eyes/nose layout).

Every image gets independent random variation in position, scale, rotation,
blob colour, dot count/placement, lighting and noise, so a model cannot win
by memorising pixels — it must learn the discriminative shape/texture cues.
This is genuine image data with a genuine fine-grained-vision difficulty; it
simply does not pretend to be the original meme photographs.

Run:  python -m data.generate            # writes data/dataset/{train,val,test}
"""
from __future__ import annotations

import os
import argparse
import numpy as np
import cv2

IMG = 64            # output images are IMG x IMG, 3-channel
CLASSES = ("chihuahua", "muffin")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _rng(seed: int) -> np.random.RandomState:
    return np.random.RandomState(seed)


def _background(r: np.random.RandomState) -> np.ndarray:
    """Soft, slightly noisy pastel background."""
    base = r.randint(205, 245, size=3).astype(np.float32)
    img = np.ones((IMG, IMG, 3), np.float32) * base
    # gentle vignette / lighting gradient
    gx = np.linspace(-1, 1, IMG)[None, :]
    gy = np.linspace(-1, 1, IMG)[:, None]
    grad = (gx * r.uniform(-12, 12) + gy * r.uniform(-12, 12))
    img += grad[..., None]
    return img


def _brown(r: np.random.RandomState) -> np.ndarray:
    """A muffin/chihuahua-ish brown-tan colour (BGR)."""
    # tan..golden..brown range, shared by both classes on purpose
    b = r.randint(70, 130)
    g = r.randint(120, 175)
    rr = r.randint(150, 205)
    return np.array([b, g, rr], np.float32)


def _blob(img, cx, cy, rx, ry, colour, r, angle=0.0):
    cv2.ellipse(img, (int(cx), int(cy)), (int(rx), int(ry)),
                angle, 0, 360, colour.tolist(), -1, cv2.LINE_AA)


def _finish(img: np.ndarray, r: np.random.RandomState) -> np.ndarray:
    img += r.normal(0, 4.0, img.shape)          # sensor noise
    img = np.clip(img, 0, 255).astype(np.uint8)
    if r.rand() < 0.5:                           # mild blur sometimes
        k = r.choice([3, 3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    return img


# --------------------------------------------------------------------------- #
# class renderers
# --------------------------------------------------------------------------- #
def render_chihuahua(seed: int) -> np.ndarray:
    r = _rng(seed)
    img = _background(r)
    colour = _brown(r)
    cx = IMG / 2 + r.uniform(-6, 6)
    cy = IMG / 2 + r.uniform(-3, 7)
    rx = r.uniform(15, 21)
    ry = r.uniform(16, 23)
    tilt = r.uniform(-18, 18)

    # ears: two triangles poking up above the head
    ear_c = (colour * r.uniform(0.8, 0.95)).tolist()
    for side in (-1, 1):
        bx = cx + side * rx * r.uniform(0.55, 0.8)
        by = cy - ry * r.uniform(0.55, 0.75)
        tipx = bx + side * r.uniform(2, 7)
        tipy = by - r.uniform(10, 18)
        w = r.uniform(5, 9)
        pts = np.array([[bx - w, by], [bx + w, by], [tipx, tipy]], np.int32)
        cv2.fillConvexPoly(img, pts, ear_c, cv2.LINE_AA)

    # head
    _blob(img, cx, cy, rx, ry, colour, r, angle=tilt)

    # fur speckle (subtle texture)
    for _ in range(r.randint(30, 70)):
        a = r.uniform(0, 2 * np.pi)
        rad = r.uniform(0, rx) * 0.9
        px = int(cx + np.cos(a) * rad)
        py = int(cy + np.sin(a) * rad)
        shade = (colour * r.uniform(0.75, 1.1)).clip(0, 255).tolist()
        cv2.circle(img, (px, py), r.randint(0, 1), shade, -1, cv2.LINE_AA)

    # FACE: two symmetric eyes + a nose below-centre — the key chihuahua cue
    eye_dx = rx * r.uniform(0.38, 0.5)
    eye_dy = -ry * r.uniform(0.05, 0.18)
    eye_r = max(1, int(r.uniform(2.2, 3.4)))
    for side in (-1, 1):
        ex, ey = int(cx + side * eye_dx), int(cy + eye_dy)
        cv2.circle(img, (ex, ey), eye_r, (20, 20, 25), -1, cv2.LINE_AA)
        cv2.circle(img, (ex - 1, ey - 1), 1, (235, 235, 235), -1, cv2.LINE_AA)
    # snout/nose
    nx, ny = int(cx + r.uniform(-1, 1)), int(cy + ry * r.uniform(0.32, 0.5))
    cv2.circle(img, (nx, ny), max(1, int(r.uniform(2.2, 3.2))),
               (35, 30, 40), -1, cv2.LINE_AA)

    return _finish(img, r)


def render_muffin(seed: int) -> np.ndarray:
    r = _rng(seed)
    img = _background(r)
    colour = _brown(r)
    cx = IMG / 2 + r.uniform(-6, 6)
    cy = IMG / 2 + r.uniform(-2, 8)
    rx = r.uniform(16, 22)
    ry = r.uniform(15, 21)

    # muffin top: a domed blob, slightly wider than tall, no ears
    _blob(img, cx, cy, rx, ry, colour, r, angle=r.uniform(-8, 8))

    # fluted rim suggestion: a darker band along the lower third
    rim = (colour * r.uniform(0.6, 0.8)).clip(0, 255).tolist()
    cv2.ellipse(img, (int(cx), int(cy + ry * 0.45)),
                (int(rx * 0.95), int(ry * 0.45)),
                0, 0, 180, rim, max(1, int(r.uniform(2, 4))), cv2.LINE_AA)
    # vertical flute lines on the base
    for k in range(r.randint(3, 6)):
        fx = int(cx + (k - 2) * rx * 0.3 + r.uniform(-2, 2))
        cv2.line(img, (fx, int(cy + ry * 0.25)), (fx, int(cy + ry * 0.95)),
                 rim, 1, cv2.LINE_AA)

    # blueberry specks: scattered dark-blue dots, NO symmetric face layout
    n = r.randint(6, 14)
    for _ in range(n):
        a = r.uniform(0, 2 * np.pi)
        rad = r.uniform(0, 0.85) * min(rx, ry)
        px = int(cx + np.cos(a) * rad)
        py = int(cy + np.sin(a) * rad * 0.85 - ry * 0.1)
        col = (r.randint(60, 110), r.randint(25, 55), r.randint(25, 60))
        cv2.circle(img, (px, py), max(1, int(r.uniform(1.6, 3.0))),
                   col, -1, cv2.LINE_AA)

    # crumb texture
    for _ in range(r.randint(40, 80)):
        a = r.uniform(0, 2 * np.pi)
        rad = r.uniform(0, rx) * 0.95
        px = int(cx + np.cos(a) * rad)
        py = int(cy + np.sin(a) * rad)
        shade = (colour * r.uniform(0.7, 1.15)).clip(0, 255).tolist()
        cv2.circle(img, (px, py), r.randint(0, 1), shade, -1, cv2.LINE_AA)

    return _finish(img, r)


RENDERERS = {"chihuahua": render_chihuahua, "muffin": render_muffin}


def make_image(label: str, seed: int) -> np.ndarray:
    return RENDERERS[label](seed)


# --------------------------------------------------------------------------- #
# dataset writer
# --------------------------------------------------------------------------- #
def build(out_dir: str, n_per_class: int = 600, seed: int = 0,
          splits=(("train", 0.7), ("val", 0.15), ("test", 0.15))):
    rng = _rng(seed)
    total = n_per_class * len(CLASSES)
    print(f"Generating {total} images ({n_per_class}/class) into {out_dir}")
    counts = {s: 0 for s, _ in splits}
    for ci, cls in enumerate(CLASSES):
        # deterministic but distinct seed space per class
        seeds = rng.randint(0, 2**31 - 1, size=n_per_class)
        # assign split indices
        idx = np.arange(n_per_class)
        rng.shuffle(idx)
        bounds = []
        acc = 0
        for name, frac in splits:
            lo = acc
            acc += int(round(frac * n_per_class))
            bounds.append((name, lo, acc))
        bounds[-1] = (bounds[-1][0], bounds[-1][1], n_per_class)  # absorb rounding
        split_of = {}
        for name, lo, hi in bounds:
            for j in idx[lo:hi]:
                split_of[j] = name
        for j in range(n_per_class):
            split = split_of[j]
            d = os.path.join(out_dir, split, cls)
            os.makedirs(d, exist_ok=True)
            img = make_image(cls, int(seeds[j]) + ci * 7919)
            cv2.imwrite(os.path.join(d, f"{cls}_{j:05d}.png"), img)
            counts[split] += 1
    print("Per-split image counts:", counts)
    return counts


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(here, "dataset"))
    ap.add_argument("--n", type=int, default=600, help="images per class")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build(a.out, n_per_class=a.n, seed=a.seed)
