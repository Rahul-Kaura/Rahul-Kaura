"""Render the koi into a moonlit night pond (Gotham night theme).

Usage: python pond.py koi/koi.gif koi/pond.webp

Takes git-koi's transparent swim GIF and composites every frame into a lit
pond: animated caustics on the pond floor, a soft shadow under the koi, shaded
rocks with moss, lily pads and lotus flowers floating above the koi, ripples,
drifting petals, light rays and a vignette. Everything that moves loops with
the koi's frames. Output is an animated WebP so the lighting keeps full color.
"""

import math
import random
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from traffic import Traffic, RING_W

SS = 2  # supersampling for drawn shapes
rng = random.Random(42)
SPRITES = []


# ---------- helpers ----------

def value_noise(size, cells, seed):
    g = np.random.default_rng(seed).random((cells + 1, cells + 1)).astype(np.float32)
    img = Image.fromarray((g * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC)
    return np.asarray(img, np.float32) / 255.0


def fbm(size, seed, octaves=(4, 8, 16, 32), weights=(0.5, 0.25, 0.15, 0.1)):
    out = np.zeros((size, size), np.float32)
    for i, (c, w) in enumerate(zip(octaves, weights)):
        out += w * value_noise(size, c, seed + i)
    return out / sum(weights)


def rgba_layer(size):
    return Image.new("RGBA", (size * SS, size * SS), (0, 0, 0, 0))


def down(layer, size):
    return layer.resize((size, size), Image.LANCZOS)


def over(dst, src):
    a = src[..., 3:4]
    dst[..., :3] = src[..., :3] * a + dst[..., :3] * (1 - a)
    dst[..., 3:4] = a + dst[..., 3:4] * (1 - a)
    return dst


def arr(img):
    return np.asarray(img.convert("RGBA"), np.float32) / 255.0


# ---------- caustics ----------

def caustic(W, t, scale=1.6):
    y, x = np.mgrid[0:W, 0:W].astype(np.float32) / W
    TAU = 6.28318
    px = (x * scale * TAU) % TAU - 250.0
    py = (y * scale * TAU) % TAU - 250.0
    ix, iy = px.copy(), py.copy()
    c = np.ones_like(px)
    inten = 0.005
    n_iter = 5
    for n in range(n_iter):
        tt = t * (1.0 - (3.5 / float(n + 1)))
        nix = px + np.cos(tt - ix) + np.sin(tt + iy)
        niy = py + np.sin(tt - iy) + np.cos(tt + ix)
        ix, iy = nix, niy
        dx = px / (np.sin(ix + tt) / inten + 1e-6)
        dy = py / (np.cos(iy + tt) / inten + 1e-6)
        c += 1.0 / np.sqrt(dx * dx + dy * dy + 1e-6)
    c /= n_iter
    c = 1.17 - np.power(np.clip(c, 0, None), 1.4)
    return np.clip(np.power(np.abs(c), 8.0), 0, 1)


def looping_caustic(W, phase, period=6.0):
    t = phase * period
    a = caustic(W, t + 20.0)
    b = caustic(W, t - period + 20.0)
    return a * (1 - phase) + b * phase


# ---------- static scene ----------

def wobble(a):
    return 1 + math.sin(a * 3 + 0.7) * 0.025 + math.sin(a * 5 + 2.1) * 0.015 + math.sin(a * 9) * 0.008


def pond_shape(W):
    cx = cy = W / 2
    base = W * 0.44
    y, x = np.mgrid[0:W, 0:W].astype(np.float32)
    ang = np.arctan2(y - cy, x - cx)
    d = np.hypot(x - cx, y - cy)
    wob = np.sin(ang * 3 + 0.7) * 0.025 + np.sin(ang * 5 + 2.1) * 0.015 + np.sin(ang * 9) * 0.008
    return d, base * (1 + wob)


def garden(W):
    n = fbm(W, 3)
    n2 = fbm(W, 9, octaves=(16, 32, 64), weights=(0.5, 0.3, 0.2))
    g = np.stack([0.02 + 0.03 * n, 0.04 + 0.07 * n, 0.05 + 0.07 * n], -1)
    g += (n2[..., None] - 0.5) * 0.04
    img = Image.fromarray(np.clip(g * 255, 0, 255).astype(np.uint8)).convert("RGBA")
    layer = rgba_layer(W)
    d = ImageDraw.Draw(layer)
    for _ in range(420):
        x, y = rng.uniform(0, W), rng.uniform(0, W)
        L = rng.uniform(18, 46)
        a = rng.uniform(0, math.pi * 2)
        shade = rng.uniform(0.6, 1.25)
        col = (int(14 * shade), int(42 * shade), int(36 * shade), 235)
        pts = []
        for k in range(14):
            u = k / 13
            pts.append((u * L, math.sin(u * math.pi) * L * 0.28))
        pts += [(u, -w) for (u, w) in reversed(pts)]
        ca, sa = math.cos(a), math.sin(a)
        d.polygon([((x + px * ca - py * sa) * SS, (y + px * sa + py * ca) * SS) for px, py in pts], fill=col)
        d.line([(x * SS, y * SS), ((x + L * ca) * SS, (y + L * sa) * SS)],
               fill=(int(40 * shade), int(80 * shade), int(66 * shade), 140), width=SS)
    img.alpha_composite(down(layer, W).filter(ImageFilter.GaussianBlur(0.4)))
    return arr(img)


def water_base(W, d, r):
    k = np.clip(d / r, 0, 1)[..., None]
    deep = np.array([0.02, 0.06, 0.11])
    shallow = np.array([0.08, 0.22, 0.30])
    edge = np.array([0.12, 0.28, 0.34])
    kk = k ** 1.6
    col = deep * (1 - kk) + shallow * kk
    e = np.clip((k - 0.85) / 0.15, 0, 1)
    col = col * (1 - e) + edge * e
    peb = fbm(W, 21, octaves=(24, 48, 96), weights=(0.4, 0.35, 0.25))
    col = col * (0.82 + 0.3 * peb[..., None])
    sand = fbm(W, 33, octaves=(6, 12), weights=(0.6, 0.4))
    col += (sand[..., None] - 0.5) * np.array([0.03, 0.03, 0.04])
    return col.astype(np.float32)


def rock_layer(W):
    layer = np.zeros((W, W, 4), np.float32)
    cx = cy = W / 2
    base = W * 0.44
    moss = fbm(W, 55, octaves=(8, 16, 32), weights=(0.5, 0.3, 0.2))
    rocks = []
    a = 0.0
    while a < math.pi * 2:
        rr = rng.uniform(12, 22)
        dist = base * wobble(a) + rr * rng.uniform(0.55, 0.85)
        rocks.append((cx + dist * math.cos(a), cy + dist * math.sin(a), rr, rng.uniform(0.6, 0.95), rng.uniform(0, math.pi)))
        a += rr * 1.25 / (base + rr)
    a = 0.1
    while a < math.pi * 2:
        rr = rng.uniform(16, 28)
        dist = base + rr * 1.6 + rng.uniform(4, 20)
        rocks.append((cx + dist * math.cos(a), cy + dist * math.sin(a), rr, rng.uniform(0.6, 0.9), rng.uniform(0, math.pi)))
        a += rr * 1.4 / (base + rr * 1.5)
    rocks.sort(key=lambda r: -math.hypot(r[0] - cx, r[1] - cy))
    light = np.array([0.55, -0.65, 0.52])
    light /= np.linalg.norm(light)
    yy, xx = np.mgrid[0:W, 0:W].astype(np.float32)
    for (x, y, rr, sq, rot) in rocks:
        x0, x1 = int(max(0, x - rr - 2)), int(min(W, x + rr + 2))
        y0, y1 = int(max(0, y - rr - 2)), int(min(W, y + rr + 2))
        if x1 <= x0 or y1 <= y0:
            continue
        X = xx[y0:y1, x0:x1] - x
        Y = yy[y0:y1, x0:x1] - y
        c, s = math.cos(rot), math.sin(rot)
        u = (X * c + Y * s) / rr
        v = (-X * s + Y * c) / (rr * sq)
        q = u * u + v * v
        z = np.sqrt(np.clip(1 - q, 0, 1))
        nx = u * c - v * s
        ny = u * s + v * c
        lam = np.clip(nx * light[0] + ny * light[1] + z * light[2], 0, 1)
        tone = rng.uniform(0.85, 1.1)
        col = np.array([0.20, 0.21, 0.25]) * tone * (0.2 + 1.0 * lam[..., None])
        col = col + np.power(lam, 18)[..., None] * np.array([0.30, 0.34, 0.42])
        m = moss[y0:y1, x0:x1]
        mossy = (np.clip((m - 0.48) * 6, 0, 1) * np.clip(-ny * 1.2 + 0.2, 0, 1))[..., None] * 0.8
        moss_col = np.array([0.07, 0.15, 0.09]) * (0.5 + 0.8 * lam[..., None])
        col = col * (1 - mossy) + moss_col * mossy
        alpha = ((q < 1) * np.clip((1 - q) * 12, 0, 1))[..., None]
        layer[y0:y1, x0:x1] = over(layer[y0:y1, x0:x1].copy(), np.concatenate([col, alpha], -1))
    return layer


PADS = [  # x, y, r, notch angle deg, flower
    (110, 112, 30, 40, True),
    (402, 386, 34, 220, True),
    (408, 118, 22, 140, False),
    (96, 392, 20, 320, False),
    (150, 66, 15, 80, False),
    (450, 300, 16, 190, False),
]


def pad_sprite(r, notch, flower):
    S = int(r * 2 + 40)
    lay = rgba_layer(S)
    d = ImageDraw.Draw(lay)
    c = S / 2

    def P(x, y):
        return (x * SS, y * SS)

    a0 = math.radians(notch)
    pts = []
    for k in range(0, 361, 3):
        a = math.radians(k)
        da = (a - a0 + math.pi) % (2 * math.pi) - math.pi
        if abs(da) < 0.22:
            continue
        rr = r * (1 + 0.03 * math.sin(a * 5))
        pts.append(P(c + rr * math.cos(a), c + rr * math.sin(a)))
    pts.append(P(c, c))
    d.polygon(pts, fill=(18, 52, 36, 255))
    sheen = rgba_layer(S)
    d = ImageDraw.Draw(sheen)
    for i in range(10):
        f = 1 - i / 10
        rr = r * 0.85 * f
        o = r * 0.15 * (1 - f)
        d.ellipse([*P(c - rr - o, c - rr - o), *P(c + rr - o, c + rr - o)], fill=(26 + i * 3, 66 + i * 4, 46 + i * 3, 28))
    mask = lay.getchannel("A")
    sheen.putalpha(Image.fromarray(np.minimum(np.asarray(sheen.getchannel("A")), np.asarray(mask))))
    lay.alpha_composite(sheen)
    veins = rgba_layer(S)
    d = ImageDraw.Draw(veins)
    for k in range(0, 360, 30):
        a = math.radians(k)
        da = (a - a0 + math.pi) % (2 * math.pi) - math.pi
        if abs(da) < 0.3:
            continue
        d.line([P(c, c), P(c + r * 0.9 * math.cos(a), c + r * 0.9 * math.sin(a))], fill=(52, 98, 66, 110), width=SS)
    lay.alpha_composite(veins)
    d = ImageDraw.Draw(lay)
    d.line(pts[:-1], fill=(222, 170, 40, 120), width=SS)
    if flower:
        fx, fy = c + r * 0.15, c - r * 0.1
        for ring, (n, L, Wd, base_col, tip) in enumerate([
            (8, r * 0.75, r * 0.32, (255, 246, 222), (196, 142, 28)),
            (8, r * 0.58, r * 0.28, (255, 250, 232), (222, 170, 40)),
            (6, r * 0.38, r * 0.22, (255, 252, 240), (240, 196, 80)),
        ]):
            off = ring * 0.4
            for k in range(n):
                a = 2 * math.pi * k / n + off
                ca, sa = math.cos(a), math.sin(a)
                for g in range(8):
                    f = 1 - g / 8
                    col = tuple(int(base_col[j] * (1 - f) + tip[j] * f) for j in range(3)) + (255,)
                    LL, WW = L * f, Wd * f
                    p2 = [(s / 15 * LL, math.sin(s / 15 * math.pi) ** 0.8 * WW / 2) for s in range(16)]
                    p2 += [(u, -w) for (u, w) in reversed(p2)]
                    d.polygon([P(fx + px * ca - py * sa, fy + px * sa + py * ca) for px, py in p2], fill=col)
        d.ellipse([*P(fx - r * 0.13, fy - r * 0.13), *P(fx + r * 0.13, fy + r * 0.13)], fill=(250, 210, 90, 255))
        d.ellipse([*P(fx - r * 0.06, fy - r * 0.08), *P(fx + r * 0.02, fy)], fill=(255, 240, 170, 255))
    return down(lay, S)


def pads_layer(W, phase):
    img = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    shadow = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    for i, (x, y, r, notch, flower) in enumerate(PADS):
        spr = SPRITES[i]
        bob = 1.6 * math.sin(2 * math.pi * phase + i * 1.7)
        sway = 1.2 * math.cos(2 * math.pi * phase + i * 2.3)
        px, py = int(round(x - spr.width / 2 + sway)), int(round(y - spr.height / 2 + bob))
        sh = Image.new("RGBA", spr.size, (0, 30, 30, 0))
        sh.putalpha(spr.getchannel("A").point(lambda v: int(v * 0.45)))
        shadow.alpha_composite(sh, (max(0, px + 7), max(0, py + 9)))
        img.alpha_composite(spr, (max(0, px), max(0, py)))
    return arr(img), arr(shadow.filter(ImageFilter.GaussianBlur(4)))


PETALS = [(0.0, 150, 0.8), (0.33, 120, 1.3), (0.66, 175, 0.5)]


def petals_layer(W, phase):
    lay = rgba_layer(W)
    d = ImageDraw.Draw(lay)
    c = W / 2
    for off, rad, rot in PETALS:
        a = 2 * math.pi * (phase + off)
        x = c + rad * math.cos(a) * 0.95
        y = c + rad * math.sin(a + 0.6) * 0.8
        ang = a * 2 + rot
        L, Wd = 9, 5
        pts = [(s / 11 * L - L / 2, math.sin(s / 11 * math.pi) * Wd / 2) for s in range(12)]
        pts += [(u, -w) for (u, w) in reversed(pts)]
        ca, sa = math.cos(ang), math.sin(ang)
        d.polygon([((x + px * ca - py * sa) * SS, (y + px * sa + py * ca) * SS) for px, py in pts], fill=(232, 186, 70, 235))
    return arr(down(lay, W))


def ripples_layer(W, phase):
    lay = rgba_layer(W)
    d = ImageDraw.Draw(lay)
    for (x, y, off) in [(190, 300, 0.0), (330, 210, 0.5), (260, 150, 0.25)]:
        for ring in range(3):
            p = (phase + off + ring * 0.12) % 1.0
            r = 4 + p * 46
            a = int(110 * (1 - p) ** 1.5)
            if a <= 2:
                continue
            d.ellipse([(x - r) * SS, (y - r * 0.62) * SS, (x + r) * SS, (y + r * 0.62) * SS], outline=(200, 214, 240, int(a * 0.7)), width=SS)
    return arr(down(lay, W))


def light_rays(W, phase):
    y, x = np.mgrid[0:W, 0:W].astype(np.float32) / W
    s = x * 0.8 + y * 0.6
    rays = (np.maximum(0, np.sin(s * 22 + 2 * math.pi * phase)) ** 6 * 0.5 +
            np.maximum(0, np.sin(s * 9 - 2 * math.pi * phase + 1.3)) ** 8 * 0.5)
    return rays * np.clip(1.2 - (x + y), 0, 1)


MOON_NOISE = None


def moon_glint(W, phase):
    """Moon reflection: a soft glow broken into shimmering horizontal streaks."""
    y, x = np.mgrid[0:W, 0:W].astype(np.float32)
    mx, my = W * 0.64, W * 0.34
    glow = np.exp(-(((x - mx) / 70) ** 2 + ((y - my) / 46) ** 2))
    streak = 0.5 + 0.5 * np.sin(y * 0.55 + 2 * math.pi * phase * 2 + np.sin(x * 0.05 + 2 * math.pi * phase) * 2 + MOON_NOISE * 9)
    core = np.exp(-(((x - mx) / 30) ** 2 + ((y - my) / 22) ** 2))
    return glow * (0.35 + 0.65 * streak ** 3) * 0.5 + core * 0.35


FIREFLIES = [(rng.uniform(30, 482), rng.uniform(30, 482), rng.uniform(6, 18), rng.random(), rng.choice([1, 2])) for _ in range(16)]


def fireflies(W, phase):
    lay = Image.new("L", (W, W), 0)
    core = Image.new("L", (W, W), 0)
    d, dc = ImageDraw.Draw(lay), ImageDraw.Draw(core)
    for (x, y, rad, off, k) in FIREFLIES:
        a = 2 * math.pi * (phase * k + off)
        fx, fy = x + rad * math.cos(a), y + rad * 0.6 * math.sin(2 * a)
        tw = 0.5 + 0.5 * math.sin(2 * math.pi * (phase * 3 + off))
        b = int(255 * (0.25 + 0.75 * tw))
        d.ellipse([fx - 7, fy - 7, fx + 7, fy + 7], fill=b)
        dc.ellipse([fx - 1.6, fy - 1.6, fx + 1.6, fy + 1.6], fill=b)
    glow = np.asarray(lay.filter(ImageFilter.GaussianBlur(6)), np.float32) / 255.0
    c = np.asarray(core.filter(ImageFilter.GaussianBlur(0.8)), np.float32) / 255.0
    return glow * 0.55 + c


CANVAS_FLIES = [(rng.uniform(20, 1180), rng.uniform(20, 740), rng.uniform(6, 22), rng.random(), rng.choice([1, 2])) for _ in range(34)]


def fireflies_canvas(CW, CH, phase):
    lay = Image.new("L", (CW, CH), 0)
    core = Image.new("L", (CW, CH), 0)
    d, dc = ImageDraw.Draw(lay), ImageDraw.Draw(core)
    for (x, y, rad, off, k) in CANVAS_FLIES:
        a = 2 * math.pi * (phase * k + off)
        fx, fy = x + rad * math.cos(a), y + rad * 0.6 * math.sin(2 * a)
        tw = 0.5 + 0.5 * math.sin(2 * math.pi * (phase * 3 + off))
        b = int(255 * (0.25 + 0.75 * tw))
        d.ellipse([fx - 7, fy - 7, fx + 7, fy + 7], fill=b)
        dc.ellipse([fx - 1.6, fy - 1.6, fx + 1.6, fy + 1.6], fill=b)
    glow = np.asarray(lay.filter(ImageFilter.GaussianBlur(6)), np.float32) / 255.0
    c = np.asarray(core.filter(ImageFilter.GaussianBlur(0.8)), np.float32) / 255.0
    return glow * 0.55 + c


def vignette(W):
    y, x = np.mgrid[0:W, 0:W].astype(np.float32) / W - 0.5
    d = np.sqrt(x * x + y * y) / 0.707
    return np.clip(1 - 0.75 * d ** 2.0, 0, 1)



# ---------- top-down night city around the pond ----------

def city(CW, CH, cx, cy, park_r, traffic, AX=0):
    """Static rooftops and streets, plus lists of lights that animate."""
    r = random.Random(1939)
    S2 = 2
    img = Image.new("RGB", (CW * S2, CH * S2), (6, 7, 9))
    shadow = Image.new("L", (CW * S2, CH * S2), 0)
    d = ImageDraw.Draw(img)
    ds = ImageDraw.Draw(shadow)
    windows, lamps, streets = [], [], []

    def S(v):
        return int(v * S2)

    # street grid
    xs, x = [], r.uniform(-40, 0)
    while x < CW:
        xs.append(x)
        x += r.uniform(130, 175)
    ys, y = [], r.uniform(-40, 0)
    while y < CH:
        ys.append(y)
        y += r.uniform(115, 150)
    SW = 18
    blocks = []
    for i in range(len(xs)):
        for j in range(len(ys)):
            x0 = xs[i] + SW / 2
            x1 = (xs[i + 1] if i + 1 < len(xs) else CW + 60) - SW / 2
            y0 = ys[j] + SW / 2
            y1 = (ys[j + 1] if j + 1 < len(ys) else CH + 60) - SW / 2
            blocks.append((x0, y0, x1, y1))
    for x in xs:
        streets.append(("v", x))
        for yy in range(0, CH, 10):
            d.rectangle([S(x - 0.6), S(yy), S(x + 0.6), S(yy + 5)], fill=(58, 48, 20))
        for yy in range(int(r.uniform(0, 80)), CH, 120):
            lamps.append((x - SW / 2 + 2, yy, r.random()))
    for y in ys:
        streets.append(("h", y))
        for xx in range(0, CW, 10):
            d.rectangle([S(xx), S(y - 0.6), S(xx + 5), S(y + 0.6)], fill=(58, 48, 20))
        for xx in range(int(r.uniform(0, 80)), CW, 120):
            lamps.append((xx, y - SW / 2 + 2, r.random()))

    traffic.draw_ground(img, S2)
    if AX:
        draw_airport(img, CW, CH, AX, S2)
    rmask = traffic.road_mask(CW, CH, pad=6)
    d = ImageDraw.Draw(img)

    def split(x0, y0, x1, y1, depth):
        w, h = x1 - x0, y1 - y0
        if depth > 2 or (w < 70 and h < 70) or r.random() < 0.18 * depth:
            return [(x0, y0, x1, y1)]
        gap = 4
        if w > h:
            m = r.uniform(x0 + w * 0.35, x0 + w * 0.65)
            return split(x0, y0, m - gap / 2, y1, depth + 1) + split(m + gap / 2, y0, x1, y1, depth + 1)
        m = r.uniform(y0 + h * 0.35, y0 + h * 0.65)
        return split(x0, y0, x1, m - gap / 2, depth + 1) + split(x0, m + gap / 2, x1, y1, depth + 1)

    roofs = []
    for b in blocks:
        for (x0, y0, x1, y1) in split(*b, 0):
            bx, by = (x0 + x1) / 2, (y0 + y1) / 2
            probes = [(bx, by), (x0, y0), (x1, y0), (x0, y1), (x1, y1), (bx, y0), (bx, y1), (x0, by), (x1, by)]
            if any(rmask[min(CH - 1, max(0, int(py_))), min(CW - 1, max(0, int(px_)))] for px_, py_ in probes):
                continue
            if x0 < AX or x1 > CW - AX:
                continue
            roofs.append((x0, y0, x1, y1))
    for (x0, y0, x1, y1) in roofs:
        hgt = r.uniform(5, 13)
        ds.rectangle([S(x0 + hgt * 0.7), S(y0 + hgt), S(x1 + hgt * 0.7), S(y1 + hgt)], fill=200)
    img.paste((3, 3, 5), mask=shadow.filter(ImageFilter.GaussianBlur(S2 * 2)))
    d = ImageDraw.Draw(img)
    for (x0, y0, x1, y1) in roofs:
        t = r.uniform(0.75, 1.25)
        base = (int(13 * t), int(14 * t), int(19 * t))
        d.rectangle([S(x0), S(y0), S(x1), S(y1)], fill=base)
        # parapet: lit by the moon on the top-right
        d.rectangle([S(x0), S(y0), S(x1), S(y0 + 2)], fill=(int(30 * t), int(32 * t), int(42 * t)))
        d.rectangle([S(x1 - 2), S(y0), S(x1), S(y1)], fill=(int(26 * t), int(28 * t), int(37 * t)))
        d.rectangle([S(x0), S(y1 - 2), S(x1), S(y1)], fill=(int(6 * t), int(6 * t), int(9 * t)))
        w, h = x1 - x0, y1 - y0
        kind = r.random()
        if kind < 0.45 and w > 30 and h > 30:
            # skylight grid
            for wx in range(int(x0 + 6), int(x1 - 8), 9):
                for wy in range(int(y0 + 6), int(y1 - 8), 9):
                    if r.random() < 0.55:
                        windows.append((wx, wy, r.random(), r.uniform(0.6, 1.0)))
                        d.rectangle([S(wx), S(wy), S(wx + 4), S(wy + 4)], fill=(30, 28, 22))
        for _ in range(r.randint(0, 3)):
            aw, ah = r.uniform(6, 14), r.uniform(5, 10)
            ax, ay = r.uniform(x0 + 4, max(x0 + 5, x1 - aw - 4)), r.uniform(y0 + 4, max(y0 + 5, y1 - ah - 4))
            d.rectangle([S(ax + 2), S(ay + 2), S(ax + aw + 2), S(ay + ah + 2)], fill=(8, 8, 10))
            d.rectangle([S(ax), S(ay), S(ax + aw), S(ay + ah)], fill=(30, 32, 39))
            d.rectangle([S(ax), S(ay), S(ax + aw), S(ay + 1.5)], fill=(46, 48, 58))
        if r.random() < 0.18 and w > 36 and h > 36:
            tx, ty, tr = r.uniform(x0 + 12, x1 - 12), r.uniform(y0 + 12, y1 - 12), r.uniform(6, 9)
            d.ellipse([S(tx - tr + 3), S(ty - tr + 3), S(tx + tr + 3), S(ty + tr + 3)], fill=(6, 6, 8))
            d.ellipse([S(tx - tr), S(ty - tr), S(tx + tr), S(ty + tr)], fill=(44, 34, 26))
            d.ellipse([S(tx - tr * 0.6), S(ty - tr * 0.7), S(tx + tr * 0.3), S(ty)], fill=(66, 52, 38))
    out = np.asarray(img.resize((CW, CH), Image.LANCZOS), np.float32) / 255.0
    lamps = [l for l in lamps if AX + 4 < l[0] < CW - AX - 4]
    return out, windows, lamps, streets, xs, ys


CARS = []


def city_lights(CW, CH, phase, windows, lamps, xs, ys):
    glow = Image.new("L", (CW, CH), 0)
    core = Image.new("RGB", (CW, CH), (0, 0, 0))
    dg, dc = ImageDraw.Draw(glow), ImageDraw.Draw(core)
    for (wx, wy, off, b) in windows:
        on = 0.5 + 0.5 * math.sin(2 * math.pi * (phase * (1 + int(off * 3)) + off * 7))
        v = b * (0.35 + 0.65 * (on > 0.25))
        dc.rectangle([wx, wy, wx + 2, wy + 2], fill=(int(250 * v), int(196 * v), int(80 * v)))
        dg.rectangle([wx - 2, wy - 2, wx + 5, wy + 5], fill=int(90 * v))
    for (lx, ly, off) in lamps:
        f = 0.85 + 0.15 * math.sin(2 * math.pi * (phase * 2 + off))
        dg.ellipse([lx - 7, ly - 7, lx + 7, ly + 7], fill=int(120 * f))
        dc.ellipse([lx - 1.5, ly - 1.5, lx + 1.5, ly + 1.5], fill=(255, 228, 150))
    g = np.asarray(glow.filter(ImageFilter.GaussianBlur(5)), np.float32)[..., None] / 255.0
    c = np.asarray(core.filter(ImageFilter.GaussianBlur(0.5)), np.float32) / 255.0
    return g * np.array([1.0, 0.82, 0.45], np.float32) * 0.75 + c


def signal_spotlights(CW, CH, phase):
    """Two signal beams sweeping over the city, each ending in a bright disc of light."""
    lay = Image.new("L", (CW, CH), 0)
    disc = Image.new("L", (CW, CH), 0)
    d, dd = ImageDraw.Draw(lay), ImageDraw.Draw(disc)
    beams = [((-80, -60), (0.22, 0.30), (0.20, 0.18), 0.0),
             ((CW + 80, CH + 60), (0.78, 0.70), (0.20, 0.20), 0.5)]
    for (ox, oy), (bx, by), (ax, ay), off in beams:
        a = 2 * math.pi * (phase + off)
        tx = CW * (bx + ax * math.sin(a))
        ty = CH * (by + ay * math.sin(2 * a + 0.8))
        ang = math.atan2(ty - oy, tx - ox)
        dist = math.hypot(tx - ox, ty - oy)
        rr = 58
        pxv, pyv = -math.sin(ang), math.cos(ang)
        d.polygon([(ox + pxv * 10, oy + pyv * 10), (ox - pxv * 10, oy - pyv * 10),
                   (tx - pxv * rr, ty - pyv * rr), (tx + pxv * rr, ty + pyv * rr)], fill=34)
        dd.ellipse([tx - rr, ty - rr * 0.82, tx + rr, ty + rr * 0.82], fill=185)
        dd.ellipse([tx - rr * 0.8, ty - rr * 0.66, tx + rr * 0.8, ty + rr * 0.66], fill=95)
        dd.ellipse([tx - rr * 0.55, ty - rr * 0.45, tx + rr * 0.55, ty + rr * 0.45], fill=125)
    beam = np.asarray(lay.filter(ImageFilter.GaussianBlur(14)), np.float32)[..., None] / 255.0
    spot = np.asarray(disc.filter(ImageFilter.GaussianBlur(5)), np.float32)[..., None] / 255.0
    return beam, spot


def pond_lights(W):
    """Warm underwater lights set around the pond's edge, shining inward."""
    y, x = np.mgrid[0:W, 0:W].astype(np.float32)
    c = W / 2
    out = np.zeros((W, W), np.float32)
    for k in range(8):
        a = 2 * math.pi * k / 8 + 0.2
        lx, ly = c + 190 * math.cos(a), c + 190 * math.sin(a)
        d2 = (x - lx) ** 2 + (y - ly) ** 2
        out += np.exp(-d2 / (2 * 38 ** 2)) * 0.9 + np.exp(-d2 / (2 * 9 ** 2)) * 0.8
        return out[..., None]


def plane_sprite():
    """Top-down airliner facing +x, drawn at 3x, dark with a soft highlight."""
    S = 3
    L, span = 92, 86
    img = Image.new("RGBA", (int((L + 10) * S), int((span + 10) * S)), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    ox, oy = 5 * S, (span / 2 + 5) * S

    k = L / 58.0

    def P(x, y):
        return (ox + x * k * S, oy + y * k * S)

    body, hi, wing = (112, 116, 128, 255), (168, 172, 184, 255), (88, 92, 104, 255)
    # wings (swept back)
    d.polygon([P(30, -2), P(17, -27), P(12, -27), P(20, -2)], fill=wing)
    d.polygon([P(30, 2), P(17, 27), P(12, 27), P(20, 2)], fill=wing)
    # tailplane
    d.polygon([P(8, -1.5), P(2, -11), P(-1, -11), P(3, -1.5)], fill=wing)
    d.polygon([P(8, 1.5), P(2, 11), P(-1, 11), P(3, 1.5)], fill=wing)
    # engines
    for side in (-11, 11):
        d.rounded_rectangle([*P(21, side - 1.8), *P(29, side + 1.8)], radius=1.5 * S, fill=(60, 63, 72, 255))
    # fuselage
    d.rounded_rectangle([*P(0, -3.2), *P(58, 3.2)], radius=3.2 * k * S, fill=body)
    d.rounded_rectangle([*P(4, -1.4), *P(54, -0.2)], radius=0.6 * k * S, fill=hi)
    d.ellipse([*P(51, -2.2), *P(57, 2.2)], fill=(28, 34, 46, 255))
    return img, L, span


PLANE = None


def planes_layer(CW, CH, phase):
    """Two planes gliding over the city corners with steady navigation lights."""
    global PLANE
    if PLANE is None:
        PLANE = plane_sprite()
    spr, L, span = PLANE
    body = Image.new("RGBA", (CW, CH), (0, 0, 0, 0))
    shadow = Image.new("L", (CW, CH), 0)
    glow = Image.new("RGB", (CW, CH), (0, 0, 0))
    core = Image.new("RGB", (CW, CH), (0, 0, 0))
    dg, dc = ImageDraw.Draw(glow), ImageDraw.Draw(core)
    flights = [((-90, 360), (580, -90), 0.0), ((CW + 90, CH - 360), (CW - 580, CH + 90), 0.5)]
    for (x0, y0), (x1, y1), off in flights:
        t = (phase + off) % 1.0
        x, y = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
        h = math.atan2(y1 - y0, x1 - x0)
        rot = spr.rotate(-math.degrees(h), resample=Image.BICUBIC, expand=True)
        small = rot.resize((rot.width // 3, rot.height // 3), Image.LANCZOS)
        px, py = int(x - small.width / 2), int(y - small.height / 2)
        a = small.getchannel("A")
        for (dx, dy, img) in ((46, 58, None),):
            sx, sy = px + dx, py + dy
            if -small.width < sx < CW and -small.height < sy < CH:
                shadow.paste(255, (sx, sy, sx + small.width, sy + small.height), a)
        if -small.width < px < CW and -small.height < py < CH:
            body.alpha_composite(small, (max(0, px), max(0, py))) if px >= 0 and py >= 0 else None
            if px < 0 or py < 0:
                cx0, cy0 = max(0, -px), max(0, -py)
                if cx0 < small.width and cy0 < small.height:
                    body.alpha_composite(small.crop((cx0, cy0, small.width, small.height)), (max(0, px), max(0, py)))
        ux, uy = math.cos(h), math.sin(h)
        vx, vy = -uy, ux
        # steady navigation lights: red on the left wingtip, green on the right, white at the tail
        for (lx, ly, col) in ((x + ux * (17 * L / 58 - L / 2) - vx * span / 2, y + uy * (17 * L / 58 - L / 2) - vy * span / 2, (255, 70, 60)),
                              (x + ux * (17 * L / 58 - L / 2) + vx * span / 2, y + uy * (17 * L / 58 - L / 2) + vy * span / 2, (70, 255, 140)),
                              (x - ux * L / 2, y - uy * L / 2, (255, 250, 235))):
            dg.ellipse([lx - 7, ly - 7, lx + 7, ly + 7], fill=tuple(int(c * 0.45) for c in col))
            dc.ellipse([lx - 1.3, ly - 1.3, lx + 1.3, ly + 1.3], fill=col)
        # soft landing lights glow ahead of the nose
        nx, ny = x + ux * L / 2, y + uy * L / 2
        dg.ellipse([nx - 10, ny - 10, nx + 10, ny + 10], fill=(60, 56, 44))
    sh = np.asarray(shadow.filter(ImageFilter.GaussianBlur(6)), np.float32)[..., None] / 255.0
    b = np.asarray(body, np.float32) / 255.0
    lights = (np.asarray(glow.filter(ImageFilter.GaussianBlur(5)), np.float32) / 255.0 +
              np.asarray(core.filter(ImageFilter.GaussianBlur(0.5)), np.float32) / 255.0)
    return sh, b, lights


# ---------- airport at both ends ----------

RWY_W, TWY_W = 44, 16


def airport_layout(CW, AX):
    left = dict(rx=AX * 0.5, tx=AX * 0.5 + 62)
    right = dict(rx=CW - AX * 0.5, tx=CW - AX * 0.5 - 62)
    return left, right


def draw_airport(img, CW, CH, AX, S):
    d = ImageDraw.Draw(img)
    ground = (9, 13, 12)
    d.rectangle([0, 0, AX * S, CH * S], fill=ground)
    d.rectangle([(CW - AX) * S, 0, CW * S, CH * S], fill=ground)
    for side in airport_layout(CW, AX):
        rx, tx = side["rx"], side["tx"]
        # taxiway and links to the runway
        d.rectangle([(tx - TWY_W / 2) * S, 0, (tx + TWY_W / 2) * S, CH * S], fill=(22, 23, 28))
        d.line([(tx * S, 0), (tx * S, CH * S)], fill=(150, 120, 34), width=max(1, S // 2))
        for y in (CH * 0.28, CH * 0.72):
            d.polygon([((rx) * S, (y - 30) * S), ((tx) * S, (y - 4) * S), ((tx) * S, (y + 12) * S), ((rx) * S, (y - 14) * S)], fill=(22, 23, 28))
        # runway
        d.rectangle([(rx - RWY_W / 2) * S, 0, (rx + RWY_W / 2) * S, CH * S], fill=(26, 27, 32))
        for xe in (rx - RWY_W / 2 + 1.5, rx + RWY_W / 2 - 1.5):
            d.line([(xe * S, 0), (xe * S, CH * S)], fill=(150, 150, 156), width=S)
        for y in range(0, CH, 34):
            d.rectangle([(rx - 0.9) * S, y * S, (rx + 0.9) * S, (y + 18) * S], fill=(176, 176, 182))
        # threshold piano keys and touchdown bars near each end
        for y0 in (26, CH - 44):
            for k in range(-4, 5):
                if k == 0:
                    continue
                xk = rx + k * 4.2
                d.rectangle([(xk - 1.3) * S, y0 * S, (xk + 1.3) * S, (y0 + 18) * S], fill=(176, 176, 182))
        for y0 in (78, CH - 96):
            for xo in (-12, 9):
                d.rectangle([(rx + xo) * S, y0 * S, (rx + xo + 3) * S, (y0 + 18) * S], fill=(150, 150, 156))


def airport_lights(CW, CH, AX):
    glow = Image.new("RGB", (CW, CH), (0, 0, 0))
    core = Image.new("RGB", (CW, CH), (0, 0, 0))
    dg, dc = ImageDraw.Draw(glow), ImageDraw.Draw(core)

    def lamp(x, y, col, g=5, r=1.1, k=0.45):
        dg.ellipse([x - g, y - g, x + g, y + g], fill=tuple(int(c * k) for c in col))
        dc.ellipse([x - r, y - r, x + r, y + r], fill=col)

    for side in airport_layout(CW, AX):
        rx, tx = side["rx"], side["tx"]
        for y in range(8, CH, 40):
            for xe in (rx - RWY_W / 2 - 3, rx + RWY_W / 2 + 3):
                lamp(xe, y, (255, 246, 214))
        for y in range(14, CH, 48):
            for xe in (tx - TWY_W / 2 - 3, tx + TWY_W / 2 + 3):
                lamp(xe, y, (70, 120, 255), g=4, r=1.0, k=0.5)
        for k in range(-5, 6):
            lamp(rx + k * 4, 18, (60, 255, 120), g=4, r=1.0)
            lamp(rx + k * 4, CH - 18, (255, 60, 50), g=4, r=1.0)
    g = np.asarray(glow.filter(ImageFilter.GaussianBlur(3)), np.float32) / 255.0
    c = np.asarray(core.filter(ImageFilter.GaussianBlur(0.5)), np.float32) / 255.0
    return g + c


def airport_planes(CW, CH, AX, phase):
    """A takeoff on the left runway and a landing on the right one, looping seamlessly."""
    global PLANE
    if PLANE is None:
        PLANE = plane_sprite()
    spr, L, span = PLANE
    left, right = airport_layout(CW, AX)
    body = Image.new("RGBA", (CW, CH), (0, 0, 0, 0))
    shadow = Image.new("L", (CW, CH), 0)
    glow = Image.new("RGB", (CW, CH), (0, 0, 0))
    core = Image.new("RGB", (CW, CH), (0, 0, 0))
    dg, dc = ImageDraw.Draw(glow), ImageDraw.Draw(core)
    span_y = CH + 130
    flights = []
    # takeoff: rolls in slowly from the bottom, speeds up, lifts off near the top
    t = phase
    y = CH + 50 - span_y * t ** 1.55
    alt = min(1.0, max(0.0, (CH * 0.45 - y) / (CH * 0.6)))
    flights.append((left["rx"], y, -math.pi / 2, alt))
    # landing: comes in high from the top, touches down, slows to a roll
    t = (phase + 0.5) % 1.0
    y = -80 + span_y * (1 - (1 - t) ** 1.55)
    alt = min(1.0, max(0.0, (CH * 0.38 - y) / (CH * 0.5)))
    flights.append((right["rx"], y, math.pi / 2, alt))
    for (x, y, h, alt) in flights:
        scale = 1.0 + 0.45 * alt
        rot = spr.rotate(-math.degrees(h), resample=Image.BICUBIC, expand=True)
        w, hh = int(rot.width / 3 * scale), int(rot.height / 3 * scale)
        small = rot.resize((max(1, w), max(1, hh)), Image.LANCZOS)
        px, py = int(x - w / 2), int(y - hh / 2)
        a = small.getchannel("A").point(lambda v: int(v * (1 - 0.55 * alt)))
        sx, sy = px + int(6 + 70 * alt), py + int(8 + 90 * alt)
        sm = small.getchannel("A").resize((int(w / scale), int(hh / scale)), Image.LANCZOS).point(lambda v: int(v * (1 - 0.6 * alt)))
        _paste_mask(shadow, sm, sx + (w - sm.width) // 2, sy + (hh - sm.height) // 2)
        _paste_rgba(body, small, px, py)
        ux, uy = math.cos(h), math.sin(h)
        vx, vy = -uy, ux
        Ls, Ss = L * scale, span * scale
        wing = 17 * L / 58 * scale
        for (lx, ly, col) in ((x + ux * (wing - Ls / 2) - vx * Ss / 2, y + uy * (wing - Ls / 2) - vy * Ss / 2, (255, 70, 60)),
                              (x + ux * (wing - Ls / 2) + vx * Ss / 2, y + uy * (wing - Ls / 2) + vy * Ss / 2, (70, 255, 140)),
                              (x - ux * Ls / 2, y - uy * Ls / 2, (255, 250, 235))):
            dg.ellipse([lx - 7, ly - 7, lx + 7, ly + 7], fill=tuple(int(c * 0.45) for c in col))
            dc.ellipse([lx - 1.4, ly - 1.4, lx + 1.4, ly + 1.4], fill=col)
        nx, ny = x + ux * Ls / 2, y + uy * Ls / 2
        bl, bw = 90, 26
        dg.polygon([(nx - vx * 4, ny - vy * 4), (nx + vx * 4, ny + vy * 4),
                    (nx + ux * bl + vx * bw, ny + uy * bl + vy * bw), (nx + ux * bl - vx * bw, ny + uy * bl - vy * bw)], fill=(70, 64, 46))
    sh = np.asarray(shadow.filter(ImageFilter.GaussianBlur(5)), np.float32)[..., None] / 255.0
    b = np.asarray(body, np.float32) / 255.0
    lights = (np.asarray(glow.filter(ImageFilter.GaussianBlur(6)), np.float32) / 255.0 +
              np.asarray(core.filter(ImageFilter.GaussianBlur(0.5)), np.float32) / 255.0)
    return sh, b, lights


def _paste_rgba(dst, src, px, py):
    x0, y0 = max(0, -px), max(0, -py)
    if x0 >= src.width or y0 >= src.height or px >= dst.width or py >= dst.height:
        return
    dst.alpha_composite(src.crop((x0, y0, src.width, src.height)), (max(0, px), max(0, py)))


def _paste_mask(dst, mask, px, py):
    x0, y0 = max(0, -px), max(0, -py)
    if x0 >= mask.width or y0 >= mask.height or px >= dst.width or py >= dst.height:
        return
    m = mask.crop((x0, y0, mask.width, mask.height))
    dst.paste(255, (max(0, px), max(0, py), max(0, px) + m.width, max(0, py) + m.height), m)


# ---------- main ----------

def main(src, dst):
    koi = Image.open(src)
    n = koi.n_frames
    duration = 110  # slower, calmer swim than git-koi's 70 ms
    W = koi.size[0]

    global MOON_NOISE
    MOON_NOISE = fbm(W, 77, octaves=(6, 12, 24), weights=(0.5, 0.3, 0.2))
    SPRITES[:] = [pad_sprite(r, notch, fl) for (_, _, r, notch, fl) in PADS]

    d, r = pond_shape(W)
    inside = np.clip((r - d) / 2.0, 0, 1)[..., None]
    bg = garden(W)
    water = water_base(W, d, r)
    rocks = rock_layer(W)
    rsh = Image.fromarray((rocks[..., 3] * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(10))
    rshadow = (np.asarray(rsh, np.float32) / 255.0)[..., None]
    vig = vignette(W)[..., None]
    depth = np.clip(d / r, 0, 1)[..., None]
    warm = np.array([0.70, 0.80, 1.0], np.float32)
    caus_col = np.array([1.0, 0.80, 0.40], np.float32)
    gold = np.array([1.0, 0.76, 0.22], np.float32)
    moon_col = np.array([0.80, 0.88, 1.0], np.float32)

    # wide canvas: the pond sits in a night garden that fades into the page
    CW, CH = 1200, 760
    OX, OY = (CW - W) // 2, (CH - W) // 2
    page = np.array([13, 17, 23], np.float32) / 255.0
    pcx, pcy = OX + W / 2, OY + W / 2
    traffic = Traffic(CW, CH, pcx, pcy, n_frames=n, ring_only=True)
    AX = int(pcx - traffic.ra - RING_W / 2 - 14)
    park_r = 0
    cityimg, windows, lamps, streets, sxs, sys_ = city(CW, CH, pcx, pcy, park_r, traffic, AX)
    park = garden(max(CW, CH))[:CH, :CW, :3] * 0.8
    yy, xx = np.mgrid[0:CH, 0:CW].astype(np.float32)
    pd = np.hypot(xx - (OX + W / 2), yy - (OY + W / 2))
    feather = np.clip((256 - pd) / 34, 0, 1)[..., None]
    ex = (xx - CW / 2) / (CW / 2)
    ey = (yy - CH / 2) / (CH / 2)
    e = (ex ** 4 + ey ** 4) ** 0.25
    t = np.clip((e - 0.80) / (0.995 - 0.80), 0, 1)
    edge_fade = (t * t * (3 - 2 * t))[..., None]
    sd = traffic.ring_sdf_grid(xx, yy)
    parkmask = np.clip((-sd - RING_W / 2 - 3) / 16, 0, 1)[..., None]
    deck, deck_sh = traffic.deck_layer(CW, CH)
    apt_lights = airport_lights(CW, CH, AX)
    da = deck[..., 3:4]
    plights = pond_lights(W)
    spot_col = np.array([1.0, 0.86, 0.52], np.float32)
    outer = park * parkmask + cityimg * (1 - parkmask)
    shade = np.ones_like(parkmask)

    # koi centroids for the sparkle trail
    cents = []
    for i in range(n):
        koi.seek(i)
        a = np.asarray(koi.convert("RGBA"))[..., 3] > 0
        ys, xs = np.nonzero(a)
        cents.append((xs.mean(), ys.mean()) if len(xs) else (W / 2, W / 2))

    def trail(i, phase):
        lay = Image.new("L", (W, W), 0)
        core = Image.new("L", (W, W), 0)
        dl, dc = ImageDraw.Draw(lay), ImageDraw.Draw(core)
        tr = random.Random(7)
        for back in range(3, 26):
            cx, cy = cents[(i - back) % n]
            fade = (1 - back / 26) ** 1.5
            for _ in range(2):
                jx, jy = tr.uniform(-16, 16), tr.uniform(-16, 16)
                tw = 0.5 + 0.5 * math.sin(back * 1.9 + phase * 12.566)
                b = int(255 * fade * (0.35 + 0.65 * tw))
                rr = 1.0 + 1.6 * fade
                dc.ellipse([cx + jx - rr, cy + jy - rr, cx + jx + rr, cy + jy + rr], fill=b)
                dl.ellipse([cx + jx - 6, cy + jy - 6, cx + jx + 6, cy + jy + 6], fill=int(b * 0.6))
        g = np.asarray(lay.filter(ImageFilter.GaussianBlur(5)), np.float32) / 255.0
        c = np.asarray(core.filter(ImageFilter.GaussianBlur(0.7)), np.float32) / 255.0
        return (g * 0.6 + c)[..., None]

    frames = []
    for i in range(n):
        phase = i / n
        koi.seek(i)
        k = arr(koi)

        cst = looping_caustic(W, phase)[..., None]
        floor = water + cst * caus_col * (0.15 + 0.14 * depth)
        floor = floor + plights * np.array([1.0, 0.78, 0.36], np.float32) * 0.22 * (0.9 + 0.1 * math.sin(2 * math.pi * phase * 2))

        ka = Image.fromarray((k[..., 3] * 255).astype(np.uint8))
        ksh = np.asarray(ka.filter(ImageFilter.GaussianBlur(5)), np.float32) / 255.0
        ksh = np.roll(np.roll(ksh, 14, 0), 10, 1)[..., None]
        floor = floor * (1 - 0.6 * ksh)

        kc = np.clip(k[..., :3] * 1.1, 0, 1) + cst * caus_col * 0.06
        ka3 = k[..., 3:4]
        pulse = 0.78 + 0.22 * math.sin(2 * math.pi * phase * 3)
        inner = np.asarray(ka.filter(ImageFilter.GaussianBlur(4)), np.float32)[..., None] / 255.0
        outer_glow = np.asarray(ka.filter(ImageFilter.GaussianBlur(18)), np.float32)[..., None] / 255.0
        floor = floor + outer_glow * gold * 0.45 * pulse
        scene = floor * (1 - ka3) + kc * ka3
        scene = scene + (inner * 0.55 + outer_glow * 0.5) * gold * pulse * (1 - ka3 * 0.85)
        scene = scene + trail(i, phase) * np.array([1.0, 0.85, 0.45], np.float32)
        scene = scene + moon_glint(W, phase)[..., None] * moon_col * 0.38 * (1 - ka3)

        scene = scene * (1 - 0.35 * rshadow * inside)
        pads, pshadow = pads_layer(W, phase)
        scene = scene * (1 - pshadow[..., 3:4] * 0.9)
        for lay in (ripples_layer(W, phase), petals_layer(W, phase), pads):
            scene = scene * (1 - lay[..., 3:4]) + lay[..., :3] * lay[..., 3:4]

        sq = bg[..., :3] * (1 - inside) + scene * inside
        sq = sq * (1 - rocks[..., 3:4]) + rocks[..., :3] * rocks[..., 3:4]
        sq = sq + light_rays(W, phase)[..., None] * warm * 0.05

        out = outer.copy()
        region = out[OY:OY + W, OX:OX + W]
        f = feather[OY:OY + W, OX:OX + W]
        out[OY:OY + W, OX:OX + W] = sq * f + region * (1 - f)
        out = out * shade
        (gsh, gbody, glights), (tsh, tbody, tlights) = traffic.frame(i)
        out = out * (1 - 0.6 * gsh)
        ga = gbody[..., 3:4]
        out = out * (1 - ga) + gbody[..., :3] * ga
        out = out * (1 - 0.55 * deck_sh)
        ground_lights = city_lights(CW, CH, phase, windows, lamps, sxs, sys_) * (1 - parkmask) + glights
        beam, spot = signal_spotlights(CW, CH, phase)
        out = out + ground_lights * (1 - da) + spot * spot_col * (1 - parkmask * 0.6) * (1 - da * 0.5)
        out = out * (1 - da) + deck[..., :3] * da
        out = out * (1 - 0.6 * tsh)
        ta_ = tbody[..., 3:4]
        out = out * (1 - ta_) + tbody[..., :3] * ta_
        out = out + tlights + beam * spot_col * 0.9
        out = out + apt_lights
        plane_sh, plane_body, plane_lights = airport_planes(CW, CH, AX, phase)
        out = out * (1 - 0.35 * plane_sh)
        plane_a = plane_body[..., 3:4]
        out = out * (1 - plane_a) + plane_body[..., :3] * plane_a
        out = out + plane_lights
        out = out + fireflies_canvas(CW, CH, phase)[..., None] * gold
        out = (1 - np.exp(-out * 1.6)) / (1 - math.exp(-1.6))
        out = out * (1 - edge_fade) + page * edge_fade
        frames.append(Image.fromarray(np.clip(out * 255, 0, 255).astype(np.uint8)))

    frames[0].save(dst, save_all=True, append_images=frames[1:], duration=duration,
                   loop=0, quality=68, method=4)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
