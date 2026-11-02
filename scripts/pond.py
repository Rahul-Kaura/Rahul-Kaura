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


CANVAS_FLIES = [(rng.uniform(20, 900), rng.uniform(20, 540), rng.uniform(6, 22), rng.random(), rng.choice([1, 2])) for _ in range(34)]


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


# ---------- main ----------

def main(src, dst):
    koi = Image.open(src)
    n = koi.n_frames
    duration = koi.info.get("duration", 70)
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
    CW, CH = 920, 560
    OX, OY = (CW - W) // 2, (CH - W) // 2
    page = np.array([13, 17, 23], np.float32) / 255.0
    outer = garden(CW)[:CH, :, :3] * 0.8
    yy, xx = np.mgrid[0:CH, 0:CW].astype(np.float32)
    pd = np.hypot(xx - (OX + W / 2), yy - (OY + W / 2))
    feather = np.clip((256 - pd) / 34, 0, 1)[..., None]
    ex = (xx - CW / 2) / (CW / 2)
    ey = (yy - CH / 2) / (CH / 2)
    e = (ex ** 4 + ey ** 4) ** 0.25
    t = np.clip((e - 0.62) / (0.99 - 0.62), 0, 1)
    edge_fade = (t * t * (3 - 2 * t))[..., None]
    shade = np.clip(1 - 0.5 * np.clip((pd - 230) / 260, 0, 1), 0, 1)[..., None]

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
        floor = water + cst * caus_col * (0.10 + 0.12 * depth)

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
        out = out + fireflies_canvas(CW, CH, phase)[..., None] * gold
        out = (1 - np.exp(-out * 1.6)) / (1 - math.exp(-1.6))
        out = out * (1 - edge_fade) + page * edge_fade
        frames.append(Image.fromarray(np.clip(out * 255, 0, 255).astype(np.uint8)))

    frames[0].save(dst, save_all=True, append_images=frames[1:], duration=duration,
                   loop=0, quality=82, method=4)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
