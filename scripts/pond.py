"""Put the koi GIF in a pixel-art pond with looping ripples.

Usage: python pond.py koi/koi.gif
The pond is drawn at half resolution and scaled 2x with nearest-neighbour so it
matches git-koi's pixel size. Ripples and sparkles loop with the koi's frames.
"""

import math
import random
import sys

from PIL import Image, ImageDraw

KEY = (255, 0, 255)
S = 256  # pond canvas at koi pixel scale (GIF is 2x)
CX, CY, R = 128, 128, 122

WATER_DEEP = (14, 42, 58)
WATER_MID = (20, 62, 78)
WATER_EDGE = (34, 92, 102)
STONES = [(92, 96, 104), (112, 116, 122), (74, 78, 88), (132, 128, 120)]
PAD = (58, 128, 74)
PAD_DARK = (40, 98, 58)
PAD_LIGHT = (92, 160, 92)
FLOWER = (244, 170, 196)
FLOWER_CORE = (250, 214, 120)
RIPPLE = (92, 150, 160)
SPARKLE = (190, 228, 230)


def base_pond():
    img = Image.new("RGB", (S, S), KEY)
    px = img.load()
    for y in range(S):
        for x in range(S):
            d = math.hypot(x - CX, y - CY)
            if d <= R:
                t = d / R
                # stepped gradient, pixel-art style
                if t < 0.55:
                    c = WATER_DEEP
                elif t < 0.85:
                    c = WATER_MID
                else:
                    c = WATER_EDGE
                # light dithering between bands
                if 0.52 < t < 0.58 and (x + y) % 2 == 0:
                    c = WATER_MID
                if 0.82 < t < 0.88 and (x + y) % 2 == 0:
                    c = WATER_EDGE if c == WATER_MID else WATER_MID
                px[x, y] = c
    # stone rim
    rnd = random.Random(11)
    d = ImageDraw.Draw(img)
    a = 0.0
    while a < 2 * math.pi:
        rr = rnd.randint(4, 7)
        x = CX + (R + 2) * math.cos(a)
        y = CY + (R + 2) * math.sin(a)
        col = rnd.choice(STONES)
        d.ellipse([x - rr, y - rr * 0.8, x + rr, y + rr * 0.8], fill=col)
        d.point([(x - rr / 3, y - rr / 3)], fill=(160, 158, 152))
        a += (rr * 1.6) / (R + 2)
    return img


def lily_pads(img):
    d = ImageDraw.Draw(img)
    pads = [(52, 58, 13, 30), (206, 196, 15, 210), (198, 64, 10, 120), (64, 204, 9, 300)]
    for x, y, r, notch in pads:
        d.ellipse([x - r, y - r, x + r, y + r], fill=PAD_DARK)
        d.ellipse([x - r + 1, y - r + 1, x + r - 2, y + r - 2], fill=PAD)
        a = math.radians(notch)
        d.polygon(
            [(x, y), (x + (r + 1) * math.cos(a - 0.35), y + (r + 1) * math.sin(a - 0.35)),
             (x + (r + 1) * math.cos(a + 0.35), y + (r + 1) * math.sin(a + 0.35))],
            fill=WATER_EDGE,
        )
        d.line([(x - r / 2, y - r / 3), (x + r / 4, y - r / 2)], fill=PAD_LIGHT)
    # one flower
    fx, fy = 206, 196
    for ang in range(0, 360, 60):
        a = math.radians(ang)
        d.ellipse([fx + 3 * math.cos(a) - 2, fy + 3 * math.sin(a) - 2,
                   fx + 3 * math.cos(a) + 2, fy + 3 * math.sin(a) + 2], fill=FLOWER)
    d.rectangle([fx - 1, fy - 1, fx + 1, fy + 1], fill=FLOWER_CORE)


def water_fx(img, phase):
    """Ripples and sparkles for a phase in [0, 1)."""
    d = ImageDraw.Draw(img)
    for (x, y, offset) in [(92, 150, 0.0), (168, 104, 0.5)]:
        p = (phase + offset) % 1.0
        r = 3 + p * 22
        if p < 0.8:
            d.ellipse([x - r, y - r * 0.6, x + r, y + r * 0.6], outline=RIPPLE)
        r2 = r - 7
        if 0.25 < p < 0.95 and r2 > 2:
            d.ellipse([x - r2, y - r2 * 0.6, x + r2, y + r2 * 0.6], outline=WATER_EDGE)
    rnd = random.Random(5)
    for _ in range(18):
        a = rnd.uniform(0, 2 * math.pi)
        rr = rnd.uniform(10, R - 12)
        x, y = CX + rr * math.cos(a), CY + rr * math.sin(a)
        off = rnd.random()
        v = (math.sin(2 * math.pi * (phase + off)) + 1) / 2
        if v > 0.75:
            d.point([(x, y)], fill=SPARKLE)
            if v > 0.92:
                d.point([(x - 1, y), (x + 1, y)], fill=RIPPLE)


def main(path):
    koi = Image.open(path)
    n = koi.n_frames
    duration = koi.info.get("duration", 70)
    W, H = koi.size

    pond = base_pond()
    frames = []
    for i in range(n):
        koi.seek(i)
        k = koi.convert("RGBA")
        bg = pond.copy()
        water_fx(bg, i / n)
        bg = bg.resize((W, H), Image.NEAREST).convert("RGBA")
        bg.alpha_composite(k)
        # lily pads float above the koi
        top = Image.new("RGB", (S, S), KEY)
        lily_pads(top)
        top = top.resize((W, H), Image.NEAREST)
        tp = top.load()
        m = Image.new("L", (W, H), 0)
        mp = m.load()
        for y in range(H):
            for x in range(W):
                if tp[x, y] != KEY:
                    mp[x, y] = 255
        bg.paste(top, (0, 0), m)
        frames.append(bg.convert("RGB"))

    # one shared palette, KEY reserved for transparency
    sample = Image.new("RGB", (W, H * 4))
    for j, idx in enumerate(range(0, n, max(1, n // 4))):
        if j < 4:
            sample.paste(frames[idx], (0, H * j))
    pal = sample.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    p = pal.getpalette()[: 255 * 3] + list(KEY)
    pal.putpalette(p)
    key_index = 255

    out = []
    for f in frames:
        q = f.quantize(palette=pal, dither=Image.Dither.NONE)
        qp = q.load()
        fp = f.load()
        for y in range(H):
            for x in range(W):
                if fp[x, y] == KEY:
                    qp[x, y] = key_index
        out.append(q)

    out[0].save(path, save_all=True, append_images=out[1:], duration=duration,
                loop=0, transparency=key_index, disposal=2, optimize=False)


if __name__ == "__main__":
    main(sys.argv[1])
