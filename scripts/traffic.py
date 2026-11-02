"""Roundabout traffic for the night city around the pond.

A one-lane roundabout circles the park. Six roads leave it for the edges of the
city. Each car enters on one road, stops at its signal if the light is red,
circles counterclockwise and leaves by another road. Every car's schedule is
solved so its trip, including any wait, takes exactly one loop of the
animation, so the whole scene loops seamlessly.
"""

import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROAD_W = 26          # spoke road width
LANE = 5.5           # lane offset from the spoke's centerline
RING_W = 24          # roundabout width
STOP_GAP = 16        # stop line distance outside the ring's outer edge
QUEUE_GAP = 26       # spacing of cars queued at a light
SCALE = 3            # sprite supersampling


class Traffic:
    def __init__(self, CW, CH, cx, cy, ring_r, n_frames, seed=1939):
        self.CW, self.CH, self.cx, self.cy, self.R, self.n = CW, CH, cx, cy, ring_r, n_frames
        self.rng = random.Random(seed)
        corner = lambda x, y: math.atan2(y - cy, x - cx)
        # six exits: west, east and the four corners
        self.spokes = [math.pi, 0.0, corner(0, 0), corner(CW, 0), corner(0, CH), corner(CW, CH)]
        # two signal groups take turns: red for half the loop each
        self.group = [0, 1, 1, 0, 0, 1]
        self.cars = []
        self._build_cars()

    # ---------- geometry ----------

    def spoke_len(self, th):
        ux, uy = math.cos(th), math.sin(th)
        ts = []
        if ux > 1e-6:
            ts.append((self.CW + 60 - self.cx) / ux)
        if ux < -1e-6:
            ts.append((-60 - self.cx) / ux)
        if uy > 1e-6:
            ts.append((self.CH + 60 - self.cy) / uy)
        if uy < -1e-6:
            ts.append((-60 - self.cy) / uy)
        return min(ts)

    def corridor_dist(self, x, y):
        """Distance from a point to the nearest road centerline (ring or spoke)."""
        dx, dy = x - self.cx, y - self.cy
        d = math.hypot(dx, dy)
        best = abs(d - self.R)
        for th in self.spokes:
            ux, uy = math.cos(th), math.sin(th)
            t = dx * ux + dy * uy
            if t > self.R:
                best = min(best, abs(-dx * uy + dy * ux))
        return best

    def route(self, a, b):
        """Polyline for a trip entering on spoke a, leaving on spoke b."""
        ta, tb = self.spokes[a], self.spokes[b]
        pts = []
        # inbound lane, travelling toward the centre
        ux, uy = math.cos(ta), math.sin(ta)
        rx, ry = -(-uy), -ux  # right of the inbound direction (-u) in screen coords
        L = self.spoke_len(ta)
        for t in np.linspace(L, self.R + RING_W / 2 + 4, 24):
            pts.append((self.cx + ux * t + rx * LANE, self.cy + uy * t + ry * LANE))
        stop_s = None
        # around the ring, counterclockwise on screen (angle decreasing)
        arc = (ta - tb) % (2 * math.pi)
        if arc < 0.6:
            arc += 2 * math.pi
        start = ta - 0.12
        steps = max(8, int(arc * 40))
        for k in range(steps + 1):
            th = start - (arc - 0.24) * k / steps
            pts.append((self.cx + self.R * math.cos(th), self.cy + self.R * math.sin(th)))
        # outbound lane
        ux, uy = math.cos(tb), math.sin(tb)
        rx, ry = -uy, ux  # right of the outbound direction (u)
        L = self.spoke_len(tb)
        for t in np.linspace(self.R + RING_W / 2 + 4, L, 24):
            pts.append((self.cx + ux * t + rx * LANE, self.cy + uy * t + ry * LANE))
        pts = np.array(pts, np.float32)
        seg = np.hypot(*(pts[1:] - pts[:-1]).T)
        cum = np.concatenate([[0], np.cumsum(seg)])
        # stop line: on the inbound lane, STOP_GAP outside the ring
        stop_r = self.R + RING_W / 2 + STOP_GAP
        stop_s = self.spoke_len(ta) - stop_r
        return pts, cum, stop_s

    def light_red(self, spoke, frame):
        half = self.n // 2
        f = frame % self.n
        red_first = self.group[spoke] == 0
        return (f < half) == red_first

    # ---------- schedule ----------

    def _schedule(self, total, stop_s, spoke, offset):
        """Speed, time to reach the stop line, and wait, so the trip takes one loop."""
        n = self.n
        wait = 0.0
        for _ in range(8):
            v = total / (n - wait)
            ta = stop_s / v
            arrive = (offset + ta) % n
            if self.light_red(spoke, int(math.floor(arrive))):
                half = n // 2
                new_wait = (half if arrive < half else n) - arrive
            else:
                new_wait = 0.0
            if abs(new_wait - wait) < 0.5:
                wait = new_wait
                break
            wait = new_wait
        v = total / (n - wait)
        return v, stop_s / v, wait

    def _build_cars(self):
        r = self.rng
        kinds = (["sedan"] * 5 + ["taxi"] * 3 + ["suv"] * 3 + ["sports"] * 2 + ["truck"] * 2 + ["police"] * 2)
        r.shuffle(kinds)
        queue = {i: 0 for i in range(len(self.spokes))}
        order = list(range(len(kinds)))
        for idx in order:
            a = idx % len(self.spokes)
            b = (a + r.randint(1, len(self.spokes) - 1)) % len(self.spokes)
            pts, cum, stop_s = self.route(a, b)
            slot = queue[a]
            queue[a] += 1
            stop_s_q = stop_s - (slot % 3) * QUEUE_GAP
            offset = r.uniform(0, self.n) if slot == 0 else (r.uniform(0, self.n))
            v, ta, wait = self._schedule(cum[-1], stop_s_q, a, offset)
            kind = kinds[idx]
            self.cars.append(dict(kind=kind, pts=pts, cum=cum, v=v, ta=ta, wait=wait, offset=offset,
                                  stop=stop_s_q, sprite=car_sprite(kind, r), seed=r.random()))

    # ---------- per-frame ----------

    def state(self, car, frame):
        tau = (frame - car["offset"]) % self.n
        if tau < car["ta"]:
            s, braking = car["v"] * tau, (car["ta"] - tau) < 4 and car["wait"] > 0
        elif tau < car["ta"] + car["wait"]:
            s, braking = car["stop"], True
        else:
            s, braking = car["stop"] + car["v"] * (tau - car["ta"] - car["wait"]), False
        cum, pts = car["cum"], car["pts"]
        s = min(max(s, 0), cum[-1] - 1e-3)
        i = int(np.searchsorted(cum, s) - 1)
        i = max(0, min(i, len(pts) - 2))
        t = (s - cum[i]) / max(cum[i + 1] - cum[i], 1e-6)
        x, y = pts[i] * (1 - t) + pts[i + 1] * t
        hx, hy = pts[i + 1] - pts[i]
        h = math.atan2(hy, hx)
        return float(x), float(y), h, braking

    def draw_roads(self, img, S):
        """Draw the roundabout and spokes onto an RGB image supersampled by S."""
        d = ImageDraw.Draw(img)
        cx, cy, R = self.cx * S, self.cy * S, self.R * S
        asphalt = (17, 18, 23)
        for th in self.spokes:
            L = self.spoke_len(th)
            ux, uy = math.cos(th), math.sin(th)
            px, py = -uy, ux
            hw = ROAD_W / 2
            a0 = (self.cx + ux * self.R, self.cy + uy * self.R)
            a1 = (self.cx + ux * L, self.cy + uy * L)
            poly = [(a0[0] + px * hw, a0[1] + py * hw), (a1[0] + px * hw, a1[1] + py * hw),
                    (a1[0] - px * hw, a1[1] - py * hw), (a0[0] - px * hw, a0[1] - py * hw)]
            d.polygon([(x * S, y * S) for x, y in poly], fill=asphalt)
            for side in (hw, -hw):
                d.line([((a0[0] + px * side) * S, (a0[1] + py * side) * S), ((a1[0] + px * side) * S, (a1[1] + py * side) * S)],
                       fill=(46, 47, 54), width=S)
            # gold double centerline
            for off in (-0.9, 0.9):
                d.line([((a0[0] + px * off) * S, (a0[1] + py * off) * S), ((a1[0] + px * off) * S, (a1[1] + py * off) * S)],
                       fill=(150, 112, 30), width=max(1, S // 2))
            # crosswalk and stop line on the inbound side
            cw0 = self.R + RING_W / 2 + 4
            for k in range(-5, 6):
                t0, t1 = cw0, cw0 + 8
                o = k * 2.4
                d.line([((self.cx + ux * t0 + px * o) * S, (self.cy + uy * t0 + py * o) * S),
                        ((self.cx + ux * t1 + px * o) * S, (self.cy + uy * t1 + py * o) * S)], fill=(70, 72, 80), width=S)
            sr = self.R + RING_W / 2 + STOP_GAP - 6
            rx, ry = uy, -ux
            d.line([((self.cx + ux * sr) * S, (self.cy + uy * sr) * S),
                    ((self.cx + ux * sr + rx * hw) * S, (self.cy + uy * sr + ry * hw) * S)], fill=(150, 150, 155), width=S)
        # ring
        ro, ri = (self.R + RING_W / 2) * S, (self.R - RING_W / 2) * S
        d.ellipse([cx - ro, cy - ro, cx + ro, cy + ro], fill=asphalt)
        d.ellipse([cx - ro, cy - ro, cx + ro, cy + ro], outline=(46, 47, 54), width=S)
        d.ellipse([cx - ri, cy - ri, cx + ri, cy + ri], outline=(60, 62, 70), width=S)
        # dashed gold centre ring
        for k in range(0, 360, 4):
            a0, a1 = math.radians(k), math.radians(k + 2)
            d.line([(cx + R * math.cos(a0), cy + R * math.sin(a0)), (cx + R * math.cos(a1), cy + R * math.sin(a1))],
                   fill=(110, 84, 26), width=max(1, S // 2))

    def signals(self):
        out = []
        for i, th in enumerate(self.spokes):
            ux, uy = math.cos(th), math.sin(th)
            rx, ry = uy, -ux  # right of the inbound direction
            sr = self.R + RING_W / 2 + STOP_GAP - 2
            out.append((i, self.cx + ux * sr + rx * (ROAD_W / 2 + 5), self.cy + uy * sr + ry * (ROAD_W / 2 + 5)))
        return out

    def frame(self, frame):
        """Car bodies (RGBA float array) and their lights (RGB float array, additive)."""
        CW, CH = self.CW, self.CH
        body = Image.new("RGBA", (CW, CH), (0, 0, 0, 0))
        shadow = Image.new("L", (CW, CH), 0)
        glow = Image.new("RGB", (CW, CH), (0, 0, 0))
        core = Image.new("RGB", (CW, CH), (0, 0, 0))
        dg, dc, dsh = ImageDraw.Draw(glow), ImageDraw.Draw(core), ImageDraw.Draw(shadow)

        # traffic signals
        for (i, x, y) in self.signals():
            red = self.light_red(i, frame)
            dsh.rectangle([x - 2.5, y - 4.5, x + 3.5, y + 5.5], fill=150)
            col = (255, 40, 30) if red else (40, 255, 120)
            gl = (150, 20, 12) if red else (20, 150, 70)
            ly = y - 2 if red else y + 2
            dg.ellipse([x - 9, ly - 9, x + 9, ly + 9], fill=gl)
            dc.ellipse([x - 1.6, ly - 1.6, x + 1.6, ly + 1.6], fill=col)
            dc.ellipse([x - 1.2, (y + 2 if red else y - 2) - 1.2, x + 1.2, (y + 2 if red else y - 2) + 1.2], fill=(30, 30, 30))

        for car in self.cars:
            x, y, h, braking = self.state(car, frame)
            if x < -40 or x > CW + 40 or y < -40 or y > CH + 40:
                continue
            spr = car["sprite"]
            rot = spr.rotate(-math.degrees(h), resample=Image.BICUBIC, expand=True)
            small = rot.resize((max(1, rot.width // SCALE), max(1, rot.height // SCALE)), Image.LANCZOS)
            px, py = int(round(x - small.width / 2)), int(round(y - small.height / 2))
            sh = Image.new("L", small.size, 0)
            sh.paste(small.getchannel("A").point(lambda v: int(v * 0.7)))
            shadow.paste(sh, (px + 2, py + 3), sh)
            _paste_clip(body, small, px, py)

            ux, uy = math.cos(h), math.sin(h)
            pxv, pyv = -uy, ux
            L2 = spr.info_len / 2
            hx, hy = x + ux * L2, y + uy * L2
            bl, bw = 62, 15
            dg.polygon([(hx + pxv * 3, hy + pyv * 3), (hx - pxv * 3, hy - pyv * 3),
                        (hx + ux * bl - pxv * bw, hy + uy * bl - pyv * bw), (hx + ux * bl + pxv * bw, hy + uy * bl + pyv * bw)],
                       fill=(62, 54, 36))
            for side in (-spr.info_w / 2 + 1.2, spr.info_w / 2 - 1.2):
                dc.ellipse([hx + pxv * side - 1.4, hy + pyv * side - 1.4, hx + pxv * side + 1.4, hy + pyv * side + 1.4],
                           fill=(255, 250, 228))
                tx, ty = x - ux * L2 + pxv * side, y - uy * L2 + pyv * side
                rr = 1.6 if braking else 1.2
                dc.ellipse([tx - rr, ty - rr, tx + rr, ty + rr], fill=(255, 40, 28))
                g = 10 if braking else 6
                dg.ellipse([tx - g, ty - g, tx + g, ty + g], fill=(230, 20, 10) if braking else (130, 12, 6))
            if car["kind"] == "police":
                on = (frame // 2) % 2 == 0
                for side, col in ((-2.2, (255, 30, 30)), (2.2, (40, 90, 255))):
                    lit = on if side < 0 else not on
                    if not lit:
                        continue
                    bx, by = x + pxv * side, y + pyv * side
                    dc.ellipse([bx - 1.6, by - 1.6, bx + 1.6, by + 1.6], fill=col)
                    dg.ellipse([bx - 16, by - 16, bx + 16, by + 16], fill=tuple(int(c * 0.8) for c in col))

        sh = np.asarray(shadow.filter(ImageFilter.GaussianBlur(1.5)), np.float32)[..., None] / 255.0
        b = np.asarray(body, np.float32) / 255.0
        lights = (np.asarray(glow.filter(ImageFilter.GaussianBlur(6)), np.float32) / 255.0 * 0.9 +
                  np.asarray(core.filter(ImageFilter.GaussianBlur(0.5)), np.float32) / 255.0)
        return sh, b, lights


def _paste_clip(dst, src, px, py):
    x0, y0 = max(0, -px), max(0, -py)
    if x0 >= src.width or y0 >= src.height:
        return
    crop = src.crop((x0, y0, src.width, src.height))
    if crop.width > 0 and crop.height > 0:
        dst.alpha_composite(crop, (max(0, px), max(0, py)))


SEDAN_COLORS = [(22, 22, 26), (92, 18, 22), (150, 152, 160), (24, 34, 62), (58, 60, 66), (120, 98, 60)]


def car_sprite(kind, r):
    """Top-down car facing +x, drawn at SCALE x size."""
    dims = {"sedan": (18, 8), "taxi": (18, 8), "suv": (20, 9), "sports": (17, 7.4), "truck": (28, 10), "police": (19, 8.2)}
    L, Wd = dims[kind]
    S = SCALE
    pad = 4
    img = Image.new("RGBA", (int((L + pad * 2) * S), int((Wd + pad * 2) * S)), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    ox, oy = pad * S, pad * S

    def R(x0, y0, x1, y1, fill, rad=2):
        d.rounded_rectangle([ox + x0 * S, oy + y0 * S, ox + x1 * S, oy + y1 * S], radius=rad * S, fill=fill)

    if kind == "taxi":
        paint = (222, 168, 34)
    elif kind == "police":
        paint = (16, 16, 20)
    elif kind == "suv":
        paint = r.choice([(14, 14, 18), (30, 32, 38), (52, 20, 24)])
    elif kind == "sports":
        paint = r.choice([(200, 150, 30), (160, 20, 24), (230, 230, 236)])
    elif kind == "truck":
        paint = (178, 180, 186)
    else:
        paint = r.choice(SEDAN_COLORS)
    hi = tuple(min(255, int(c * 1.35 + 18)) for c in paint)
    glass = (34, 44, 60)

    if kind == "truck":
        R(0, 0, L * 0.72, Wd, paint, 1)                       # cargo box
        R(0.6, 0.6, L * 0.72 - 0.6, 1.6, hi, 1)
        R(L * 0.74, 0.5, L, Wd - 0.5, (60, 64, 74), 2)         # cab
        R(L * 0.86, 1.2, L * 0.96, Wd - 1.2, glass, 1)
    else:
        R(0, 0, L, Wd, paint, 3 if kind != "sports" else 3.5)
        # roof / cabin
        c0, c1 = (L * 0.3, L * 0.72) if kind != "sports" else (L * 0.36, L * 0.68)
        if kind == "suv":
            c0, c1 = L * 0.22, L * 0.8
        R(c0, 1.1, c1, Wd - 1.1, glass, 1.5)
        R(c0 + L * 0.06, 1.6, c1 - L * 0.08, Wd - 1.6, hi if kind != "police" else (230, 230, 236), 1)
        # hood highlight
        R(c1 + 0.8, 1.6, L - 1.4, 2.4, hi, 1)
        if kind == "police":
            R(0, 0, L * 0.26, Wd, (230, 230, 236), 2)          # white doors front/back
            R(L * 0.76, 0, L, Wd, (230, 230, 236), 2)
            R(L * 0.46, 0.8, L * 0.54, Wd - 0.8, (40, 40, 48), 0.5)  # light bar
        if kind == "taxi":
            R(L * 0.45, Wd * 0.3, L * 0.56, Wd * 0.7, (250, 236, 170), 0.6)  # roof sign
            for k in range(3):
                R(L * 0.08 + k * 2.2, Wd - 1.1, L * 0.08 + k * 2.2 + 1.1, Wd - 0.2, (20, 20, 20), 0.2)
        if kind == "sports":
            R(L * 0.05, Wd * 0.44, L * 0.95, Wd * 0.56, (20, 20, 22), 0.3)  # racing stripe
    img.info_len = L
    img.info_w = Wd
    return img
