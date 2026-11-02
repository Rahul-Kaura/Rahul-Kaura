"""Night traffic around the pond: ring road, side streets, elevated highways.

Layout (canvas CW x CH, pond centred):
  - A two-lane ring road (rounded rectangle) circles the park, counterclockwise
    on screen. The inner lane carries a steady row of cars; the outer lane is
    used by cars joining and leaving.
  - Two horizontal side streets on each side connect the ring to the city edge.
    Each has a signal at its entry to the ring.
  - An elevated highway runs down each end of the canvas. The side streets pass
    under it. Cloverleaf loop ramps bring cars off the highway and under the
    deck; on-ramps merge cars back up onto it.

Cars on the routes are simulated with simple car following and red lights
until the traffic settles into a repeating pattern, then one period is
recorded, so the animation loops seamlessly.
"""

import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SCALE = 3          # car sprite supersampling
DECK_W = 64        # highway deck width
RING_W = 34        # ring road width (two lanes)
ST_W = 26          # side street width
LANE = 6.5         # lane offset from a two-way street's centreline

# speed limits in px per frame: slow, calm traffic
V_GROUND, V_RING, V_RAMP, V_HWY = 0.67, 0.6, 0.55, 0.84


def bezier(p0, p1, p2, p3, k=28):
    out = []
    for i in range(k + 1):
        t = i / k
        a, b, c, d = (1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t * t, t ** 3
        out.append((a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0], a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1]))
    return out


def arc(cx, cy, r, a0, a1, k=40):
    return [(cx + r * math.cos(a0 + (a1 - a0) * i / k), cy + r * math.sin(a0 + (a1 - a0) * i / k)) for i in range(k + 1)]


class Path:
    """Polyline with per-point kind ('ground', 'ramp', 'hwy') and speed limit."""

    def __init__(self):
        self.pts, self.kind, self.vmax = [], [], []

    def add(self, pts, kind, vmax):
        for p in pts:
            if self.pts and math.hypot(p[0] - self.pts[-1][0], p[1] - self.pts[-1][1]) < 0.5:
                continue
            self.pts.append((float(p[0]), float(p[1])))
            self.kind.append(kind)
            self.vmax.append(vmax)
        return self

    def finish(self):
        self.P = np.array(self.pts, np.float32)
        seg = np.hypot(*(self.P[1:] - self.P[:-1]).T)
        self.cum = np.concatenate([[0], np.cumsum(seg)])
        self.length = float(self.cum[-1])
        return self

    def at(self, s):
        s = min(max(s, 0.0), self.length - 1e-3)
        i = int(np.searchsorted(self.cum, s) - 1)
        i = max(0, min(i, len(self.P) - 2))
        t = (s - self.cum[i]) / max(self.cum[i + 1] - self.cum[i], 1e-6)
        x, y = self.P[i] * (1 - t) + self.P[i + 1] * t
        dx, dy = self.P[i + 1] - self.P[i]
        return float(x), float(y), math.atan2(dy, dx), self.kind[i], self.vmax[i]

    def nearest_s(self, x, y):
        P = np.array(self.pts, np.float32)
        cum = np.concatenate([[0], np.cumsum(np.hypot(*(P[1:] - P[:-1]).T))])
        d = np.hypot(P[:, 0] - x, P[:, 1] - y)
        return float(cum[int(np.argmin(d))])


class Traffic:
    def __init__(self, CW, CH, cx, cy, n_frames, seed=1939, ring_only=False):
        self.ring_only = ring_only
        self.CW, self.CH, self.cx, self.cy, self.n = CW, CH, cx, cy, n_frames
        self.rng = random.Random(seed)
        self.ra, self.rb, self.rc = 345, 318, 150     # ring centreline half sizes, corner radius
        self.HL, self.HR = 96, CW - 96                 # highway centrelines
        self.y_up, self.y_lo = cy - 92, cy + 92        # side streets
        self._ring_cache = {}
        self._build_routes()
        self._simulate()
        self.sprites = {}

    # ---------- ring geometry ----------

    def ring_lane(self, off):
        """Closed polyline of the ring offset outward by off, counterclockwise on screen."""
        key = round(off, 2)
        if key in self._ring_cache:
            return self._ring_cache[key]
        a, b, rc = self.ra + off, self.rb + off, self.rc + off
        cx, cy = self.cx, self.cy
        pts = []
        pts += [(cx + a, y) for y in np.linspace(cy + b - rc, cy - b + rc, 40)]
        pts += arc(cx + a - rc, cy - b + rc, rc, 0, -math.pi / 2)
        pts += [(x, cy - b) for x in np.linspace(cx + a - rc, cx - a + rc, 50)]
        pts += arc(cx - a + rc, cy - b + rc, rc, -math.pi / 2, -math.pi)
        pts += [(cx - a, y) for y in np.linspace(cy - b + rc, cy + b - rc, 40)]
        pts += arc(cx - a + rc, cy + b - rc, rc, math.pi, math.pi / 2)
        pts += [(x, cy + b) for x in np.linspace(cx - a + rc, cx + a - rc, 50)]
        pts += arc(cx + a - rc, cy + b - rc, rc, math.pi / 2, 0)
        self._ring_cache[key] = pts
        return pts

    def ring_segment(self, off, p_from, p_to):
        pts = self.ring_lane(off)
        P = np.array(pts)
        i0 = int(np.argmin(np.hypot(P[:, 0] - p_from[0], P[:, 1] - p_from[1])))
        i1 = int(np.argmin(np.hypot(P[:, 0] - p_to[0], P[:, 1] - p_to[1])))
        out, i = [], i0
        while True:
            out.append(pts[i])
            if i == i1:
                break
            i = (i + 1) % len(pts)
        return out

    def ring_sdf_grid(self, X, Y, off=0.0):
        a, b, rc = self.ra + off, self.rb + off, self.rc + off
        qx, qy = np.abs(X - self.cx) - a + rc, np.abs(Y - self.cy) - b + rc
        return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - rc

    # ---------- routes ----------

    def _build_routes(self):
        if self.ring_only:
            self.routes = []
            self.flows = []
            for off, gap in ((RING_W / 4, 86), (-RING_W / 4, 104)):
                ring = Path()
                rp = self.ring_lane(off)
                ring.add(rp + [rp[0]], "ground", V_RING)
                self.flows.append((ring.finish(), gap, 1))
            return
        CW, CH, cx = self.CW, self.CH, self.cx
        HL, HR, yu, yl = self.HL, self.HR, self.y_up, self.y_lo
        out_lane, in_lane = RING_W / 4, -RING_W / 4
        xl_ring = cx - self.ra - RING_W / 2
        xr_ring = cx + self.ra + RING_W / 2
        G, R, RP, H = V_GROUND, V_RING, V_RAMP, V_HWY
        hd = DECK_W / 2

        # A: right edge -> under right highway -> upper right street -> top of ring -> upper left street -> under left highway
        A = Path()
        A.add([(CW + 50, yu - LANE), (xr_ring + 2, yu - LANE)], "ground", G)
        A.add(self.ring_segment(out_lane, (xr_ring, yu - LANE), (xl_ring, yu - LANE)), "ground", R)
        A.add([(xl_ring - 2, yu - LANE), (-50, yu - LANE)], "ground", G)
        A.stop, A.light = A.nearest_s(xr_ring + 16, yu - LANE), 0

        # B: left highway southbound -> cloverleaf loop -> under the deck -> upper left street ->
        #    left side of ring -> lower left street -> on-ramp -> left highway northbound
        B = Path()
        sb = HL - 22
        lc = (HL - hd - 30, yu - 30)
        B.add([(sb, -50), (sb, yu - 150)], "hwy", H)
        B.add(bezier((sb, yu - 150), (sb, yu - 110), (lc[0] + 30, yu - 95), (lc[0] + 30, yu - 60)), "ramp", RP)
        B.add(arc(lc[0], lc[1], 30, 0, math.pi, 40), "ramp", RP)
        B.add(bezier((lc[0] - 30, lc[1]), (lc[0] - 30, yu + LANE), (lc[0] - 10, yu + LANE), (lc[0] + 15, yu + LANE)), "ramp", RP)
        B.add([(lc[0] + 15, yu + LANE), (xl_ring - 2, yu + LANE)], "ground", G)
        B.add(self.ring_segment(out_lane, (xl_ring, yu + LANE), (xl_ring, yl - LANE)), "ground", R)
        nb = HL + 22
        B.add([(xl_ring - 2, yl - LANE), (HL + hd + 60, yl - LANE)], "ground", G)
        B.add(bezier((HL + hd + 60, yl - LANE), (HL + hd + 20, yl - LANE), (nb, yl - 30), (nb, yl - 80)), "ramp", RP)
        B.add([(nb, yl - 80), (nb, -50)], "hwy", H)
        B.stop, B.light = B.nearest_s(xl_ring - 16, yu + LANE), 1

        # C: left edge -> under left highway -> lower left street -> bottom of ring -> lower right street -> under right highway
        C = Path()
        C.add([(-50, yl + LANE), (xl_ring - 2, yl + LANE)], "ground", G)
        C.add(self.ring_segment(out_lane, (xl_ring, yl + LANE), (xr_ring, yl + LANE)), "ground", R)
        C.add([(xr_ring + 2, yl + LANE), (CW + 50, yl + LANE)], "ground", G)
        C.stop, C.light = C.nearest_s(xl_ring - 16, yl + LANE), 0

        # D: right highway northbound -> cloverleaf loop -> under the deck -> lower right street ->
        #    right side of ring -> upper right street -> on-ramp -> right highway southbound
        D = Path()
        nbr = HR + 22
        lc = (HR + hd + 30, yl + 30)
        D.add([(nbr, CH + 50), (nbr, yl + 150)], "hwy", H)
        D.add(bezier((nbr, yl + 150), (nbr, yl + 110), (lc[0] - 30, yl + 95), (lc[0] - 30, yl + 60)), "ramp", RP)
        D.add(arc(lc[0], lc[1], 30, math.pi, 2 * math.pi, 40), "ramp", RP)
        D.add(bezier((lc[0] + 30, lc[1]), (lc[0] + 30, yl - LANE), (lc[0] + 10, yl - LANE), (lc[0] - 15, yl - LANE)), "ramp", RP)
        D.add([(lc[0] - 15, yl - LANE), (xr_ring + 2, yl - LANE)], "ground", G)
        D.add(self.ring_segment(out_lane, (xr_ring, yl - LANE), (xr_ring, yu + LANE)), "ground", R)
        sbr = HR - 22
        D.add([(xr_ring + 2, yu + LANE), (HR - hd - 60, yu + LANE)], "ground", G)
        D.add(bezier((HR - hd - 60, yu + LANE), (HR - hd - 20, yu + LANE), (sbr, yu + 30), (sbr, yu + 80)), "ramp", RP)
        D.add([(sbr, yu + 80), (sbr, CH + 50)], "hwy", H)
        D.stop, D.light = D.nearest_s(xr_ring + 16, yl - LANE), 1

        self.routes = []
        for p in (A, B, C, D):
            stop, light = p.stop, p.light
            p.finish()
            p.stop, p.light = stop, light
            self.routes.append(p)

        # steady rows of cars: highway inner lanes and the ring's inner lane
        self.flows = []
        for x, down in ((HL - 9, True), (HL + 9, False), (HR - 9, True), (HR + 9, False)):
            f = Path()
            f.add([(x, -60), (x, CH + 60)] if down else [(x, CH + 60), (x, -60)], "hwy", H)
            self.flows.append((f.finish(), 60 + self.rng.uniform(0, 12), 1))

    # ---------- signals ----------

    def light_red(self, group, frame):
        half = self.n // 2
        return ((frame % self.n) < half) == (group == 0)

    # ---------- simulation ----------

    def _simulate(self):
        n = self.n
        kinds = ["sedan", "sedan", "supercar", "taxi", "taxi", "suv", "suv", "sports", "truck", "sedan"]
        spawn_every = n
        warm = 40 * n
        self.route_frames = []
        for ri, path in enumerate(self.routes):
            cars, frames = [], []
            for f in range(warm + n):
                if f % spawn_every == 0:
                    k = kinds[(ri * 3 + (f // spawn_every)) % len(kinds)]
                    if not cars or cars[-1]["s"] > 30:
                        cars.append(dict(s=0.0, v=0.4, kind=k, brake=False, id=(ri, f % n, (f // n) % 3)))
                red = self.light_red(path.light, f)
                for i, c in enumerate(cars):
                    vmax = path.at(c["s"])[4]
                    gap = 1e9
                    if i > 0:
                        gap = cars[i - 1]["s"] - c["s"] - 24
                    if red and c["s"] < path.stop - 1:
                        gap = min(gap, path.stop - c["s"] - 4)
                    safe = max(0.0, gap * 0.05)
                    nv = min(c["v"] + 0.012, vmax, safe)
                    c["brake"] = nv < c["v"] - 0.005 or (nv < 0.1 and gap < 40)
                    c["v"] = max(0.0, nv)
                for c in cars:
                    c["s"] += c["v"]
                cars = [c for c in cars if c["s"] < path.length - 1]
                if f >= warm:
                    frames.append([(c["s"], c["kind"], c["brake"], (c["kind"],) + c["id"][:2]) for c in cars])
            self.route_frames.append(frames)

    def sprite(self, key, kind):
        if key not in self.sprites:
            self.sprites[key] = car_sprite(kind, random.Random(hash(key) & 0xFFFFFFFF))
        return self.sprites[key]

    def cars_at(self, frame):
        out = []
        for ri, path in enumerate(self.routes):
            for (s, kind, brake, key) in self.route_frames[ri][frame % self.n]:
                x, y, h, k, _ = path.at(s)
                out.append((x, y, h, self.elevated(x, k), kind, brake, ("r",) + key))
        flow_kinds = ["sedan", "suv", "supercar", "taxi", "truck", "sedan", "sports", "suv", "sedan", "taxi", "sedan", "suv"]
        for fi, (path, gap, m) in enumerate(self.flows):
            count = max(1, int(path.length // gap))
            spacing = path.length / count
            shift = spacing * m * (frame % self.n) / self.n
            for j in range(count):
                s = (j * spacing + shift) % path.length
                x, y, h, k, _ = path.at(s)
                kind = flow_kinds[(j * 7 + fi * 3) % len(flow_kinds)]
                out.append((x, y, h, self.elevated(x, k), kind, False, ("f", fi, j)))
        return out

    def elevated(self, x, kind):
        if self.ring_only:
            return False
        if kind == "ground":
            return False
        return abs(x - self.HL) <= DECK_W / 2 + 2 or abs(x - self.HR) <= DECK_W / 2 + 2

    # ---------- static drawing ----------

    def road_mask(self, CW, CH, pad=10):
        m = Image.new("L", (CW, CH), 0)
        d = ImageDraw.Draw(m)
        if not self.ring_only:
            for y in (self.y_up, self.y_lo):
                d.rectangle([0, y - ST_W / 2 - pad, CW, y + ST_W / 2 + pad], fill=255)
            for x in (self.HL, self.HR):
                d.rectangle([x - DECK_W / 2 - pad, 0, x + DECK_W / 2 + pad, CH], fill=255)
        for path in self.routes:
            pts = [(float(px), float(py)) for (px, py), k in zip(path.P, path.kind) if k == "ramp"]
            if len(pts) > 1:
                d.line(pts, fill=255, width=int(14 + pad * 2))
        X, Y = np.meshgrid(np.arange(CW, dtype=np.float32), np.arange(CH, dtype=np.float32))
        sd = self.ring_sdf_grid(X, Y)
        return (np.asarray(m) > 0) | (sd < RING_W / 2 + pad)

    def draw_ground(self, img, S):
        d = ImageDraw.Draw(img)
        asphalt, edge, gold = (17, 18, 23), (48, 49, 56), (150, 112, 30)
        CW, CH = self.CW, self.CH
        for y in (() if self.ring_only else (self.y_up, self.y_lo)):
            d.rectangle([0, (y - ST_W / 2) * S, CW * S, (y + ST_W / 2) * S], fill=asphalt)
            for yy in (y - ST_W / 2, y + ST_W / 2):
                d.line([(0, yy * S), (CW * S, yy * S)], fill=edge, width=S)
            for off in (-0.9, 0.9):
                d.line([(0, (y + off) * S), (CW * S, (y + off) * S)], fill=gold, width=max(1, S // 2))
        for path in self.routes:
            pts = [(float(px) * S, float(py) * S) for (px, py), k in zip(path.P, path.kind) if k == "ramp"]
            if len(pts) > 1:
                d.line(pts, fill=edge, width=16 * S, joint="curve")
                d.line(pts, fill=asphalt, width=13 * S, joint="curve")
        X = np.arange(CW * S, dtype=np.float32)[None, :] / S
        Y = np.arange(CH * S, dtype=np.float32)[:, None] / S
        sd = self.ring_sdf_grid(X, Y)
        arr = np.asarray(img).copy()
        arr[np.abs(sd) < RING_W / 2] = asphalt
        arr[np.abs(np.abs(sd) - RING_W / 2) < 0.7] = edge
        img.paste(Image.fromarray(arr))
        d = ImageDraw.Draw(img)
        pts = self.ring_lane(0)
        for i in range(0, len(pts) - 1, 2):
            d.line([(pts[i][0] * S, pts[i][1] * S), (pts[i + 1][0] * S, pts[i + 1][1] * S)], fill=(120, 122, 130), width=max(1, S // 2))
        for path in self.routes:
            x, y, h, _, _ = path.at(path.stop)
            ux, uy = math.cos(h), math.sin(h)
            px, py = -uy, ux
            d.line([((x + ux * 3 - px * 1) * S, (y + uy * 3 - py * 1) * S),
                    ((x + ux * 3 + px * LANE * 1.9) * S, (y + uy * 3 + py * LANE * 1.9) * S)], fill=(160, 160, 165), width=S)
            for k in range(-5, 6):
                o = k * 2.4
                d.line([((x + ux * 6 + px * o) * S, (y + uy * 6 + py * o) * S),
                        ((x + ux * 13 + px * o) * S, (y + uy * 13 + py * o) * S)], fill=(64, 66, 74), width=S)

    def deck_layer(self, CW, CH):
        if self.ring_only:
            return np.zeros((CH, CW, 4), np.float32), np.zeros((CH, CW, 1), np.float32)
        S = 2
        lay = Image.new("RGBA", (CW * S, CH * S), (0, 0, 0, 0))
        sh = Image.new("L", (CW * S, CH * S), 0)
        d, ds = ImageDraw.Draw(lay), ImageDraw.Draw(sh)
        for x in (self.HL, self.HR):
            x0, x1 = (x - DECK_W / 2) * S, (x + DECK_W / 2) * S
            ds.rectangle([x0 + 10 * S, 0, x1 + 10 * S, CH * S], fill=170)
            d.rectangle([x0, 0, x1, CH * S], fill=(27, 28, 34, 255))
            d.rectangle([x0, 0, x0 + 2 * S, CH * S], fill=(84, 86, 96, 255))
            d.rectangle([x1 - 2 * S, 0, x1, CH * S], fill=(84, 86, 96, 255))
            d.rectangle([x * S - 1.5 * S, 0, x * S + 1.5 * S, CH * S], fill=(66, 68, 76, 255))
            for xo in (-15.5, 15.5):
                for yy in range(0, CH, 12):
                    d.line([((x + xo) * S, yy * S), ((x + xo) * S, (yy + 6) * S)], fill=(110, 112, 120, 255), width=S)
            for xo in (-2.6, 2.6):
                d.line([((x + xo) * S, 0), ((x + xo) * S, CH * S)], fill=(150, 112, 30, 255), width=S)
            for yy in range(0, CH, 60):
                d.line([(x0, yy * S), (x1, yy * S)], fill=(20, 21, 26, 255), width=S)
        for path in self.routes:
            pts = [(float(px) * S, float(py) * S) for (px, py), k in zip(path.P, path.kind)
                   if k == "ramp" and self.elevated(float(px), k)]
            if len(pts) > 1:
                d.line(pts, fill=(27, 28, 34, 255), width=13 * S, joint="curve")
        sh = sh.filter(ImageFilter.GaussianBlur(8 * S)).resize((CW, CH), Image.LANCZOS)
        lay = lay.resize((CW, CH), Image.LANCZOS)
        return np.asarray(lay, np.float32) / 255.0, np.asarray(sh, np.float32)[..., None] / 255.0

    def deck_lamps(self):
        if self.ring_only:
            return []
        return [(x, y) for x in (self.HL, self.HR) for y in range(20, self.CH, 72)]

    # ---------- per frame ----------

    def frame(self, frame):
        """((shadow, body, lights) for ground level, (shadow, body, lights) for the deck level)."""
        CW, CH = self.CW, self.CH
        layers = {lvl: dict(body=Image.new("RGBA", (CW, CH), (0, 0, 0, 0)), shadow=Image.new("L", (CW, CH), 0),
                            glow=Image.new("RGB", (CW, CH), (0, 0, 0)), core=Image.new("RGB", (CW, CH), (0, 0, 0)))
                  for lvl in ("g", "t")}
        g = layers["g"]
        dg, dc = ImageDraw.Draw(g["glow"]), ImageDraw.Draw(g["core"])
        for path in self.routes:
            x, y, h, _, _ = path.at(path.stop)
            ux, uy = math.cos(h), math.sin(h)
            px, py = -uy, ux
            sx, sy = x + ux * 2 + px * (ST_W / 2 + 6), y + uy * 2 + py * (ST_W / 2 + 6)
            red = self.light_red(path.light, frame)
            col, gl = ((255, 40, 30), (160, 20, 12)) if red else ((40, 255, 120), (20, 150, 70))
            dg.ellipse([sx - 10, sy - 10, sx + 10, sy + 10], fill=gl)
            dc.ellipse([sx - 1.8, sy - 1.8, sx + 1.8, sy + 1.8], fill=col)
        t = layers["t"]
        dg, dc = ImageDraw.Draw(t["glow"]), ImageDraw.Draw(t["core"])
        for (x, y) in self.deck_lamps():
            dg.ellipse([x - 16, y - 16, x + 16, y + 16], fill=(70, 56, 24))
            dc.ellipse([x - 1.5, y - 1.5, x + 1.5, y + 1.5], fill=(255, 226, 150))

        for (x, y, h, elev, kind, brake, key) in self.cars_at(frame):
            if x < -40 or x > CW + 40 or y < -40 or y > CH + 40:
                continue
            L = layers["t" if elev else "g"]
            spr = self.sprite(key, kind)
            rot = spr.rotate(-math.degrees(h), resample=Image.BICUBIC, expand=True)
            small = rot.resize((max(1, rot.width // SCALE), max(1, rot.height // SCALE)), Image.LANCZOS)
            px0, py0 = int(round(x - small.width / 2)), int(round(y - small.height / 2))
            _paste_l(L["shadow"], small.getchannel("A").point(lambda v: int(v * 0.75)), px0 + 2, py0 + 3)
            _paste_clip(L["body"], small, px0, py0)
            dg, dc = ImageDraw.Draw(L["glow"]), ImageDraw.Draw(L["core"])
            ux, uy = math.cos(h), math.sin(h)
            pxv, pyv = -uy, ux
            hl = spr.info_len / 2
            hx, hy = x + ux * hl, y + uy * hl
            bl, bw = 54, 13
            dg.polygon([(hx + pxv * 2.5, hy + pyv * 2.5), (hx - pxv * 2.5, hy - pyv * 2.5),
                        (hx + ux * bl - pxv * bw, hy + uy * bl - pyv * bw), (hx + ux * bl + pxv * bw, hy + uy * bl + pyv * bw)],
                       fill=(50, 44, 29))
            for side in (-spr.info_w / 2 + 1.2, spr.info_w / 2 - 1.2):
                dc.ellipse([hx + pxv * side - 1.3, hy + pyv * side - 1.3, hx + pxv * side + 1.3, hy + pyv * side + 1.3], fill=(214, 208, 190))
                tx, ty = x - ux * hl + pxv * side, y - uy * hl + pyv * side
                rr = 1.6 if brake else 1.1
                dc.ellipse([tx - rr, ty - rr, tx + rr, ty + rr], fill=(200, 34, 24))
                gg = 7 if brake else 5
                dg.ellipse([tx - gg, ty - gg, tx + gg, ty + gg], fill=(120, 14, 8) if brake else (72, 8, 4))
            if kind == "supercar":
                dg.ellipse([x - 11, y - 11, x + 11, y + 11], fill=(80, 10, 10))

        def pack(L):
            sh = np.asarray(L["shadow"].filter(ImageFilter.GaussianBlur(1.4)), np.float32)[..., None] / 255.0
            body = np.asarray(L["body"], np.float32) / 255.0
            lights = (np.asarray(L["glow"].filter(ImageFilter.GaussianBlur(6)), np.float32) / 255.0 * 0.9 +
                      np.asarray(L["core"].filter(ImageFilter.GaussianBlur(0.5)), np.float32) / 255.0)
            return sh, body, lights

        return pack(layers["g"]), pack(layers["t"])


def _paste_clip(dst, src, px, py):
    x0, y0 = max(0, -px), max(0, -py)
    if x0 >= src.width or y0 >= src.height or px >= dst.width or py >= dst.height:
        return
    dst.alpha_composite(src.crop((x0, y0, src.width, src.height)), (max(0, px), max(0, py)))


def _paste_l(dst, mask, px, py):
    x0, y0 = max(0, -px), max(0, -py)
    if x0 >= mask.width or y0 >= mask.height or px >= dst.width or py >= dst.height:
        return
    m = mask.crop((x0, y0, mask.width, mask.height))
    dst.paste(255, (max(0, px), max(0, py), max(0, px) + m.width, max(0, py) + m.height), m)


SEDAN_COLORS = [(22, 22, 26), (92, 18, 22), (150, 152, 160), (24, 34, 62), (58, 60, 66), (120, 98, 60)]


def car_sprite(kind, r):
    """Top-down car facing +x, drawn at SCALE x size."""
    dims = {"sedan": (18, 8), "taxi": (18, 8), "suv": (20, 9), "sports": (17, 7.4), "truck": (28, 10), "police": (19, 8.2), "supercar": (19, 8.6)}
    L, Wd = dims[kind]
    S = SCALE
    pad = 4
    img = Image.new("RGBA", (int((L + pad * 2) * S), int((Wd + pad * 2) * S)), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    ox, oy = pad * S, pad * S

    def R(x0, y0, x1, y1, fill, rad=2):
        d.rounded_rectangle([ox + x0 * S, oy + y0 * S, ox + x1 * S, oy + y1 * S], radius=rad * S, fill=fill)

    paint = {"taxi": (222, 168, 34), "police": (16, 16, 20), "truck": (178, 180, 186)}.get(kind)
    if kind == "suv":
        paint = r.choice([(14, 14, 18), (30, 32, 38), (52, 20, 24)])
    elif kind == "sports":
        paint = r.choice([(200, 150, 30), (160, 20, 24), (230, 230, 236)])
    elif paint is None:
        paint = r.choice(SEDAN_COLORS)
    hi = tuple(min(255, int(c * 1.35 + 18)) for c in paint)
    glass = (34, 44, 60)
    if kind == "supercar":
        red, deep, shine = (214, 18, 24), (130, 8, 12), (255, 120, 112)
        R(0, 0.2, L, Wd - 0.2, deep, 3.2)                       # body shadow edge
        R(0.3, 0.6, L - 0.3, Wd - 0.6, red, 3)                   # wedge body
        R(L * 0.40, 1.4, L * 0.70, Wd - 1.4, (14, 14, 18), 1.6)  # black canopy
        R(L * 0.46, 2.0, L * 0.64, Wd - 2.0, (40, 44, 54), 1)    # glass sheen
        R(L * 0.72, 1.3, L - 1.2, 2.2, shine, 1)                 # hood highlight
        R(L * 0.30, 0.3, L * 0.42, 1.2, (20, 6, 8), 0.4)         # side intakes
        R(L * 0.30, Wd - 1.2, L * 0.42, Wd - 0.3, (20, 6, 8), 0.4)
        R(0.2, 0.8, 1.6, Wd - 0.8, (18, 18, 20), 0.5)            # rear spoiler
        img.info_len, img.info_w = L, Wd
        return img
    if kind == "truck":
        R(0, 0, L * 0.72, Wd, paint, 1)
        R(0.6, 0.6, L * 0.72 - 0.6, 1.6, hi, 1)
        R(L * 0.74, 0.5, L, Wd - 0.5, (60, 64, 74), 2)
        R(L * 0.86, 1.2, L * 0.96, Wd - 1.2, glass, 1)
    else:
        R(0, 0, L, Wd, paint, 3 if kind != "sports" else 3.5)
        c0, c1 = (L * 0.3, L * 0.72) if kind != "sports" else (L * 0.36, L * 0.68)
        if kind == "suv":
            c0, c1 = L * 0.22, L * 0.8
        R(c0, 1.1, c1, Wd - 1.1, glass, 1.5)
        R(c0 + L * 0.06, 1.6, c1 - L * 0.08, Wd - 1.6, hi if kind != "police" else (230, 230, 236), 1)
        R(c1 + 0.8, 1.6, L - 1.4, 2.4, hi, 1)
        if kind == "police":
            R(0, 0, L * 0.26, Wd, (230, 230, 236), 2)
            R(L * 0.76, 0, L, Wd, (230, 230, 236), 2)
            R(L * 0.46, 0.8, L * 0.54, Wd - 0.8, (40, 40, 48), 0.5)
        if kind == "taxi":
            R(L * 0.45, Wd * 0.3, L * 0.56, Wd * 0.7, (250, 236, 170), 0.6)
        if kind == "sports":
            R(L * 0.05, Wd * 0.44, L * 0.95, Wd * 0.56, (20, 20, 22), 0.3)
    img.info_len = L
    img.info_w = Wd
    return img
