"""Build assets/banner.svg: a night city skyline over the pond, name in gold.

Usage: python banner.py path/to/orbitron-900.woff2 path/to/orbitron-700.woff2 assets/banner.svg
"""

import base64
import random
import sys

W, H = 1200, 340
HORIZON = 238  # where the skyline meets the water
rng = random.Random(1939)


def building_path(x, w, h, kind):
    """Silhouette for one building, from the horizon up."""
    top = HORIZON - h
    pts = [(x, HORIZON), (x, top)]
    if kind == "spire":
        pts += [(x + w * 0.35, top), (x + w * 0.5, top - h * 0.28), (x + w * 0.65, top), (x + w, top)]
    elif kind == "stepped":
        s = w * 0.2
        pts += [(x + s, top), (x + s, top - 14), (x + w - s, top - 14), (x + w - s, top), (x + w, top)]
    elif kind == "gothic":
        pts += [(x + w * 0.15, top), (x + w * 0.15, top - 10), (x + w * 0.22, top - 22), (x + w * 0.29, top - 10),
                (x + w * 0.29, top), (x + w * 0.71, top), (x + w * 0.71, top - 10), (x + w * 0.78, top - 22),
                (x + w * 0.85, top - 10), (x + w * 0.85, top), (x + w, top)]
    elif kind == "antenna":
        pts += [(x + w * 0.48, top), (x + w * 0.48, top - h * 0.22), (x + w * 0.52, top - h * 0.22), (x + w * 0.52, top), (x + w, top)]
    else:
        pts += [(x + w, top)]
    pts += [(x + w, HORIZON)]
    return "M" + " L".join(f"{px:.1f},{py:.1f}" for px, py in pts) + " Z"


def skyline(layer_scale, color, count_range, seed_off, windows):
    r = random.Random(1939 + seed_off)
    paths, wins = [], []
    x = -20
    while x < W + 20:
        w = r.uniform(*count_range)
        h = r.uniform(50, 150) * layer_scale
        # keep the middle lower so the name reads cleanly
        if 230 < x < 970:
            h *= 0.42
        kind = r.choice(["flat", "flat", "spire", "stepped", "gothic", "antenna"])
        paths.append(building_path(x, w, h, kind))
        if windows:
            for wy in range(int(HORIZON - h + 10), HORIZON - 8, 9):
                for wx in range(int(x + 5), int(x + w - 6), 8):
                    if r.random() < 0.16:
                        delay = r.uniform(0, 8)
                        dur = r.uniform(3, 9)
                        wins.append(f'<rect x="{wx}" y="{wy}" width="3" height="4" class="win" '
                                    f'style="animation-delay:{delay:.1f}s;animation-duration:{dur:.1f}s"/>')
        x += w + r.uniform(-4, 6)
    body = "".join(f'<path d="{p}"/>' for p in paths)
    return f'<g fill="{color}">{body}</g>', "".join(wins)


def main(f900, f700, out):
    b900 = base64.b64encode(open(f900, "rb").read()).decode()
    b700 = base64.b64encode(open(f700, "rb").read()).decode()
    far, _ = skyline(0.85, "#121726", (26, 52), 1, False)
    near, wins = skyline(1.15, "#06070c", (34, 70), 2, True)

    stars = "".join(
        f'<circle cx="{rng.uniform(0, W):.0f}" cy="{rng.uniform(0, HORIZON - 70):.0f}" r="{rng.choice([0.6, 0.8, 1.0, 1.3])}" '
        f'class="star" style="animation-delay:{rng.uniform(0, 6):.1f}s;animation-duration:{rng.uniform(3, 7):.1f}s"/>'
        for _ in range(90)
    )
    flies = "".join(
        f'<circle cx="{rng.uniform(40, W - 40):.0f}" cy="{rng.uniform(HORIZON + 12, H - 14):.0f}" r="1.8" class="fly" '
        f'style="animation-delay:{rng.uniform(0, 6):.1f}s;animation-duration:{rng.uniform(4, 8):.1f}s"/>'
        for _ in range(12)
    )
    shimmer = "".join(
        f'<rect x="{rng.uniform(80, W - 200):.0f}" y="{HORIZON + 6 + i * 7}" width="{rng.uniform(40, 160):.0f}" height="1.2" '
        f'class="shim" style="animation-delay:{rng.uniform(0, 4):.1f}s"/>'
        for i in range(13)
    )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 {W} {H}" width="{W}" height="{H}">
<style>
@font-face{{font-family:Orb;font-weight:900;src:url(data:font/woff2;base64,{b900}) format("woff2")}}
@font-face{{font-family:Orb;font-weight:700;src:url(data:font/woff2;base64,{b700}) format("woff2")}}
.star{{fill:#cfd8ff;animation:tw ease-in-out infinite}}
.win{{fill:#f2c14e;animation:win ease-in-out infinite}}
.fly{{fill:#ffd36b;filter:url(#glow);animation:fly ease-in-out infinite}}
.shim{{fill:#c9d6ff;opacity:0;animation:shim 4s ease-in-out infinite}}
@keyframes tw{{0%,100%{{opacity:.2}}50%{{opacity:1}}}}
@keyframes win{{0%,100%{{opacity:.95}}45%{{opacity:.95}}50%{{opacity:.25}}55%{{opacity:.95}}}}
@keyframes fly{{0%,100%{{opacity:.15;transform:translate(0,0)}}50%{{opacity:1;transform:translate(6px,-5px)}}}}
@keyframes shim{{0%,100%{{opacity:0}}50%{{opacity:.35}}}}
@keyframes sweepA{{0%,100%{{transform:rotate(-28deg)}}50%{{transform:rotate(14deg)}}}}
@keyframes sweepB{{0%,100%{{transform:rotate(22deg)}}50%{{transform:rotate(-16deg)}}}}
@keyframes drift{{from{{transform:translateX(-260px)}}to{{transform:translateX(1260px)}}}}
@keyframes rise{{from{{opacity:0;transform:translateY(10px)}}to{{opacity:1;transform:none}}}}
@keyframes pulse{{0%,100%{{opacity:.55}}50%{{opacity:.9}}}}
.t{{font-family:Orb,sans-serif;font-weight:900;font-size:66px;letter-spacing:12px;fill:url(#gold);animation:rise 1.4s ease-out both}}
.s{{font-family:Orb,sans-serif;font-weight:700;font-size:17px;letter-spacing:8px;fill:#c9b37a;animation:rise 1.4s ease-out .5s both}}
</style>
<defs>
<linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">
  <stop offset="0" stop-color="#020306"/><stop offset=".6" stop-color="#0a0f1c"/><stop offset="1" stop-color="#141b2e"/>
</linearGradient>
<linearGradient id="water" x1="0" y1="0" x2="0" y2="1">
  <stop offset="0" stop-color="#0d1c27"/><stop offset="1" stop-color="#05090f"/>
</linearGradient>
<linearGradient id="gold" x1="0" y1="0" x2="0" y2="1">
  <stop offset="0" stop-color="#fff1b8"/><stop offset=".45" stop-color="#e3b341"/><stop offset="1" stop-color="#9a6b12"/>
</linearGradient>
<radialGradient id="moonGlow"><stop offset="0" stop-color="#e8eeff" stop-opacity=".55"/><stop offset="1" stop-color="#e8eeff" stop-opacity="0"/></radialGradient>
<radialGradient id="moonBody" cx=".42" cy=".4"><stop offset="0" stop-color="#fbfcff"/><stop offset="1" stop-color="#b9c3dc"/></radialGradient>
<linearGradient id="beam" x1="0" y1="1" x2="0" y2="0"><stop offset="0" stop-color="#f6d77a" stop-opacity=".38"/><stop offset="1" stop-color="#f6d77a" stop-opacity="0"/></linearGradient>
<linearGradient id="fade" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#fff" stop-opacity=".5"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></linearGradient>
<mask id="reflMask"><rect x="0" y="{HORIZON}" width="{W}" height="{H - HORIZON}" fill="url(#fade)"/></mask>
<filter id="glow" x="-200%" y="-200%" width="500%" height="500%"><feGaussianBlur stdDeviation="2.5" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
<filter id="textGlow" x="-10%" y="-40%" width="120%" height="180%"><feGaussianBlur stdDeviation="6" result="b"/><feColorMatrix in="b" values="1 0 0 0 0.35  0 .8 0 0 .22  0 0 .2 0 0  0 0 0 .55 0"/><feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge></filter>
<filter id="ripple"><feTurbulence type="fractalNoise" baseFrequency="0.004 0.09" numOctaves="2" seed="4"/><feDisplacementMap in="SourceGraphic" scale="9"/></filter>
<filter id="soft" x="-100%" y="-100%" width="300%" height="300%"><feGaussianBlur stdDeviation="14"/></filter>
<clipPath id="frame"><rect width="{W}" height="{H}"/></clipPath>
<linearGradient id="fadeB" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#0d1117" stop-opacity="0"/><stop offset="1" stop-color="#0d1117"/></linearGradient>
<linearGradient id="fadeL" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#0d1117"/><stop offset="1" stop-color="#0d1117" stop-opacity="0"/></linearGradient>
<linearGradient id="fadeR" x1="1" y1="0" x2="0" y2="0"><stop offset="0" stop-color="#0d1117"/><stop offset="1" stop-color="#0d1117" stop-opacity="0"/></linearGradient>
<linearGradient id="fadeT" x1="0" y1="1" x2="0" y2="0"><stop offset="0" stop-color="#0d1117" stop-opacity="0"/><stop offset="1" stop-color="#0d1117"/></linearGradient>
</defs>
<g clip-path="url(#frame)">
<rect width="{W}" height="{H}" fill="url(#sky)"/>
{stars}
<circle cx="1010" cy="78" r="120" fill="url(#moonGlow)" style="animation:pulse 8s ease-in-out infinite"/>
<circle cx="1010" cy="78" r="38" fill="url(#moonBody)"/>
<circle cx="998" cy="70" r="6" fill="#aab4cf" opacity=".5"/><circle cx="1022" cy="90" r="4" fill="#aab4cf" opacity=".45"/>
<g filter="url(#soft)" opacity=".55">
  <ellipse cx="0" cy="62" rx="210" ry="20" fill="#2a3350" style="animation:drift 70s linear infinite"/>
  <ellipse cx="0" cy="118" rx="160" ry="16" fill="#222a44" style="animation:drift 95s linear -40s infinite"/>
</g>
<g style="transform-origin:180px {HORIZON - 60}px;animation:sweepA 11s ease-in-out infinite"><polygon points="180,{HORIZON - 60} 150,-60 210,-60" fill="url(#beam)"/></g>
<g style="transform-origin:1080px {HORIZON - 70}px;animation:sweepB 13s ease-in-out infinite"><polygon points="1080,{HORIZON - 70} 1045,-60 1115,-60" fill="url(#beam)"/></g>
<g id="city">
{far}
{near}
<g>{wins}</g>
</g>
<rect x="0" y="{HORIZON}" width="{W}" height="{H - HORIZON}" fill="url(#water)"/>
<g mask="url(#reflMask)" filter="url(#ripple)"><use xlink:href="#city" href="#city" transform="translate(0 {2 * HORIZON}) scale(1 -1)"/></g>
<g filter="url(#soft)"><rect x="985" y="{HORIZON + 4}" width="40" height="{H - HORIZON - 30}" fill="#dfe6ff" opacity=".28" style="animation:pulse 5s ease-in-out infinite"/></g>
{shimmer}
{flies}
<line x1="120" y1="{HORIZON}" x2="{W - 120}" y2="{HORIZON}" stroke="#e3b341" stroke-opacity=".25"/>
<ellipse cx="600" cy="138" rx="430" ry="70" fill="#020306" opacity=".55" filter="url(#soft)"/>
<rect x="0" y="{H - 45}" width="{W}" height="45" fill="url(#fadeB)"/>
<rect x="0" y="0" width="90" height="{H}" fill="url(#fadeL)"/>
<rect x="{W - 90}" y="0" width="90" height="{H}" fill="url(#fadeR)"/>
<rect x="0" y="0" width="{W}" height="30" fill="url(#fadeT)"/>
<text class="t" x="600" y="128" text-anchor="middle" filter="url(#textGlow)">RAHUL KAURA</text>
<text class="s" x="600" y="170" text-anchor="middle">FORWARD DEPLOYED ENGINEER · UNIPHORE</text>
</g>
</svg>'''
    open(out, "w").write(svg)


if __name__ == "__main__":
    main(*sys.argv[1:4])
