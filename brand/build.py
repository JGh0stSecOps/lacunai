# ruff: noqa: E501
# Long embedded SVG paths and HTML preserve deterministic asset bytes.
"""Rebuild the LacunAI helmet identity; every production glyph is outlined."""

import json
import math
from pathlib import Path

P = Path(__file__).parent
NAVY = "#081D36"
IVORY = "#F2E7D0"
SILVER = "#D5D9DC"
ORANGE = "#D77C35"
TAGLINE = json.loads((P / "tagline-outlines.json").read_text())
SHELL = "M100 350 A260 310 0 0 1 620 350 H468 A108 130 0 0 0 252 350 Z"


def arc(cx, cy, rx, ry, start, end, rotation=0):
    """Deterministic elliptical arc, with explicit open ends."""
    a, b, r = map(math.radians, [start, end, rotation])

    def point(t):
        x, y = rx * math.cos(t), ry * math.sin(t)
        return cx + x * math.cos(r) - y * math.sin(r), cy + x * math.sin(r) + y * math.cos(r)

    x1, y1 = point(a)
    x2, y2 = point(b)
    return f"M{x1:.2f} {y1:.2f} A{rx:.2f} {ry:.2f} {rotation:.2f} {int(end - start > 180)} 1 {x2:.2f} {y2:.2f}"


def helmet(color):
    nodes = []
    for row, y in enumerate([70, 107, 148, 191, 234, 277, 320]):
        for col in range(12):
            x = 118 + col * 44 + (19 if row % 2 else 0)
            if ((x - 360) / 246) ** 2 + ((y - 350) / 294) ** 2 > 0.96:
                continue
            if ((x - 360) / 128) ** 2 + ((y - 350) / 149) ** 2 < 1.05:
                continue
            rx = 10 + 8 * (1 - min(1, abs(x - 360) / 260))
            ry = 6.5 + y / 32
            rotation = (x - 360) / 260 * 54
            nodes.append((x, y, rx, ry, rotation, row, col))
    parts = [
        f'<g stroke="{color}" fill="none" stroke-linecap="round" stroke-linejoin="round">',
        f'<defs><clipPath id="helmet-shell"><path d="{SHELL}"/></clipPath></defs>',
        '<g clip-path="url(#helmet-shell)">',
    ]
    # Porous right side. Short curved bridges connect irregular, open electrodes.
    right = [n for n in nodes if n[0] > 405]
    for i, a in enumerate(right):
        candidates = sorted(
            (b for b in right[i + 1 :] if 25 < math.hypot(a[0] - b[0], a[1] - b[1]) < 62),
            key=lambda b: math.hypot(a[0] - b[0], a[1] - b[1]),
        )[:3]
        for j, b in enumerate(candidates):
            dx, dy = b[0] - a[0], b[1] - a[1]
            length = math.hypot(dx, dy)
            ux, uy = dx / length, dy / length
            ax, ay = a[0] + ux * a[2] * 0.85, a[1] + uy * a[3] * 0.85
            bx, by = b[0] - ux * b[2] * 0.85, b[1] - uy * b[3] * 0.85
            bend = (1 if (i + j) % 2 else -1) * 9
            cx, cy = (ax + bx) / 2 - uy * bend, (ay + by) / 2 + ux * bend
            parts.append(
                f'<path stroke-width="1.65" d="M{ax:.2f} {ay:.2f} Q{cx:.2f} {cy:.2f} {bx:.2f} {by:.2f}"/>'
            )
    for x, y, rx, ry, rotation, row, col in nodes:
        if x < 380:
            parts.append(
                f'<ellipse cx="{x}" cy="{y}" rx="{rx:.2f}" ry="{ry:.2f}" transform="rotate({rotation:.2f} {x} {y})" stroke-width="2.5"/>'
            )
            parts.append(
                f'<ellipse cx="{x}" cy="{y}" rx="{rx * 0.66:.2f}" ry="{ry * 0.65:.2f}" transform="rotate({rotation:.2f} {x} {y})" stroke-width="1.65"/>'
            )
            parts.append(
                f'<ellipse cx="{x}" cy="{y}" rx="{rx * 0.18:.2f}" ry="{ry * 0.18:.2f}" transform="rotate({rotation:.2f} {x} {y})" fill="{color}" stroke="none"/>'
            )
        elif x < 425:
            parts.append(
                f'<ellipse cx="{x}" cy="{y}" rx="{rx:.2f}" ry="{ry:.2f}" transform="rotate({rotation:.2f} {x} {y})" stroke-width="2.3"/>'
            )
            parts.append(
                f'<path stroke-width="1.6" d="{arc(x, y, rx * 0.72, ry * 0.72, 35, 318, rotation)}"/>'
            )
        else:
            start = (row * 29 + col * 17) % 90
            end = start + (287 if (row + col) % 3 else 329)
            parts.append(
                f'<path stroke-width="2.2" d="{arc(x, y, rx * (1.08 if row % 2 else 0.94), ry * 1.08, start, end, rotation)}"/>'
            )
    parts.extend(
        [
            "</g>",
            f'<path d="{SHELL}" stroke-width="3.6"/>',
            '<path d="M105 346 A255 304 0 0 1 615 346 M257 345 V336 A103 120 0 0 1 463 336 V345" stroke-width="1.2"/>',
        ]
    )
    # Flat hardware avoids faux 3D rendering; coupler silhouettes retain the reference.
    left = f'''<g stroke-width="2.2"><path d="M73 339 H98 V351 H73 Z" fill="{color}"/><path d="M63 355 H125 V372 H63 Z"/><path d="M68 378 H120 V383 H68 Z" fill="{color}"/><path d="M56 389 H125 V418 H56 Z" fill="{color}"/><path d="M34 394 H49 V413 H34 Z" fill="{color}"/><path d="M132 396 H169 V411 H132 Z"/><path d="M140 403 H151 M157 403 H161" stroke-width="3.5"/><path d="M64 425 H116 L107 474 L92 482 L72 469 Z"/><path d="M72 427 L78 465 L92 474 L100 427" stroke-width="1.2"/><path d="M79 480 L83 489 H102 L105 480"/></g>'''
    parts.extend([left, f'<g transform="translate(720 0) scale(-1 1)">{left}</g>', "</g>"])
    return "".join(parts)


GLYPHS = [
    (
        "L",
        0,
        "M0 15 H29 V18 H24 C20 18 18 20 18 25 V78 H36 C44 78 48 72 51 62 H54 L51 84 H0 V81 H5 C10 81 10 79 10 75 V24 C10 20 8 18 4 18 H0 Z",
    ),
    (
        "A",
        68,
        "M28 15 H31 L54 76 C56 80 58 81 63 81 V84 H35 V81 H40 C45 81 45 79 43 73 L39 61 H15 L11 73 C9 79 9 81 17 81 V84 H-3 V81 C3 81 5 78 7 73 Z M17 57 H37 L27 29 Z",
    ),
    (
        "C",
        146,
        "M60 18 H63 V37 H60 C56 24 48 17 36 17 C20 17 12 29 12 49 C12 69 21 82 37 82 C48 82 57 76 64 66 L67 68 C60 80 49 87 35 87 C14 87 1 71 1 51 C1 29 15 13 36 13 C46 13 53 16 59 21 Z",
    ),
    (
        "U",
        225,
        "M0 15 H29 V18 H24 C20 18 19 20 19 25 V61 C19 76 25 82 38 82 C52 82 59 74 59 60 V27 C59 20 57 18 48 18 V15 H73 V18 C64 18 63 20 63 27 V60 C63 78 53 87 36 87 C18 87 9 79 9 62 V25 C9 20 8 18 4 18 H0 Z",
    ),
    (
        "N",
        311,
        "M0 15 H18 L61 68 V29 C61 20 59 18 49 18 V15 H76 V18 C66 18 65 20 65 29 V86 H61 L14 29 V70 C14 79 16 81 26 81 V84 H0 V81 C9 81 10 79 10 70 V25 C7 20 5 18 0 18 Z",
    ),
    (
        "A",
        400,
        "M28 15 H31 L54 76 C56 80 58 81 63 81 V84 H35 V81 H40 C45 81 45 79 43 73 L39 61 H15 L11 73 C9 79 9 81 17 81 V84 H-3 V81 C3 81 5 78 7 73 Z M17 57 H37 L27 29 Z",
    ),
    (
        "I",
        482,
        "M0 15 H29 V18 H24 C20 18 19 20 19 25 V74 C19 79 20 81 24 81 H29 V84 H0 V81 H5 C9 81 10 79 10 74 V25 C10 20 9 18 5 18 H0 Z",
    ),
]


def wordmark(ink, accent):
    return (
        '<g fill-rule="evenodd">'
        + "".join(
            f'<path fill="{accent if index >= 5 else ink}" d="{d}" transform="translate({x} 0)"/>'
            for index, (_, x, d) in enumerate(GLYPHS)
        )
        + "</g>"
    )


def micro(color):
    return f'''<g fill="none" stroke="{color}" stroke-linecap="round" stroke-linejoin="round"><path d="M18 91 V77 C18 40 37 18 64 18 C91 18 110 40 110 77 V91 H84 V79 C84 64 76 56 64 56 C52 56 44 64 44 79 V91 Z" stroke-width="7"/><path d="M15 97 H35 M93 97 H113 M24 99 V112 M104 99 V112" stroke-width="8"/><circle cx="43" cy="45" r="6" stroke-width="4"/><circle cx="30" cy="71" r="5" stroke-width="4"/><path d="M62 33 A6 6 0 1 0 68 39 M81 43 L88 47 L86 55 M97 63 L101 69 L97 76" stroke-width="4"/></g>'''


def master(kind="stacked", dark=False, mono=False, tagline=False, background=False):
    ink = "#000000" if mono else (IVORY if dark else NAVY)
    metal = ink if mono or not dark else SILVER
    accent = ink if mono else ORANGE
    if kind == "icon":
        width, height, content = 128, 128, micro(ink)
    elif kind == "horizontal":
        width, height = 960, 260
        content = f'<g transform="translate(7 1) scale(.5)">{helmet(metal)}</g><g transform="translate(397 64) scale(1.02)">{wordmark(ink, accent)}</g>'
        if tagline:
            content += f'<path fill="{ink}" d="{TAGLINE["path"]}" transform="translate(397 191)"/>'
    else:
        width, height = 800, 750 if tagline else 680
        content = f'<g transform="translate(40 0)">{helmet(metal)}</g><g transform="translate(93.4 522) scale(1.2)">{wordmark(ink, accent)}</g>'
        if tagline:
            content += f'<path fill="{ink}" d="{TAGLINE["path"]}" transform="translate({(800 - TAGLINE["width"]) / 2:.3f} 687)"/>'
    if background:
        content = f'<path fill="{NAVY}" d="M0 0 H{width} V{height} H0 Z"/>' + content
    desc = "An arched memory-clinic helmet with paired couplers. Concentric electrodes on the left open into a fragmented mesh on the right."
    if kind != "icon":
        desc += (
            " Serif capitals read LACUNAI, with AI in orange."
            if not mono
            else " Serif capitals read LACUNAI."
        )
    if tagline:
        desc += " For agents learning to forget."
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc"><title id="title">LacunAI</title><desc id="desc">{desc}</desc>{content}</svg>\n'


ASSETS = {
    "lacunai-primary.svg": master(dark=True, tagline=True, background=True),
    "lacunai-stacked-dark.svg": master(dark=True),
    "lacunai-stacked-light.svg": master(),
    "lacunai-stacked-tagline-dark.svg": master(dark=True, tagline=True),
    "lacunai-stacked-tagline-light.svg": master(tagline=True),
    "lacunai-horizontal-dark.svg": master(kind="horizontal", dark=True, tagline=True),
    "lacunai-horizontal-light.svg": master(kind="horizontal", tagline=True),
    "lacunai-monochrome.svg": master(mono=True),
    "lacunai-icon.svg": master(kind="icon"),
    "lacunai-icon-dark.svg": master(kind="icon", dark=True),
    "lacunai-icon-monochrome.svg": master(kind="icon", mono=True),
}
for name, value in ASSETS.items():
    (P / name).write_text(value)
print(f"Generated {len(ASSETS)} helmet identity masters.", flush=True)
SERIAL = 0


def embed(name, css_class="", size=None):
    global SERIAL
    SERIAL += 1
    value = ASSETS[name].replace("<svg ", f'<svg class="{css_class}" ', 1)
    if size:
        value = value.replace('width="128" height="128"', f'width="{size}" height="{size}"')
    for old in ["title", "desc", "helmet-shell"]:
        value = value.replace(f'id="{old}"', f'id="a{SERIAL}-{old}"').replace(
            f"url(#{old})", f"url(#a{SERIAL}-{old})"
        )
    return value.replace(
        'aria-labelledby="title desc"', f'aria-labelledby="a{SERIAL}-title a{SERIAL}-desc"'
    )


HTML = (
    """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LacunAI — helmet identity</title><style>
:root{color-scheme:light;--navy:#081D36;--ivory:#F2E7D0;--silver:#D5D9DC;--orange:#D77C35}*{box-sizing:border-box}body{margin:0;background:var(--ivory);color:var(--navy);font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}main{max-width:1240px;margin:auto;padding:34px 44px 54px}header{display:flex;justify-content:space-between;gap:22px;font-size:11px;letter-spacing:.12em;text-transform:uppercase;padding-bottom:20px;border-bottom:1px solid #081d3640}.hero{background:var(--navy);margin-top:30px;text-align:center}.hero svg{display:block;width:min(100%,720px);height:auto;margin:auto}h1{font:400 35px/1.18 Georgia,serif;letter-spacing:-.02em;margin:36px 0 16px}h2{font:400 25px/1.2 Georgia,serif;margin:0 0 20px}.intro{max-width:740px;margin-bottom:30px}.pair{display:grid;grid-template-columns:1fr 1fr;border:1px solid #081d3640}.sample{padding:30px 28px;min-width:0}.dark{background:var(--navy);color:var(--ivory)}.companion{width:100%;height:auto;display:block;margin:12px 0 30px}.caption{font-size:11px;letter-spacing:.06em;margin:0 0 22px}.sizes{display:flex;gap:32px;align-items:end}.size{display:grid;gap:12px;justify-items:center}.size svg{display:block}.size span{font-size:11px}.details{display:grid;grid-template-columns:1fr 1fr;gap:55px;margin-top:40px}p{margin:0 0 20px}.swatches{display:grid;grid-template-columns:1fr 1fr;gap:18px}.chip{height:40px;border:1px solid #081d3630;margin-bottom:5px}.swatches strong,.swatches code{display:block;font-size:11px}.swatches strong{font-weight:500}.mono{margin-top:30px;border-top:1px solid #081d3640;padding-top:26px;display:flex;align-items:center;gap:40px}.mono svg{width:240px;height:auto}.mono p{max-width:420px;font-size:13px}footer{display:flex;justify-content:space-between;gap:22px;border-top:1px solid #081d3640;margin-top:32px;padding-top:22px;font-size:11px}a{color:inherit;text-underline-offset:4px}a:focus-visible{outline:2px solid var(--orange);outline-offset:4px}@media(max-width:680px){main{padding:22px 18px 36px}header{font-size:9px}.pair,.details{grid-template-columns:1fr}.sample{padding:26px 22px}.details{gap:22px}.mono{flex-direction:column;align-items:start;gap:20px}footer{flex-direction:column;gap:8px}h1{font-size:30px}.hero{margin-top:22px}}
</style></head><body><main><header><span>LacunAI / identity specimen</span><span>helmet series</span></header><section class="hero" aria-label="Primary identity">"""
    + embed("lacunai-primary.svg")
    + """</section><h1>For agents learning to forget.</h1><p class="intro">A memory-clinic helmet becomes a research emblem: orderly concentric electrodes give way to open, interrupted connections. The name is LacunAI; the visual wordmark is set in serif capitals.</p><div class="pair"><section class="sample"><h2>Light surface</h2>"""
    + embed("lacunai-horizontal-light.svg", "companion")
    + """<p class="caption">responsive helmet / actual pixel sizes</p><div class="sizes">"""
    + "".join(
        f'<div class="size">{embed("lacunai-icon.svg", size=s)}<span>{s} px</span></div>'
        for s in [16, 24, 48]
    )
    + """</div></section><section class="sample dark"><h2>Navy surface</h2>"""
    + embed("lacunai-horizontal-dark.svg", "companion")
    + """<p class="caption">responsive helmet / actual pixel sizes</p><div class="sizes">"""
    + "".join(
        f'<div class="size">{embed("lacunai-icon-dark.svg", size=s)}<span>{s} px</span></div>'
        for s in [16, 24, 48]
    )
    + """</div></section></div><div class="details"><section><h2>Ordered, then open.</h2><p>The helmet follows the supplied starter’s arched silhouette, paired mechanical couplers, and transition from concentric electrodes to broken mesh. Fine outlines are reserved for the large mark. Small icons preserve the arch, couplers, and directional change.</p><p>The tagline is an aspirational research line. It does not assert that an agent can reliably erase a memory or remove its influence.</p></section><section><h2>Navy, ivory, orange</h2><div class="swatches">"""
    + "".join(
        f'<div><div class="chip" style="background:{c}"></div><strong>{n}</strong><code>{c}</code></div>'
        for n, c in [("Navy", NAVY), ("Ivory", IVORY), ("Silver", SILVER), ("AI orange", ORANGE)]
    )
    + """</div></section></div><section class="mono">"""
    + embed("lacunai-monochrome.svg")
    + """<p>The one-color master keeps the full helmet and serif lettering. All glyphs, including the optional tagline, are vector outlines. Transparent variants remain transparent throughout the shell and mesh.</p></section><footer><span>vector masters / outlined typography / no external dependencies</span><a href="README.md">asset notes</a></footer></main></body></html>"""
)
(P / "specimen.html").write_text(HTML)
print("Generated self-contained helmet specimen.", flush=True)
