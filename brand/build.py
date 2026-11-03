# ruff: noqa: E501
# Long embedded SVG paths and HTML preserve deterministic asset bytes.
"""Rebuild original LacunAI vector masters and the self-contained specimen."""

from pathlib import Path

p = Path(__file__).parent
# Three archival strata, interrupted by a continuous absent strip.
# Both sides are separately drawn: the aperture is genuinely transparent.
strata = """<path d="M58 39 C42 40 32 45 32 51 C32 57 42 62 58 63 V58 C45 57 37 54 37 51 C37 48 45 45 58 44 Z M70 39 V44 C83 45 91 48 91 51 C91 54 83 57 70 58 V63 C86 62 96 57 96 51 C96 45 86 40 70 39 Z"/>
<path d="M32 61 V67 C32 73 42 78 58 79 V74 C45 73 37 70 37 67 V64 Z M91 64 V67 C91 70 83 73 70 74 V79 C86 78 96 73 96 67 V61 Z"/>
<path d="M32 77 V83 C32 89 42 94 58 95 V90 C45 89 37 86 37 83 V80 Z M91 80 V83 C91 86 83 89 70 90 V95 C86 94 96 89 96 83 V77 Z"/>"""
# Full seal: double engraved frame plus a modest indexing notch.
seal = (
    """<path fill-rule="evenodd" d="M64 10 A54 54 0 1 1 64 118 A54 54 0 1 1 64 10 Z M64 13 A51 51 0 1 0 64 115 A51 51 0 1 0 64 13 Z"/>
<path fill-rule="evenodd" d="M64 18 A46 46 0 1 1 64 110 A46 46 0 1 1 64 18 Z M64 19.5 A44.5 44.5 0 1 0 64 108.5 A44.5 44.5 0 1 0 64 19.5 Z"/>
<path d="M58 28 H70 V30 H58 Z"/>"""
    + strata
)
# Small-use adaptation retains the aperture and stacked strata, with one firm frame.
icon = (
    """<path fill-rule="evenodd" d="M64 8 A56 56 0 1 1 64 120 A56 56 0 1 1 64 8 Z M64 16 A48 48 0 1 0 64 112 A48 48 0 1 0 64 16 Z"/>"""
    + strata
)
# Bespoke transitional serif lettering: contrast, bracketed feet, and ink traps.
glyphs = [
    (
        "L",
        0,
        "M0 15 H29 V18 H24 C20 18 18 20 18 25 V78 H36 C44 78 48 72 51 62 H54 L51 84 H0 V81 H5 C10 81 10 79 10 75 V24 C10 20 8 18 4 18 H0 Z",
    ),
    (
        "a",
        63,
        "M32 49 C32 40 29 36 23 36 C18 36 14 38 12 42 C15 42 17 44 17 47 C17 50 15 52 12 52 C8 52 6 49 6 46 C6 39 14 32 24 32 C35 32 41 38 41 49 V75 C41 80 43 81 47 78 L48 81 C44 85 40 86 37 85 C34 84 32 81 32 78 C27 83 22 86 15 86 C6 86 1 80 1 73 C1 61 15 57 32 53 Z M32 57 C18 61 10 65 10 72 C10 78 13 81 18 81 C24 81 29 77 32 73 Z",
    ),
    (
        "c",
        119,
        "M39 42 C39 46 37 49 34 49 C31 49 29 47 29 44 C29 41 31 39 33 39 C30 36 27 35 23 35 C13 35 10 45 10 58 C10 72 15 81 25 81 C31 81 35 77 39 71 L42 73 C38 81 31 86 22 86 C8 86 0 75 0 60 C0 44 10 32 24 32 C33 32 39 36 39 42 Z",
    ),
    (
        "u",
        170,
        "M0 35 H17 V70 C17 78 20 81 26 81 C32 81 37 77 41 72 V42 C41 38 39 38 33 38 V35 H50 V75 C50 80 51 81 58 81 V84 H42 V77 C36 83 30 86 24 86 C13 86 8 80 8 70 V42 C8 38 6 38 0 38 Z",
    ),
    (
        "n",
        235,
        "M0 36 L16 32 V42 C22 35 28 32 35 32 C46 32 51 39 51 50 V75 C51 80 53 81 58 81 V84 H34 V81 C40 81 42 80 42 75 V51 C42 41 39 37 33 37 C27 37 22 41 17 47 V75 C17 80 19 81 25 81 V84 H1 V81 C6 81 8 80 8 75 V44 C8 40 7 39 0 39 Z",
    ),
    (
        "A",
        300,
        "M28 15 H31 L54 76 C56 80 58 81 63 81 V84 H35 V81 H40 C45 81 45 79 43 73 L39 61 H15 L11 73 C9 79 9 81 17 81 V84 H-3 V81 C3 81 5 78 7 73 Z M17 57 H37 L27 29 Z",
    ),
    (
        "I",
        372,
        "M0 15 H29 V18 H24 C20 18 19 20 19 25 V74 C19 79 20 81 24 81 H29 V84 H0 V81 H5 C9 81 10 79 10 74 V25 C10 20 9 18 5 18 H0 Z",
    ),
]


def word():
    return (
        '<g transform="translate(146 14)" fill-rule="evenodd">'
        + "".join(f'<path d="{d}" transform="translate({x} 0)"/>' for n, x, d in glyphs)
        + "</g>"
    )


def svg(color, small=False, seal_only=False):
    mark = icon if small else seal
    standalone = small or seal_only
    desc = (
        "An archival seal encloses three layered traces interrupted by a transparent vertical absence."
        + ("" if standalone else " Bespoke outlined serif lettering spells LacunAI.")
    )
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{128 if standalone else 568}" height="128" viewBox="0 0 {128 if standalone else 568} 128" role="img" aria-labelledby="title desc">
<title id="title">LacunAI{(" symbol" if standalone else "")}</title><desc id="desc">{desc}</desc>
<g fill="{color}">{mark}{"" if standalone else word()}</g>
</svg>\n'''


assets = {
    "lacunai-horizontal-light.svg": svg("#382A20"),
    "lacunai-horizontal-dark.svg": svg("#E8C68B"),
    "lacunai-monochrome.svg": svg("#000000"),
    "lacunai-icon.svg": svg("#382A20", small=True),
    "lacunai-icon-dark.svg": svg("#E8C68B", small=True),
    "lacunai-icon-monochrome.svg": svg("#000000", small=True),
    "lacunai-seal.svg": svg("#382A20", seal_only=True),
}
for name, content in assets.items():
    (p / name).write_text(content)
print(f"Wrote {len(assets)} self-contained serif/seal SVG assets.", flush=True)
serial = 0


def embed(name, cls="", size=None):
    global serial
    serial += 1
    s = assets[name].replace("<svg ", f'<svg class="{cls}" ', 1)
    if size is not None:
        s = s.replace('width="128" height="128"', f'width="{size}" height="{size}"')
    return (
        s.replace('id="title"', f'id="s{serial}-title"')
        .replace('id="desc"', f'id="s{serial}-desc"')
        .replace(
            'aria-labelledby="title desc"', f'aria-labelledby="s{serial}-title s{serial}-desc"'
        )
    )


html = (
    """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LacunAI — archival identity</title><style>
:root{color-scheme:light;--paper:#E8C68B;--cream:#F4E8D2;--ink:#382A20;--dark:#2E241C;--line:#84683F}*{box-sizing:border-box}body{margin:0;background:var(--cream);color:var(--ink);font:15px/1.6 "Courier New",Courier,monospace}main{max-width:1240px;margin:auto;padding:40px 42px 60px}header{display:flex;justify-content:space-between;gap:20px;font-size:11px;letter-spacing:.06em;border-bottom:1px solid var(--line);padding-bottom:18px}.stationery{margin:42px 0 44px;background:var(--paper);border:1px solid var(--ink);outline:1px solid var(--ink);outline-offset:-7px;padding:50px 56px 42px;position:relative;overflow:hidden;min-height:440px}.stationery .watermark{position:absolute;width:280px;height:280px;top:82px;left:calc(50% - 140px);opacity:.075;pointer-events:none}.letter{position:relative;z-index:1;max-width:660px;font-size:17px;line-height:1.7}.letter p{margin:0 0 25px}.letter .subject{font-weight:700;margin:30px 0 10px}.signature{display:flex;justify-content:flex-end;position:relative;margin:42px 0 0}.signature svg{width:min(100%,410px);height:auto}h1{font-family:Georgia,serif;font-weight:400;font-size:36px;line-height:1.18;letter-spacing:-.025em;margin:0 0 20px}h2{font-family:Georgia,serif;font-size:26px;font-weight:400;margin:0 0 20px}.caption{font-size:11px;letter-spacing:.06em;margin:0 0 24px}.pair{display:grid;grid-template-columns:1fr 1fr;border:1px solid var(--line)}.sample{padding:30px 32px;min-width:0}.sample svg.lockup{width:100%;height:auto;margin:12px 0 32px;display:block}.dark{background:var(--dark);color:var(--paper)}.sizes{display:flex;align-items:end;gap:36px}.size{display:grid;justify-items:center;gap:12px}.size span{font-size:11px;white-space:nowrap}.sizes svg{display:block}.details{display:grid;grid-template-columns:1fr 1fr;gap:60px;margin-top:42px}p{margin:0 0 20px}.swatches{display:grid;grid-template-columns:1fr 1fr;gap:20px}.chip{height:40px;border:1px solid var(--line);margin-bottom:6px}.swatches strong,.swatches code{display:block;font-size:11px}.swatches strong{font-weight:400}.mono{margin-top:36px;padding-top:26px;border-top:1px solid var(--line)}.mono svg{width:284px;height:64px;max-width:100%}.note{font-size:12px;margin-top:32px;max-width:760px}footer{font-size:11px;margin-top:30px;padding-top:20px;border-top:1px solid var(--line);display:flex;justify-content:space-between;gap:24px}a{color:inherit;text-underline-offset:4px}a:focus-visible{outline:2px solid var(--ink);outline-offset:5px}@media(max-width:680px){main{padding:22px 20px 36px}header{font-size:10px}.stationery{padding:35px 25px 30px;min-height:0;margin:28px 0}.letter{font-size:14px}.stationery .watermark{width:220px;height:220px;left:calc(50% - 110px);top:100px}.signature{margin-top:30px}.pair,.details{grid-template-columns:1fr}.details{gap:25px;margin-top:30px}.sample{padding:26px 23px}h1{font-size:30px}footer{flex-direction:column;gap:8px}}
</style></head><body><main><header><span>LacunAI / identity specimen</span><span>archival series</span></header>
<section class="stationery" aria-label="Stationery composition">"""
    + embed("lacunai-seal.svg", "watermark")
    + """<div class="letter"><p>research correspondence</p><p class="subject">on the structure of absence.</p><p>A study of targeted interventions in language-model weights and memory.</p><p>What changes?<br>What remains?</p></div><div class="signature">"""
    + embed("lacunai-horizontal-light.svg")
    + """</div></section>
<h1>A quiet mark for an open question.</h1><p>A restrained serif wordmark and an archival seal, drawn in warm paper and ink.</p>
<div class="pair"><section class="sample"><h2>Ink on paper</h2>"""
    + embed("lacunai-horizontal-light.svg", "lockup")
    + """<p class="caption">small-use symbol / actual pixel sizes</p><div class="sizes">"""
    + "".join(
        f'<div class="size">{embed("lacunai-icon.svg", size=s)}<span>{s} px</span></div>'
        for s in [16, 24, 48]
    )
    + """</div></section><section class="sample dark"><h2>Paper on ink</h2>"""
    + embed("lacunai-horizontal-dark.svg", "lockup")
    + """<p class="caption">small-use symbol / actual pixel sizes</p><div class="sizes">"""
    + "".join(
        f'<div class="size">{embed("lacunai-icon-dark.svg", size=s)}<span>{s} px</span></div>'
        for s in [16, 24, 48]
    )
    + """</div></section></div>
<div class="details"><section><h2>The missing strip</h2><p>Layered traces sit inside an engraved double-rule seal. A narrow transparent strip passes through every layer: an absence within the archive.</p><p>Bracketed serifs and a two-storey “a” lend the wordmark a printed, institutional character. The exact spelling remains LacunAI.</p><p>The small-use symbol reduces the frame to one firm ring while preserving the central aperture.</p></section><section><h2>Paper &amp; ink</h2><div class="swatches">"""
    + "".join(
        f'<div><div class="chip" style="background:{c}"></div><strong>{n}</strong><code>{c}</code></div>'
        for n, c in [
            ("Ochre paper", "#E8C68B"),
            ("Warm ivory", "#F4E8D2"),
            ("Brown ink", "#382A20"),
            ("Deep ink", "#2E241C"),
        ]
    )
    + """</div></section></div>
<section class="mono"><p class="caption">one-color reproduction</p>"""
    + embed("lacunai-monochrome.svg")
    + """</section><p class="note">A research identity, not a promise of memory erasure. The supplied film stationery informs the tone, type contrast, and framing; its wording, address, and watermark are not reproduced.</p><footer><span>transparent SVG / outlined lettering / no external dependencies</span><a href="README.md">asset notes</a></footer></main></body></html>"""
)
(p / "specimen.html").write_text(html)
print("Wrote embedded-SVG stationery specimen.", flush=True)
