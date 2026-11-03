# LacunAI helmet identity

The supplied helmet starter is the visual source for this identity: an arched memory-clinic helmet, paired mechanical couplers, concentric left-side electrodes, an open right-side mesh, silver/ivory on navy, and orange AI lettering. The artwork is redrawn as clean vector geometry. No pixels from the supplied image are embedded.

The product name is **LacunAI**. The visual wordmark is **LACUNAI**, set in bespoke outlined serif capitals. The optional tagline is **For agents learning to forget.** It is an aspirational research line, not a claim of reliable deletion or erasure.

## Production assets

| File | Use |
| --- | --- |
| `lacunai-primary.svg` | Primary stacked lockup, tagline, and solid navy field |
| `lacunai-stacked-dark.svg` | Transparent stacked artwork for dark surfaces, without tagline |
| `lacunai-stacked-light.svg` | Transparent stacked artwork for light surfaces, without tagline |
| `lacunai-stacked-tagline-dark.svg` | Transparent dark-surface lockup with tagline |
| `lacunai-stacked-tagline-light.svg` | Transparent light-surface lockup with tagline |
| `lacunai-horizontal-dark.svg` | Horizontal companion with tagline for dark surfaces |
| `lacunai-horizontal-light.svg` | Horizontal companion with tagline for light surfaces |
| `lacunai-monochrome.svg` | Black stacked master, without tagline |
| `lacunai-icon.svg` | Simplified navy helmet icon for light surfaces |
| `lacunai-icon-dark.svg` | Simplified ivory helmet icon for dark surfaces |
| `lacunai-icon-monochrome.svg` | Black simplified helmet icon |
| `specimen.html` | Self-contained responsive specimen with embedded artwork |
| `build.py` | Deterministic generator for all production masters and specimen |
| `tagline-outlines.json` | Portable path data for the tagline; consumed by the generator |

The primary SVG deliberately includes its navy field. All other production SVG backgrounds are transparent, including the helmet opening and mesh. Production masters contain vector paths, ellipses, circles, local clipping geometry, and accessible title/description elements. They contain no raster images, live text, scripts, or external references.

The wordmark’s serif glyphs are custom path geometry. The tagline was converted locally from Helvetica Regular through CoreText and stored as path data; neither the SVGs nor the build requires the font or CoreText. The HTML specimen uses system fonts for explanatory copy and embeds all identity artwork. Its README link resolves locally.

## Geometry and responsive use

The full helmet is drawn in a `720 × 500` coordinate space. Its main shell follows an outer half-ellipse with radii `260 × 310` and an inner opening with radii `108 × 130`. Electrodes are generated inside the shell and transition from concentric closed rings to interrupted rings and curved bridges. The paired couplers are mirrored geometric assemblies.

The stacked artboard is `800 × 680` without the tagline and `800 × 750` with it. The horizontal artboard is `960 × 260`. The small icon uses a `128 × 128` artboard and deliberately replaces fine mesh with a few distinct closed and open electrode shapes. The specimen and inspection image show these icons at actual `16`, `24`, and `48` pixel sizes.

Use the full helmet for covers, splash screens, and larger lockups; use the simplified icon where fine mesh would disappear. At the smallest sizes, the arch and couplers carry recognition. Keep the supplied artboards intact, preserve the open center, and avoid stretching, glow, drop shadows, or additional texture. For one-color printing, use the monochrome master.

## Palette

| Role | Hex |
| --- | --- |
| Navy | `#081D36` |
| Ivory | `#F2E7D0` |
| Silver | `#D5D9DC` |
| AI orange | `#D77C35` |

## Verification

- Primary, transparent light/dark, and horizontal masters rendered locally with `rsvg-convert`; the resulting PNGs were visually inspected.
- Simplified icons rendered and inspected at their actual pixel sizes on both light and dark backgrounds.
- Every production SVG parsed as XML; accessible names and local clip references checked; no live text, embedded raster content, scripts, or external references.
- HTML structural checker passed with semantic structure and unique accessible IDs.
- Ruff formatting and lint checks passed for the generator; rebuilding preserves the generated asset bytes.
- Local browser access was disabled by Work settings. The specimen has **not** received browser visual, console, viewport, or interaction verification. Its responsive layout is implemented, not browser-verified.

Preview PNGs and `verification-symbol-sizes.svg` are inspection evidence. Named production SVGs are the master artwork. Previous identity directions are archived outside this project and are not part of this asset set.
