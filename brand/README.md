# LacunAI identity

An original identity for a research tool exploring targeted language-model weight and memory manipulation. The supplied film stationery informs the warm ochre, brown ink, restrained serif lettering, typewriter-style specimen copy, and double-rule framing. The artwork does not reproduce the film’s logo, watermark, correspondence, or address. It does not promise memory erasure.

## Production assets

| File | Use |
| --- | --- |
| `lacunai-horizontal-light.svg` | Brown-ink horizontal logo for warm light surfaces |
| `lacunai-horizontal-dark.svg` | Ochre-paper horizontal logo for dark surfaces |
| `lacunai-monochrome.svg` | Black horizontal logo for single-color reproduction |
| `lacunai-icon.svg` | Brown-ink small-use symbol, transparent |
| `lacunai-icon-dark.svg` | Ochre-paper small-use symbol, transparent |
| `lacunai-icon-monochrome.svg` | Black small-use symbol, transparent |
| `lacunai-seal.svg` | Full double-rule seal for larger applications and watermarks |
| `specimen.html` | Self-contained responsive stationery specimen |
| `build.py` | Deterministic source generator for the masters and specimen |

Every production SVG uses filled paths and accessible title/description elements. All backgrounds and internal apertures are transparent. All lettering is bespoke outlined geometry: no fonts, raster images, scripts, masks, filters, or network dependencies. The HTML specimen embeds the SVGs and uses system fonts only for explanatory copy. Its README link resolves locally.

## Design

The seal’s layered traces are split by a continuous absent strip. The double frame suggests an institutional impression on paper; the interior suggests an archive with an omission. The absence is transparent geometry, not a background-colored rectangle.

The exact wordmark is **LacunAI**. Bespoke transitional serif letters use contrasting strokes, bracketed feet, ball terminals, and a two-storey “a”. The A and I remain ordinary capitals in the same ink. Their casing supplies the AI distinction without introducing a second visual language.

## Geometry and small use

- Seal artboard: `128 × 128`; center: `(64,64)`.
- Full seal outer radii: `54/51`; inner radii: `46/44.5`.
- Central absent strip: `12` units, between `x=58` and `x=70`.
- Horizontal artboard: `568 × 128`; wordmark begins at `x=146`.
- Wordmark capital top: `y=29`; baseline: `y=98`.
- Small-use symbol: one `8`-unit frame replaces the engraved double rule. The interior remains identical.
- The specimen displays that small-use symbol at actual `16`, `24`, and `48` CSS pixels. At the smallest size, the overall seal and central absence carry recognition; the thin internal strata are subordinate.

Use the small-use icon when the full seal’s fine double rule would become too small. Keep the complete horizontal artboard intact; add clear space outside it when placing it among other elements. Do not stretch the logo, close its aperture, add outlines, or introduce a glow.

## Palette

| Role | Hex |
| --- | --- |
| Ochre paper | `#E8C68B` |
| Warm ivory | `#F4E8D2` |
| Brown ink | `#382A20` |
| Deep ink | `#2E241C` |

## Verification

- Light and dark horizontal masters rendered locally with `rsvg-convert` and inspected as PNG previews.
- Small-use symbols rendered and inspected at actual pixel sizes on both surfaces.
- All production SVGs parsed as XML; accessible names present; no text elements, embedded raster content, scripts, or external references.
- HTML structural checker passed with semantic structure and unique accessible IDs.
- Local browser access was disabled by Work settings. The HTML specimen has **not** received browser visual, console, viewport, or interaction verification. Its responsive layouts are implemented, not browser-verified.

Preview PNGs and `verification-symbol-sizes.svg` are inspection evidence. The named production SVGs are the master artwork.
