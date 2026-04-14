# Camera Color Science Toolkit

A pipeline for measuring and characterizing the color accuracy of camera systems using an X-Rite ColorChecker Classic. Developed to compare a Sony Alpha 7 IV (ARW) against an iPhone 13 Pro Max under controlled lighting conditions.

---

## Pipeline Overview

```
RAW Ingest → Patch Extraction → CCM Computation → ΔE2000 Analysis → Gamut Visualization → Report
```

| Stage | Description |
|---|---|
| **RAW Ingest** | Load ARW file via `rawpy`, extract demosaiced linear RGB |
| **Patch Extraction** | Detect and sample the 24 ColorChecker patches |
| **CCM Computation** | Least-squares solve for a 3×3 camera RGB → linear sRGB matrix |
| **ΔE2000 Analysis** | Compare corrected patches to ColorChecker ground truth in Lab space |
| **Gamut Visualization** | Plot camera gamut vs. sRGB on a CIE xy chromaticity diagram |
| **Report Output** | Multi-figure visual summary export |

---

## Tech Stack

| Library | Role |
|---|---|
| `rawpy` | RAW ingest — demosaic + white balance metadata |
| `numpy` | CCM least-squares computation |
| `colormath` | Lab conversion and ΔE2000 calculation |
| `matplotlib` | Patch visualization, gamut plot, report figures |

---

## Shoot Setup

- **Location:** Indoors under controlled light
- **Light source:** Two Aputure LS 60x set to 75% intensity and 6500K
- **Framing:** Chart fills ~60–70% of the frame, shot straight-on, no shadows on patches

### Sony Alpha 7 IV
- **Settings:** Manual exposure (1/200 shutter, f/5.6, ISO 200), no in-camera corrections, Picture Profile set to neutral/flat, lossless compressed RAW
- **Lens:** Sony FE 24-70mm f/2.8 GM II stopped down to f/5.6 for field flatness

### iPhone 13 Pro Max
- **Settings:** AE lock, no additional corrections, RAW capture
- **Lens:** Default (main wide camera)

---

## Getting Started

### Prerequisites

```bash
pip install rawpy numpy colormath matplotlib
```

### Usage

```bash
python run_pipeline.py --input your_image.ARW
```

---

## Output

- Per-patch ΔE2000 error table
- Corrected vs. ground-truth patch color grid
- CIE xy chromaticity diagram with camera gamut overlaid on sRGB
- Exportable multi-figure report

---

## License

MIT License © 2026 Brandon J. Hernandez

---

## Reference

- [X-Rite ColorChecker Classic spectral data](https://www.xrite.com/service-support/new_colorchecker)
- [CIE ΔE2000 specification](http://www.ece.rochester.edu/~gsharma/ciede2000/)
- [rawpy documentation](https://letmaik.github.io/rawpy/api/)
