# Shot Creator Landing - Codebase Audit

**Generated:** 2026-01-28
**Repo Path:** `/Users/lmdhir/Downloads/shot-creator-landing-master`

---

## 1. Branch Analysis

### All Branches

| Branch | Type | Last Commit Date | Author | Last Commit Message |
|--------|------|-------------------|--------|---------------------|
| `master` (default) | local + remote | 2026-01-27 | Lamitr Dhir | Improve stance & alignment viz and add extracted angles table |
| `angle-extractor` | local + remote | 2026-01-27 | Lamitr Dhir | Improve stance & alignment viz and add extracted angles table |
| `product-ui` | local + remote | 2026-01-21 | Lamitr Dhir | Improve Player Match CTA and add 3rd quick fix |
| `skeleton-viewer` | local + remote | 2026-01-23 | Lamitr Dhir | Add 3D skeleton loop viewer with camera controls |
| `tool-version` | local only | 2026-01-19 | Lamitr Dhir | Save current tool version |
| `Shot_Tracker` | remote only | 2025-12-16 | Namrata Gupta | Add Shot Tracker code (Python files, docs, config) |
| `VideoPose3d` | remote only | 2026-01-06 | Lamitr Dhir | Replace Courier Prime font with Bebas Neue in tool |
| `videopose3d-integration` | remote only | 2026-01-21 | Namrata Gupta | Add webcam functionality, feedback panel... |

### Branch Divergence from `master`

| Branch | Behind master | Ahead of master | Status |
|--------|--------------|-----------------|--------|
| `angle-extractor` | 0 | 0 | Fully merged / identical |
| `product-ui` | 7 behind | 0 ahead | Stale — master has 7 commits not in this branch |
| `skeleton-viewer` | 4 behind | 1 ahead | Diverged |
| `tool-version` | 10 behind | 1 ahead | Diverged, significantly behind |

### Stale Branches (30+ days without commits)

| Branch | Last Commit | Days Since |
|--------|-------------|------------|
| `Shot_Tracker` (remote) | 2025-12-16 | **43 days** |

### Branch Cleanup Candidates

- **`angle-extractor`** — identical to `master`, can be deleted
- **`Shot_Tracker`** — 43 days stale, remote only
- **`tool-version`** — 10 commits behind master, only 1 unique commit ("Save current tool version")

---

## 2. Component Inventory

### Root — Landing Page (`/`)

| File | Purpose | Last Modified | Actively Used? |
|------|---------|---------------|----------------|
| `index.html` | Main landing page (9-section marketing site) | Recent | Yes — primary entry point |
| `index-athletic.html` | Alternative athletic-themed landing page | Older | Yes — standalone page |
| `styles.css` | Main landing page styles | Recent | **Unclear** — not linked in any HTML via grep |
| `styles-athletic.css` | Athletic theme styles | Older | Yes — linked in `index-athletic.html` |
| `styles-dark.css` | Dark theme styles | Older | **Not linked** in any HTML file |
| `script.js` | WebGL background, liquid glass effects, interactivity | Recent | Yes — used in `index.html` |
| `script-athletic.js` | Athletic theme JS (parallax, animations) | Older | Yes — used in `index-athletic.html` |
| `twitter-wall.js` | Twitter/X API integration for testimonial wall | Older | Yes — social proof section |

### `/shotsync/` — ShotSync Sub-page

| File | Purpose | Actively Used? |
|------|---------|----------------|
| `index.html` | ShotSync-specific landing or app page | Yes — linked from main site |

### `/shotlab/` — ShotLab Sub-page

| File | Purpose | Actively Used? |
|------|---------|----------------|
| `index.html` | ShotLab-specific landing or app page | Yes — linked from main site |

### `/components/` — TypeScript/React Components

| File | Purpose | Actively Used? |
|------|---------|----------------|
| `IdealShot.tsx` | React component for ideal shot analysis with Recharts radar visualization | **Not imported** by other files in this repo (likely consumed externally) |
| `ShotAnalysisRules.ts` | Rule engine for classifying shot metrics, fatigue detection, consistency scoring | **Not imported** in this repo (untracked file) |
| `DrillLibrary.ts` | Drill recommendation engine with practice session builder | **Not imported** in this repo (untracked file) |

### `/tool/` — Flask Backend & Analysis Tool

| File | Purpose | Actively Used? |
|------|---------|----------------|
| `app.py` | Flask server — video upload, MediaPipe pose detection, shot comparison, S3 integration | Yes — main backend |
| `index.html` | Tool UI — video upload, analysis display, webcam capture | Yes — tool frontend |
| `shot_stage_tf.py` | TensorFlow-based shot stage classification (heavily commented out) | Partially — mostly dead code |
| `style.css` | Tool styles (root level) | **Not referenced** by any HTML |
| `static/style.css` | Tool styles (static dir) | Yes — linked in tool UI |
| `static/app.js` | Tool frontend JS | Yes — linked in tool UI |
| `requirements.txt` | Python dependencies | Yes — pip install target |
| `player_data/` | JSON benchmark data for NBA players | Yes — loaded by `app.py` |
| `player_icons/` | Player headshot images | Yes — displayed in tool UI |

### `/analysis-*.html` — Analysis Pages

| File | Purpose | Actively Used? |
|------|---------|----------------|
| `analysis-ideal-form.html` | Ideal Form analysis results page | Yes — linked from ShotSync flow |
| `analysis-player-match.html` | Player Match comparison page | Yes — linked from ShotSync flow |

### Documentation Files

| File | Lines | Purpose |
|------|-------|---------|
| `README_FOR_ASSISTANT.md` | ~360 | AI assistant guide to the codebase |
| `QUICK_REFERENCE.md` | ~342 | 5-minute design overview |
| `SECTION_BREAKDOWN.md` | — | Visual specs with ASCII diagrams |
| `DESIGN_DOCUMENTATION.md` | ~895 | Complete design system specification |

---

## 3. Dead Code Detection

### Files Never Imported Anywhere

| File | Reason | Action |
|------|--------|--------|
| `components/IdealShot.tsx` | No import found in repo | May be consumed by an external build — verify |
| `components/ShotAnalysisRules.ts` | Untracked, not imported | New file, not yet integrated |
| `components/DrillLibrary.ts` | Untracked, not imported | New file, not yet integrated |
| `styles.css` | Not linked in any HTML `<link>` tag | Verify — may be inlined or loaded dynamically |
| `styles-dark.css` | Not linked in any HTML `<link>` tag | Likely unused |
| `tool/style.css` | Not referenced (duplicate of `tool/static/style.css`?) | Likely unused |

### Commented-Out Code Blocks

| File | Lines | Description |
|------|-------|-------------|
| `script-athletic.js` | 202–211 | Cursor trail effect — explicitly marked "Performance Heavy" |
| `tool/shot_stage_tf.py` | 1–8 | Commented-out imports |
| `tool/shot_stage_tf.py` | 12–392 | **~380 lines** of commented-out pose detection, angle calculation, and shot recording logic |
| `tool/shot_stage_tf.py` | 393–401 | Duplicate commented-out imports |

### TODO/FIXME Comments

**None found.** The codebase is clean of TODO/FIXME/HACK/XXX markers.

---

## 4. Dependency Check

### Python Dependencies (`tool/requirements.txt`)

| Dependency | Actually Imported? | Used In |
|------------|--------------------|---------|
| flask | Yes | `tool/app.py` |
| flask-cors | Yes | `tool/app.py` |
| mediapipe | Yes | `tool/app.py` |
| opencv-python (cv2) | Yes | `tool/app.py`, `tool/shot_stage_tf.py` |
| numpy | Yes | `tool/app.py`, `tool/shot_stage_tf.py` |
| scipy | Yes | `tool/app.py` |
| fastdtw | Yes | `tool/app.py` |
| boto3 | Yes | `tool/app.py` |
| torch | Yes | `tool/app.py` |
| torchvision | Yes | `tool/app.py` |
| ultralytics | Yes | `tool/app.py` |
| Pillow | Yes | `tool/app.py` |
| requests | Yes | `tool/app.py` |
| tqdm | Yes | Listed in requirements |
| filterpy | Yes | Listed in requirements |
| scikit-image | Yes | Listed in requirements |
| lap | Yes | Listed in requirements |
| cvzone | **Unclear** | Not found in direct imports — may be a transitive dep |
| hydra-core | **Unclear** | Not found in direct imports — may be a transitive dep |
| matplotlib | **Unclear** | Only in commented-out code in `shot_stage_tf.py` |
| PyYAML | **Unclear** | Not found in direct imports |

### Frontend Dependencies (via CDN, no package.json)

| Library | Source | Used In |
|---------|--------|---------|
| Google Fonts (Forum, Courier Prime, Bebas Neue, Work Sans, DM Sans) | CDN | Various HTML files |
| Icons8 | CDN | Icon assets |
| Recharts | Import in TSX | `components/IdealShot.tsx` |
| React | Import in TSX | `components/IdealShot.tsx` |

> **Note:** There is **no `package.json`** in this repo. The React/TypeScript components import React and Recharts but there's no local dependency management. These components may be intended for use in a separate build system.

---

## 5. Tech Stack Summary

| Layer | Technology | Version (if specified) |
|-------|-----------|----------------------|
| **Frontend** | Vanilla HTML5/CSS3/JS | — |
| **Frontend (components)** | React + TypeScript | — |
| **CSS Framework** | None (custom CSS) | — |
| **Visual Effects** | WebGL (custom shaders), CSS glassmorphism | — |
| **Backend** | Python Flask | 2.0.0+ |
| **Pose Estimation** | Google MediaPipe | Latest |
| **Object Detection** | YOLOv8 (Ultralytics) | Latest |
| **Deep Learning** | PyTorch + TorchVision | Latest |
| **Video Processing** | OpenCV (cv2) | 4.5.4+ |
| **Data Visualization** | Recharts (React) | — |
| **Sequence Comparison** | FastDTW + SciPy | — |
| **Cloud Storage** | AWS S3 (boto3) | — |
| **Hosting** | Netlify (static) | — |
| **Forms** | Typeform (embedded iframe) | — |
| **Social** | Twitter/X API | — |
| **Database/ORM** | **None** — file-based player data (JSON) | — |

---

## 6. Git Health

### Uncommitted Changes

```
 M shotsync/index.html          ← Modified, not staged
?? components/DrillLibrary.ts   ← Untracked
?? components/ShotAnalysisRules.ts  ← Untracked
```

### Stash

```
stash@{0}: WIP on angle-extractor: dd2ac04 Improve stance & alignment viz...
```

There is **1 stash** that may contain work-in-progress changes.

### Files That Should Probably Be Gitignored

| File | Reason |
|------|--------|
| `ScreenRecording_11-10-2025 19-48-36_1.mp4` | Large binary screen recording in repo root |
| `tool/roots_use.mov` | Video file (`.mp4` version is already gitignored but `.mov` is not) |

### Merge Conflicts

No merge conflicts detected on the current branch.

---

## Recommendations Summary

### High Priority
1. **Commit or discard** the modified `shotsync/index.html` and decide whether to track the two new `components/` files
2. **Gitignore** `ScreenRecording_11-10-2025 19-48-36_1.mp4` and `tool/roots_use.mov` — large binaries don't belong in git
3. **Delete `angle-extractor` branch** — identical to `master`

### Medium Priority
4. **Clean up `tool/shot_stage_tf.py`** — ~380 lines of commented-out code should be removed or archived
5. **Verify `styles.css` and `styles-dark.css`** — confirm whether they're loaded dynamically or truly unused
6. **Audit Python deps** — `cvzone`, `hydra-core`, `matplotlib`, `PyYAML` may be unused; remove if not needed
7. **Add `package.json`** if the `components/` TypeScript files are meant to be built locally

### Low Priority
8. **Delete or merge stale branches** — `Shot_Tracker` (43 days), `tool-version` (10 commits behind)
9. **Pop or drop the stash** — `stash@{0}` on `angle-extractor` may contain forgotten work
10. **Remove `tool/style.css`** if it's a duplicate of `tool/static/style.css`
