# Internal Data Plan: Getting grid-cv to Production

The current detector was trained on **InsPLAD + MPID**, which are public
*transmission* datasets licensed for **non-commercial use only**. It is good
for proving the pipeline works. It is **not** usable for real inspections, for
two reasons:

1. **License.** The weights come from non-commercial data. A production model
   must be trained from base weights on our own imagery, and never fine-tuned
   from the POC checkpoint.
2. **Domain.** Transmission towers are not distribution poles. The model has
   never seen a crossarm, pole-top transformer, cutout, or distribution
   insulator from our system.

This document covers what has to happen on our side to close that gap.

---

## 1. Approvals and decisions (before anyone labels anything)

| # | Item | Owner | Notes |
|---|------|-------|-------|
| 1 | **Confirm we own or can use the photos for ML training** | Legal / Asset Mgmt | Photos shot by contractors: check the contract. Usage rights for "inspection" don't always cover "training a model." |
| 2 | **Data handling classification** | IT Security | Geotagged images of grid assets may count as CEII or internal-sensitive. That decides where photos and labels can live. It's the main reason for self-hosted labeling tools (§3). |
| 3 | **Ultralytics AGPL-3.0 review** | Legal | YOLOv8 is AGPL. Internal use is generally fine. Any closed-source distribution or network-served product needs review or a commercial license. |
| 4 | **Pick the first-release scope** | Engineering + Ops | Don't try all 34 defect classes at once. See §2. |
| 5 | **Storage and compute location** | IT | Raw photos plus labels will be tens of GB. Training runs on a Mac mini M4 (~14 h for a detector) or any machine with a GPU. |

---

## 2. Scope the first release

The taxonomy in `grid_cv/config.py` has **9 component classes** and **34
defect classes**. Most defects are rare, so a classifier with 34 classes and
a handful of examples each will not learn. Recommendation:

**Release 1: components (Stage 1) plus about 6 high-value defects (Stage 2)**

- Components: all 9 (`insulator`, `splice`, `arrester`, `crossarm`,
  `transformer`, `pole`, `conductor`, `guy_wire`, `hardware`). Every photo
  contains these, so they're cheap to collect.
- Defects: pick the ones that are **(a)** CRITICAL or MEDIUM severity and
  **(b)** common in our photo archive. Likely candidates:
  `insulator_flashover`, `insulator_chipped_shed`, `crossarm_rot` /
  `crossarm_crack`, `transformer_oil_leak`, `vegetation_contact`,
  `pole_lean`, plus `healthy`.
- Leave the thermal-only classes (`transformer_overheating`,
  `splice_overheating`) for a later release unless we have radiometric
  thermal imagery.

**Rough data targets** (these are rules of thumb, not guarantees):

| What | Starting target |
|------|-----------------|
| Labeled photos for the detector | 1,000–2,000 photos |
| Boxes per component class | 500+ (rare ones like `arrester` will lag) |
| Examples per defect class | 150+ to start, 500+ to be reliable |
| `healthy` examples | Plentiful; cap at about 3× the largest defect class |

**Mining our archive for defects:** work orders, outage reports, and past
inspection findings point at photos that actually show defects. That is much
faster than scrolling through thousands of healthy-pole photos.

---

## 3. Labeling software

Requirements: **free**, **self-hosted or desktop** (photos stay on our
network), **exports YOLO format**, and **supports a condition attribute on
each box**.

### Recommended: CVAT (Community Edition, self-hosted)

- https://github.com/cvat-ai/cvat (MIT license)
- Runs in Docker on any internal machine and opens in a browser, so several
  labelers can work on one project.
- **Per-box attributes:** draw a box labeled `crossarm` and set
  `condition = crossarm_rot` on it. One pass produces both Stage 1 and
  Stage 2 labels (see §4).
- Built-in review/QA workflow (assign, label, review, accept), task
  splitting, and keyboard-driven labeling.
- AI-assisted labeling: once our own model is decent, it can pre-label so
  people only correct the boxes.
- Exports YOLO and COCO directly.
- Downside: needs Docker and someone to keep the server up.

### Good alternative: Label Studio (Community Edition)

- https://github.com/HumanSignal/label-studio (Apache-2.0)
- `pip install label-studio` and it's running, so the setup is easier than CVAT.
- Supports per-region choices (box plus condition dropdown), YOLO export,
  and multiple users.
- Its review and QA features are lighter than CVAT's. Some team features are
  Enterprise-only.

### One person on a laptop: X-AnyLabeling

- https://github.com/CVHub520/X-AnyLabeling (GPL-3.0)
- Desktop app with no server. Has Segment-Anything-assisted boxes and YOLO
  export.
- Fine for a pilot of a few hundred photos. It has no multi-user workflow.

### Avoid

| Tool | Why |
|------|-----|
| **Roboflow free tier** | Free-tier projects are **public**, so our grid photos would be published. |
| **Any cloud-hosted labeling SaaS** (including CVAT.ai cloud) | Photos leave our network. Needs IT Security sign-off from §1 first. |
| **LabelImg** | Archived and unmaintained. |

---

## 4. Labeling guidelines

Use **one labeling pass** with boxes and attributes:

1. Draw a **tight box around every visible component** in the photo, using
   one of the 9 component classes. Label every instance, not just the
   interesting one.
2. On each box, set the **`condition`** attribute to `healthy` or one of the
   Release 1 defect classes.
3. Skip components that are too small to judge (under about 20 px) or more
   than about 70% occluded. Mark the image `review` if you're unsure.
4. When a component has two defects, pick the **more severe** one, using the
   `SEVERITY_MAP` order: CRITICAL > MEDIUM > LOW.

Quality control:

- Write a **one-page visual guide** with an example photo for each class,
  especially for look-alike pairs (contamination vs. flashover, rot vs. crack).
- Have **two people label the same ~50 photos** early on, compare results,
  and fix the guide wherever they disagree.
- Reviewers spot-check at least 10% of each labeler's work.

Keep the **original, full-resolution JPEGs**. Don't resize, screenshot, or
re-save them. The pipeline reads GPS from EXIF and DJI XMP, and SAHI slicing
depends on full resolution.

---

## 5. Train/validation/test split

Split by **location or flight, not by photo**. Photos of the same pole from
one flight are near-duplicates. If they land on both sides of the split, the
validation scores will look great and mean nothing.

- **Train / val:** split by feeder or circuit (the code's `stable_val_split`
  takes a group key for exactly this).
- **Test set:** hold back a few feeders that **nobody looks at during
  development**. Report final accuracy only on those.

---

## 6. Plugging it into the code

Once labels are exported:

1. Put the data under `datasets/internal/`:
   ```
   datasets/internal/
     detector/images/{train,val}/*.jpg
     detector/labels/{train,val}/*.txt      # YOLO export from CVAT
     classifier/{train,val}/<defect_class>/*.jpg   # crops, see below
     provenance.csv
   ```
2. Register the source in `scripts/data_sources.py` with
   `commercial_ok=True`.
3. **Still to build:** an import script that reads the CVAT export, writes
   the YOLO labels, cuts each box (plus 15% padding, matching inference)
   into a classifier crop sorted by its `condition` attribute, assigns the
   feeder-based split, and writes `provenance.csv`. Once the label schema
   is final, this takes about half a day.
4. Build and train **from base weights**:
   ```bash
   python scripts/build_training_set.py --name prod --commercial-only
   python scripts/train.py detector   --data datasets/build/prod/detector.yaml
   python scripts/train.py classifier --data datasets/build/prod/classifier
   ```
5. Check `datasets/build/prod/provenance.csv`. Every row should be
   `source=internal`, `commercial_ok=true`.

---

## 7. Suggested sequence

1. Get the approvals in §1 (items 1–2 block everything else).
2. Choose the Release 1 defect list (§2).
3. Stand up CVAT internally and load a **pilot batch of ~200 photos**.
4. Label the pilot, measure agreement between labelers, and fix the guide.
5. Build the import script, then train a pilot model as an end-to-end check.
6. Scale labeling to the §2 targets, using the pilot model to pre-label.
7. Train Release 1 and evaluate it on the held-out feeders.
