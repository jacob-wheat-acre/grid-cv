"""Central configuration: thresholds, slice sizes, severity mappings."""

from __future__ import annotations

from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR: Final[Path] = Path(__file__).parent.parent
MODEL_DIR: Final[Path] = ROOT_DIR / "models"
WEIGHTS_DETECTOR: Final[Path] = MODEL_DIR / "detector.pt"
WEIGHTS_CLASSIFIER: Final[Path] = MODEL_DIR / "classifier.pt"

# ---------------------------------------------------------------------------
# SAHI slice parameters
# ---------------------------------------------------------------------------
SLICE_HEIGHT: Final[int] = 640
SLICE_WIDTH: Final[int] = 640
OVERLAP_RATIO: Final[float] = 0.20   # 20% overlap in both dimensions

# ---------------------------------------------------------------------------
# Detection thresholds
# ---------------------------------------------------------------------------
DETECTOR_CONF_THRESHOLD: Final[float] = 0.30
DETECTOR_IOU_THRESHOLD: Final[float] = 0.45
CLASSIFIER_CONF_THRESHOLD: Final[float] = 0.40
BBOX_PADDING_RATIO: Final[float] = 0.15  # 15% padding around detected crop

# ---------------------------------------------------------------------------
# Component classes (Stage 1 — localization)
# Mapped to InsPLAD-det class names where available.
# ---------------------------------------------------------------------------
COMPONENT_CLASSES: Final[list[str]] = [
    "insulator",
    "splice",
    "arrester",
    "crossarm",
    "transformer",
    "pole",
    "conductor",
    "guy_wire",
    "hardware",
]

# ---------------------------------------------------------------------------
# Defect / health classes (Stage 2 — classification)
#
# 33 failure modes across poles, crossarms, insulators, transformers,
# conductors, and hardware.  "healthy" is always index 0.
#
# Grouped by component for readability; order = class index in classifier.
# ---------------------------------------------------------------------------
DEFECT_CLASSES: Final[list[str]] = [
    # ---- index 0 ----
    "healthy",

    # ---- Poles (1-7) ----
    "pole_lean",               # 1  visible tilt from vertical
    "pole_base_rot",           # 2  decay at ground line (wood)
    "woodpecker_damage",       # 3  excavated cavities in wood pole
    "pole_fire_damage",        # 4  scorch / char at base or mid-pole
    "pole_impact_damage",      # 5  vehicle strike deformation
    "concrete_spalling",       # 6  concrete pole surface delamination
    "guy_wire_damage",         # 7  broken strand, missing anchor, corroded

    # ---- Crossarms (8-13) ----
    "crossarm_rot",            # 8  wood decay, soft spots
    "crossarm_crack",          # 9  longitudinal or transverse split
    "crossarm_sag",            # 10 visible droop / deflection
    "crossarm_brace_missing",  # 11 brace absent or detached
    "crossarm_burn",           # 12 fire / arc damage
    "crossarm_corrosion",      # 13 rust on metal crossarm

    # ---- Insulators (14-19) ----
    "insulator_chipped_shed",  # 14 cracked or missing shed segment
    "insulator_broken_disc",   # 15 missing disc in a cap-and-pin string
    "insulator_flashover",     # 16 carbon tracking / arc burn on surface
    "insulator_contamination", # 17 heavy salt / pollution deposit
    "insulator_cement_spalling",# 18 cap-and-pin cement erosion
    "insulator_bird_fouling",  # 19 concentrated bird dropping buildup

    # ---- Transformers (20-24) ----
    "transformer_oil_leak",    # 20 staining / drips on tank or base
    "transformer_overheating", # 21 hotspot (thermal camera)
    "transformer_corrosion",   # 22 rust on tank body
    "transformer_bushing_damage",# 23 cracked or broken bushing
    "transformer_fin_damage",  # 24 bent / missing cooling fins

    # ---- Conductors & wire (25-28) ----
    "conductor_broken_strand", # 25 birdcaging or visible strand failure
    "conductor_sag",           # 26 clearance violation sag
    "splice_overheating",      # 27 hotspot at sleeve (thermal camera)
    "conductor_corrosion",     # 28 oxidation at dead-end clamp

    # ---- Hardware & other (29-31) ----
    "arrester_damage",         # 29 cracked housing or blown cap
    "hardware_missing",        # 30 absent cap, pin, bolt, or tie wire
    "vegetation_contact",      # 31 branch / vine touching energised parts

    # ---- Added for InsPLAD labels (32-33); appended to keep indices stable ----
    "hardware_corrosion",      # 32 rust on shackles, yokes, clamps, grips
    "bird_nest",               # 33 nest material on structure or hardware
]

# ---------------------------------------------------------------------------
# Severity mapping  (defect_class -> tier)
#
# CRITICAL  — immediate dispatch / outage risk
# MEDIUM    — schedule within inspection cycle
# LOW       — monitor / note in work order
# NONE      — healthy, no action
# ---------------------------------------------------------------------------
SEVERITY_MAP: Final[dict[str, str]] = {
    "healthy":                   "NONE",

    # Poles
    "pole_lean":                 "MEDIUM",
    "pole_base_rot":             "MEDIUM",
    "woodpecker_damage":         "LOW",
    "pole_fire_damage":          "CRITICAL",
    "pole_impact_damage":        "CRITICAL",
    "concrete_spalling":         "LOW",
    "guy_wire_damage":           "MEDIUM",

    # Crossarms
    "crossarm_rot":              "MEDIUM",
    "crossarm_crack":            "MEDIUM",
    "crossarm_sag":              "MEDIUM",
    "crossarm_brace_missing":    "MEDIUM",
    "crossarm_burn":             "CRITICAL",
    "crossarm_corrosion":        "LOW",

    # Insulators
    "insulator_chipped_shed":    "LOW",
    "insulator_broken_disc":     "MEDIUM",
    "insulator_flashover":       "CRITICAL",
    "insulator_contamination":   "LOW",
    "insulator_cement_spalling": "LOW",
    "insulator_bird_fouling":    "LOW",

    # Transformers
    "transformer_oil_leak":      "CRITICAL",
    "transformer_overheating":   "CRITICAL",
    "transformer_corrosion":     "LOW",
    "transformer_bushing_damage":"MEDIUM",
    "transformer_fin_damage":    "LOW",

    # Conductors
    "conductor_broken_strand":   "CRITICAL",
    "conductor_sag":             "MEDIUM",
    "splice_overheating":        "CRITICAL",
    "conductor_corrosion":       "LOW",

    # Hardware & other
    "arrester_damage":           "CRITICAL",
    "hardware_missing":          "LOW",
    "vegetation_contact":        "CRITICAL",
    "hardware_corrosion":        "LOW",
    "bird_nest":                 "MEDIUM",
}

# Which defect classes require a thermal (radiometric) image to detect reliably
THERMAL_REQUIRED: Final[set[str]] = {
    "transformer_overheating",
    "splice_overheating",
}

# ---------------------------------------------------------------------------
# Source-dataset label maps (used by scripts/download_*.py)
#
# Names match the actual labels in each source.  A source label missing from
# a map is dropped during conversion.
# ---------------------------------------------------------------------------

# InsPLAD supervised fault labels (Voxel51 HF build, `fault` field).
# The faulted assets are a polymer-insulator upper shackle, a glass
# insulator, a yoke suspension, a lightning-rod suspension and a vari-grip.
# "lightning rod" here is the shield-wire (para-raios) suspension: tower
# hardware, not a surge arrester.
INSPLAD_FAULT_MAP: Final[dict[str, str]] = {
    "good":        "healthy",
    "rust":        "hardware_corrosion",
    "missing_cap": "insulator_broken_disc",   # missing glass disc in a string
    "bird_nest":   "bird_nest",
}

# InsPLAD-det labels (`ground_truth` field).  InsPLAD is transmission
# imagery, so it only yields `insulator` and `hardware` boxes.
INSPLAD_DET_MAP: Final[dict[str, str]] = {
    "glass insulator":                 "insulator",
    "polymer insulator":               "insulator",
    "glass insulator big shackle":     "hardware",
    "glass insulator small shackle":   "hardware",
    "glass insulator tower shackle":   "hardware",
    "polymer insulator lower shackle": "hardware",
    "polymer insulator upper shackle": "hardware",
    "polymer insulator tower shackle": "hardware",
    "lightning rod shackle":           "hardware",
    "lightning rod suspension":        "hardware",
    "stockbridge damper":              "hardware",
    "spiral damper":                   "hardware",
    "spacer":                          "hardware",   # bundle spacer, not a splice
    "vari-grip":                       "hardware",
    "yoke":                            "hardware",
    "yoke suspension":                 "hardware",
    # dropped: "tower id plate", "sphere" (aviation marker ball)
}

# MPID: each sub-folder has a single class, and all of them are insulators.
MPID_DET_MAP: Final[dict[str, str]] = {
    "iso_poly": "insulator",   # composite_MPID
    "1":        "insulator",   # glass_MPID
    "iso-su":   "insulator",   # porcelain_MPID
}

# ---------------------------------------------------------------------------
# Visual annotation colours  (BGR for OpenCV)
# ---------------------------------------------------------------------------
SEVERITY_COLORS: Final[dict[str, tuple[int, int, int]]] = {
    "NONE":     (0,   200,   0),   # green
    "LOW":      (0,   165, 255),   # orange
    "MEDIUM":   (0,   140, 255),   # amber
    "CRITICAL": (0,     0, 220),   # red
}

SEVERITY_ORDER: Final[dict[str, int]] = {
    "NONE": 0, "LOW": 1, "MEDIUM": 2, "CRITICAL": 3
}

# ---------------------------------------------------------------------------
# API & CLI defaults
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR: Final[Path] = ROOT_DIR / "results"
API_HOST: Final[str] = "0.0.0.0"
API_PORT: Final[int] = 8000
MAX_UPLOAD_MB: Final[int] = 50
