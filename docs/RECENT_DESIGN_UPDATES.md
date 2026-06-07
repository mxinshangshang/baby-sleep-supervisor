# Recent Design Updates: Baby Sleep Supervisor

This document summarizes the latest engineering changes to the baby sleep monitoring pipeline. The focus of this update is making the system more useful in real crib scenes: side-facing baby, partial face visibility, adult hands entering the frame, blanket/coverage ambiguity, and Raspberry Pi performance constraints.

## Goals

- Keep the live preview responsive while lowering heavy model inference load.
- Improve side-face and top-view face handling.
- Make occlusion detection safer: do not treat `FaceMesh unavailable` as `airway clear`.
- Reduce false positives from adult hands/arms and blanket texture.
- Make crying/distress detection pay more attention to mouth opening and mouth motion.
- Add a baby-centric topology layer to reduce head/leg confusion in crib views.

## Runtime Pipeline

### Before

The inference client received frames and ran heavy detection in the same loop. When algorithms slowed down, preview and camera ingestion slowed down as well.

### Now

`inference_client.py` separates the flow into three responsibilities:

1. `LatestFrameReceiver`
   - Continuously drains the camera TCP stream.
   - Keeps only the latest decoded frame.
   - Prevents socket back-pressure from slowing `camera_server.py`.

2. `LatestInferenceWorker`
   - Runs `SleepSupervisor.process_frame()` on the latest frame at the configured detection FPS.
   - Applies thermal throttling.
   - Exposes the latest completed result to the preview thread.

3. Main preview loop
   - Renders the latest camera frame using cached/latest detection results.
   - Does not block on heavy inference.

Current config intent:

```yaml
inference:
  inference_fps: 3
preview:
  display_fps: 15
thermal:
  temp_warn_c: 65.0
  throttle_inference_fps: 2
```

The important distinction is: **algorithm FPS can be reduced without intentionally reducing camera capture or preview flow**.

## Face and Side-Face Handling

### Problem

MediaPipe Face Detection / FaceMesh can fail in crib top-view and side-face scenes even when a human clearly sees the baby's face.

### Changes

`src/vision/face_detector.py` now uses multiple strategies:

- Near and far BlazeFace models for face boxes.
- Full-frame FaceMesh first.
- If full-frame FaceMesh fails, use pose-derived `head_bbox`:
  - crop head ROI,
  - expand and upscale it,
  - retry FaceMesh with several rotations: `0, ±20, ±35, ±50` degrees,
  - map landmarks back to the original frame.

`classify_face_orientation()` classifies:

- `front`
- `slight_side`
- `side`

using FaceMesh geometry, with yaw ratio and direction.

UI examples:

```text
Face: Front face mesh
Face: Slight side mesh/right
Face: Side face mesh/right
Face yaw: 0.46
```

## Baby Topology Layer

### Problem

In a crib view, MediaPipe Pose can be confused by blankets, adult arms, partial baby visibility, or head/leg direction changes.

### Changes

`src/supervision.py` adds `_compute_baby_topology()`.

It builds a conservative baby-centric interpretation using:

- Face/head anchor.
- Pose head/torso/body boxes.
- Body axis from head to torso/body.
- Pose/face consistency.
- Nearby/external hands.

Output includes:

```python
{
  "posture": "side_lying" | "supine_or_face_up" | "head_visible_pose_only" | "unknown",
  "axis_angle": ...,
  "axis_confidence": ...,
  "head_pose_consistent": true/false,
  "topology_reliable": true/false,
  "nearby_hands": [...],
  "external_hand_count": ...,
}
```

UI now displays:

```text
Posture: side_lying topo=1.00
```

This layer is not a medical posture classifier yet; it is a sanity layer to avoid trusting raw pose landmarks when the topology is unstable.

## Airway and Occlusion Detection

### Problem

The earlier occlusion logic depended heavily on FaceMesh mouth/nose landmarks. If FaceMesh failed because a hand covered the face, the system could show `Occlusion: N/A` instead of risk. Conversely, broad hand/head overlap caused false positives.

### Changes

Occlusion detection now has two modes:

1. **FaceMesh available**
   - Use exact mouth/nose ROI.
   - Visual-only cues such as skin ratio, texture, and edge density are capped below danger threshold.
   - A hand must clearly overlap the exact mouth/nose ROI before raising to danger.

2. **FaceMesh unavailable but head is visible**
   - Use a conservative airway ROI from the head box.
   - Detect hands near the head ROI.
   - If hands overlap the airway ROI, report fallback occlusion risk.
   - If head exists but FaceMesh is missing, report uncertainty rather than pretending the airway is clear.

Added methods:

- `detect_hands_near_head()`
- `score_hands_against_roi()`
- `detect_head_occlusion_fallback()`

UI distinguishes:

```text
Hand nearby       # yellow; nearby but not covering airway
Airway occluder   # red; overlaps airway/mouth-nose ROI
Occlusion: 0.xx
Occlusion(fallback): 0.xx
```

## Coverage vs. Limb Exposure

### Problem

The previous `Exposure` signal was too easy to trigger from adult arms, a single exposed arm, or skin-color noise.

### Changes

The UI now presents this as `Coverage`, and the alerting logic is more conservative:

- External hand/arm nearby can downgrade coverage status.
- Topology must be reliable to escalate.
- Single-arm exposure no longer triggers a warning by itself.
- Alerts require clearer leg/body exposure and no external-hand pollution.

UI examples:

```text
Coverage: 0.50 limb_exposed
Coverage: 0.42 nearby_hand hand nearby
Coverage: N/A
```

Status bar uses `COVERAGE` instead of `EXPOSURE`.

## Crying / Visual Distress Detection

### Problem

Crying was missed in side-face or partial-face scenes because the earlier logic over-penalized side-face geometry. When FaceMesh briefly disappeared, the UI jumped to `Cry: N/A`, even during ongoing crying.

### Changes

The visual crying pipeline now prioritizes mouth behavior:

FaceMesh available:

- `mouth_aspect_ratio`
- `mouth_open_score`
- sustained mouth openness
- mouth opening variation across frames
- face orientation/yaw
- head/limb agitation

FaceMesh unavailable:

- `detect_mouth_open_fallback()` searches the lower-central head ROI for a dark open-mouth-like candidate.
- This fallback is conservative and feeds suspected crying/distress, not a definitive audio cry.

State holding:

- Recent strong crying is held for a short period (`cry_hold_s`) so temporary FaceMesh loss does not immediately show N/A.

Distress fallback:

- `_compute_distress_evidence()` combines:
  - crying score,
  - occlusion / hidden airway risk,
  - motion agitation,
  - face landmark availability.
- This allows a `cry_detected`/distress notification when the baby is present, face/airway is obscured, and motion suggests distress.

UI examples:

```text
Cry(fused): 0.61 high M0.78
Cry(recent): 0.75 3.2s
Cry(suspected): 0.58 no mesh
Cry(mouth): 0.65 M0.62
Distress: 0.72 airway_or_face_hidden_with_motion
```

Important limitation: without a microphone, this is still **visual suspected crying/distress**, not definitive acoustic crying. A future audio cry model should be treated as the primary crying evidence.

## Notifications

Current alert types are focused on:

- `cry_detected`
- `face_not_visible`
- `occlusion_detected` through the supervision flow

Recent behavior changes:

- Cry/distress can now produce `cry_detected` events when visual evidence is strong.
- Occlusion is still danger-level when airway ROI is blocked.
- Coverage/limb exposure is intentionally conservative to reduce noise.

## Known Limitations

- MediaPipe models are not trained specifically for newborn/infant crib top-view scenes.
- Visual crying can still miss crying when the mouth is fully hidden and there is little motion.
- Visual crying can still false-positive on yawning/open-mouth if no audio model is present.
- Hands/FaceMesh together are CPU-heavy on Raspberry Pi; current inference FPS is reduced for thermal safety.
- True sleep-safety semantics still need explicit soft-object/blanket/edge region modeling.

## Recommended Next Steps

1. Add audio crying detection and fuse it with visual distress.
2. Add explicit hazard regions:
   - pillow/soft object,
   - bed edge,
   - blanket-near-face zone.
3. Upgrade from basic FaceMesh geometry to MediaPipe Face Landmarker blendshapes if feasible:
   - `jawOpen`,
   - `mouthOpen`,
   - `browInnerUp`,
   - `eyeBlink`,
   - `mouthFrown`.
4. Cache successful FaceMesh ROI angle/scale to reduce CPU.
5. Add result age to the overlay so cached detection boxes are visually distinguishable from the live camera frame.
