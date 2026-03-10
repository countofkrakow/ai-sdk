# YOLOv5 Laser-Play Demo: Architecture Notes

This document explains the **current moving pieces** of the enhanced YOLOv5 demo in this folder.

> Scope: the Linux/embedded runtime that combines YOLOv5 detection, threaded inference,
> cat-target filtering, laser-dot tracking, play-behavior selection, and servo/laser GPIO control.

## 1) High-level runtime pipeline

The program in `main.c` runs two cooperating loops:

1. **Main control loop (camera + actuation)**
   - Captures camera frames.
   - Detects/stabilizes the red laser dot position.
   - Reads latest cat track produced by inference thread.
   - Chooses a behavior strategy and target point.
   - Updates servo angles and laser power state.
   - Applies safety/deadman rules.

2. **Inference worker thread**
   - Waits for latest frame from main thread.
   - Runs model pre-process + inference + post-process.
   - Publishes lightweight cat track info (`Yolov5CatTrackInfo`) back to main thread.

A shared struct protected by `pthread_mutex_t` + `pthread_cond_t` coordinates frame handoff and output track publication.

## 2) File-by-file responsibilities

### `main.c`
- Program orchestration and lifecycle.
- Hardware bring-up for PWM and MOSFET/GPIO lines.
- Inference thread creation and synchronization.
- Main control tick logic:
  - laser-dot detection + stabilization
  - cat track filtering
  - play-mode target generation
  - servo update + laser on/off policy
- Startup alignment sequence (center-line sweep pattern) for quick physical validation.
- Deadman handling and cleanup paths.

### `tracking_utils.h/.cpp`
- Laser dot computer-vision utilities:
  - HSV red masking
  - blob filtering by area/circularity
  - red-channel dominance check
- Laser observation stabilization (brief hold through short misses).
- Cat track smoothing / identity-stickiness helper (`filter_cat_track`) to reduce abrupt target switches.

### `play_algorithms.h/.cpp`
- Behavior/state machine style logic for play patterns.
- Computes next target point for the laser based on:
  - cat bbox center and confidence
  - current laser position
  - engagement heuristic over time
- Includes “retreat/avoid through waypoints” style logic to keep motion playful and less likely to run through the cat bbox.

### `servo_control.h/.cpp`
- PWM wrappers and utility conversions for angle control.
- GPIO wrappers for MOSFET-driven power toggles.
- Servo state update step that turns target-vs-current error into bounded pan/tilt adjustments.

### `yolov5_post_process.h/.cpp`
- Existing YOLOv5 decode + NMS path.
- Extended to optionally emit `Yolov5CatTrackInfo` so main control code can consume a compact “best cat” track result.

### `Makefile`
- Adds new source modules and required linking (`pthread`, `periphery`, OpenCV, etc.) for the integrated runtime.

## 3) Data contracts between modules

### `Yolov5CatTrackInfo`
A compact transport struct containing whether cat was found plus bbox + confidence, used to decouple inference from control logic.

### Laser observation structs
`LaserDotObservation` and `LaserTrackState` represent raw detection and short-term stabilization state.

### Servo/power structs
`ServoPwm`, `MosfetPowerGpio`, and `ServoState` encapsulate low-level handles and control state to keep hardware logic isolated.

## 4) Concurrency model and timing

- Main thread is the **single writer** for hardware commands (servo + laser GPIO), avoiding multi-threaded actuator races.
- Worker thread performs expensive inference and only shares small results via mutex-protected state.
- The condition variable wakes worker only when a new frame is available.
- Main loop can continue running servo/play logic even if inference is momentarily delayed, using the last filtered track.

## 5) Safety and robustness mechanisms currently present

- Servo power lines controlled through GPIO (explicit open/close lifecycle).
- Laser GPIO can be gated off during selected calm windows/conditions.
- Startup laser alignment sequence for sanity-checking orientation and travel, followed by an explicit recenter to neutral pan/tilt before control loop start.
- Laser-dot stabilization holds recent position through short detection dropouts.
- Cat track filter includes smoothing and brief identity lock behavior to reduce rapid target switching.

## 6) Runtime tuning status (important)

At the moment, **lighting-related laser CV thresholds are hard-coded** in `tracking_utils.cpp`.
Examples include HSV ranges, binary thresholding, area/circularity gates, and red-dominance checks.

So currently there is **no external config file or CLI/env tuning surface** for those thresholds in this demo.

## 7) PWM mapping for A7Z overlays

Current defaults in `main.c` are set to:

- pan: `chip=10`, `channel=1`
- tilt: `chip=10`, `channel=2`

## 8) Typical control tick (conceptual)

1. Grab frame from camera.
2. Push latest frame for worker inference (non-blocking producer style).
3. Pull latest cat track result (if available) and filter it.
4. If a cat is present, steer servos toward cat bbox center.
5. If no cat is present, hold servos at center.
6. Keep laser enabled while rails are powered (deadman may disable on camera stall).
7. Repeat.

## 9) Known limitations / next logical improvements

- Add true multi-object tracking (ID association) instead of stickiness-only lock.
- Expose laser CV thresholds as runtime tunables (env/config CLI).
- Add telemetry log modes (CSV/JSON) to aid field tuning.
- Optional stronger retreat planner if corner/collision behavior needs improvement.

---

If you are onboarding to this code, start reading in this order:
1. `main.c`
2. `tracking_utils.*`
3. `play_algorithms.*`
4. `servo_control.*`
5. `yolov5_post_process.*`


### PWM troubleshooting

If servo init fails at startup (`pwm_open` or `pwm_enable`), the program now logs detected `pwmchip*` devices and each `npwm` value from `/sys/class/pwm`. Use that output to set the correct channel mapping for your board overlay.

### MOSFET polarity troubleshooting

If servos move when MOSFETs are bypassed but stop moving when rails are switched through MOSFET boards, verify gate polarity.

- Default behavior is active-high (GPIO high => power enabled).
- If your board is active-low, set `MOSFET_ACTIVE_LOW=1`.
- You can override per rail: `PAN_POWER_ACTIVE_LOW=1`, `TILT_POWER_ACTIVE_LOW=1`, `LASER_ACTIVE_LOW=1`.

At startup the program logs the effective active-low flags so wiring/polarity assumptions are visible in runtime logs.


### Servo hardware quick test mode

Use `--test` to run a dedicated servo movement test (3 cycles, 1 second between each movement) and then exit:

- `./yolov5 <model.nbg> --test`
- Optional args still work with test mode: `./yolov5 <model.nbg> /dev/video0 100 --test`

This is useful for validating PWM + MOSFET power path behavior without running the full tracking loop.

### Minimal control flow

Servo/laser behavior is intentionally simplified: when a cat is detected, servos move toward cat bbox center; when no cat is detected, servos hold center; laser stays enabled while rails are powered.
