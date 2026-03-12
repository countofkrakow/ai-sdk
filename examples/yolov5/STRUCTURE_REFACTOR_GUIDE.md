# YOLOv5 example: structure & debugging refactor guide

This guide proposes a **non-behavioral** refactor to make the code easier to read,
navigate, and debug.

## 1) Split `main.c` into lifecycle modules

`main.c` currently mixes CLI parsing, hardware init, inference thread management,
control loop logic, and shutdown flow. Break it into dedicated files:

- `app_config.{h,cpp}`
  - parse argv and hold runtime options (camera, PWM, tuning paths, timing)
- `app_init.{h,cpp}`
  - open camera, GPIO, PWM, NPU context and return one `AppRuntime` struct
- `app_loop.{h,cpp}`
  - single-frame tick (`app_step`) and main event loop
- `app_shutdown.{h,cpp}`
  - centralized teardown and safe-laser-off behavior

Result: easier to trace startup failures and avoid repeated cleanup paths.

## 2) Introduce an explicit frame pipeline model

Use a small struct passed through each stage to avoid ad-hoc state coupling:

```text
capture -> infer -> track -> filter -> play_target -> servo_cmd -> actuate
```

Proposed structs:
- `FrameInputs` (raw frame, timestamps)
- `PerceptionState` (raw detections + active track + filtered track)
- `ControlDecision` (target point, algorithm, intent, intensity)
- `ActuationCommand` (pan/tilt command, laser duty)

This makes logs and debug dumps much more coherent.

## 3) Separate thread boundary responsibilities

Current shared state includes both frame data and perception outputs. Replace with:

- `FrameMailbox` (main -> inference)
- `InferenceMailbox` (inference -> main)

Each mailbox should carry sequence id + timestamp. Main loop can then detect stale
outputs explicitly instead of relying on boolean flags.

## 4) Add lightweight debug/trace interfaces

Create a single `debug_trace.{h,cpp}` module with:
- log levels (`ERROR/WARN/INFO/TRACE`)
- category tags (`INIT`, `INFER`, `TRACK`, `PLAY`, `SERVO`, `SAFETY`)
- optional CSV trace output for per-frame telemetry

A `FrameTraceRow` written once per tick makes algorithm behavior diagnosable without
reading mixed `fprintf` lines.

## 5) Reduce public surface area of play modules

`play_algorithms.h` currently exposes a large mutable state. Keep internals private:

- `play_engine.h` should export opaque handle + small API:
  - `play_engine_init(...)`
  - `play_engine_step(...)`
  - `play_engine_reset(...)`
- move large fields to `play_engine_internal.h`

This avoids accidental coupling from `main.c` and makes testing easier.

## 6) Co-locate tuning schema with loader code

For both tuning JSON files:
- define typed schema structs in one place
- add `validate_*_tuning()` with range checks and warnings
- emit loaded values once at startup (or on reload)

This makes bad tuning obvious and simplifies support/debug sessions.

## 7) Normalize constants and magic numbers

Adopt a consistent layout:
- all defaults in `*_defaults.h`
- all runtime overrides in tuning JSON
- all safety limits in `safety_limits.h`

Keep literals out of loop code whenever possible.

## 8) Add deterministic simulation mode

Add `--replay <frames_dir>` and `--dry-run`:
- replay frames without hardware writes
- still run tracking/play logic
- output trace rows and optional overlay video

This enables real debugging and regression checks off-device.

## 9) Clarify ownership and naming

Examples:
- rename `plant_data` to `input_tensor_bytes`
- distinguish `raw_track`, `active_track`, `filtered_track` consistently
- encode units in names: `_sec`, `_px`, `_deg`

Small naming hygiene reduces cognitive load during incident debugging.

## 10) Staged migration plan

1. Extract config + init/shutdown (no behavior changes).
2. Introduce frame pipeline structs + trace row logging.
3. Split inference mailboxes and add seq/timestamp.
4. Hide play internals behind opaque engine API.
5. Add tuning validation and dry-run replay mode.

Each stage should be independently reviewable and bisect-friendly.
