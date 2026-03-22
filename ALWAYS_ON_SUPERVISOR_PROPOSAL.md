# Always-On Supervisor Loop Proposal

## Goal

Define a resident, event-driven runtime architecture for a cat-play device that:

- stays resident and reacts to the room,
- wakes on cat detection or a human wave gesture,
- gates higher-cost behaviors behind detections and interest signals,
- keeps the laser and narration off when the room is empty,
- and returns cleanly to low-power sleep after disengagement.

This proposal is intentionally high level. It describes the control plane that should sit above the existing model runtime and exported inference graphs.

## Design Principles

1. **Event driven, not polling-heavy.**
   The system should publish semantic events (`cat_present`, `wave_detected`, `room_empty`, `interest_dropped`) and let a supervisor state machine react to them.
2. **Detection gated.**
   High-cost behaviors should only run when lower-cost signals justify them.
3. **Stay resident.**
   A persistent camera loop and supervisor process should remain active even while the main play loop is idle.
4. **Graceful hysteresis.**
   Presence and activity must debounce across frames so the device does not flap between active and inactive states.
5. **Safe outputs by default.**
   Laser and narration should only be enabled in states that explicitly permit them.
6. **Progressive power use.**
   Empty-room conditions should reduce inference cadence and optionally lower accelerator clocks or power state.

## Resident Architecture

The proposed runtime is split into long-lived components:

### 1. Camera Ingress Loop

A persistent loop that:

- keeps the camera active at all times,
- captures frames into a ring buffer,
- publishes frame timestamps and frame handles,
- supports lower-rate sampling in idle mode and higher-rate sampling in active mode.

### 2. Watcher Detectors

Low-cost, always-available detectors that operate in the resident loop:

- cat/person presence detector,
- human wave gesture detector,
- coarse motion detector,
- optional face/body-size region classifier for occupancy confidence.

These detectors should produce events rather than directly control outputs.

### 2a. Camera Watchdog / Deadman Monitor

A first-class watchdog attached to camera ingress that:

- tracks the timestamp of the most recent valid frame,
- emits `camera_stalled` when no fresh frame arrives within the configured deadline,
- emits `camera_recovered` once valid frames resume for a short confirmation window,
- and forces a safe output state while the stream is unhealthy.

This preserves a deadman path for camera-ingress loss so the supervisor never leaves laser, narration, or active play state latched on stale presence.

### 3. Engagement Estimator

A lightweight logic block that converts recent detections into:

- `cat_presence_score`,
- `human_presence_score`,
- `cat_interest_score`,
- `wave_confidence`,
- `room_empty_score`.

It should use frame windows rather than single-frame decisions.

### 4. Supervisor State Machine

A single authoritative event consumer that:

- owns the current device state,
- performs transitions,
- controls detector cadence,
- enables/disables laser and narration,
- controls when play logic becomes active,
- requests accelerator power changes.

### 5. Play Session Controller

Active only in wake/play states. It owns:

- laser path generation,
- lure mode,
- play timers,
- cat engagement checks,
- session metrics,
- narration triggers.

### 6. Output Controllers

Separate guarded interfaces for:

- laser output,
- narration/audio,
- LEDs or status lights,
- optional premium features.

These should obey supervisor policy and never self-activate outside allowed states.

## Event Model

Suggested normalized events:

- `frame_tick`
- `camera_stalled`
- `camera_recovered`
- `cat_detected`
- `cat_lost`
- `human_detected`
- `human_lost`
- `wave_detected`
- `room_empty_confirmed`
- `room_reoccupied`
- `cat_interest_high`
- `cat_interest_low`
- `disengage_timeout_elapsed`
- `idle_timeout_elapsed`
- `cooldown_elapsed`
- `accelerator_idle_ready`
- `manual_pause`
- `manual_resume`
- `safety_stop`

Each event should carry:

- timestamp,
- confidence,
- detector source,
- frame window statistics,
- and the prior state snapshot.

## Camera Deadman Policy

The resident design must treat camera health as a first-class supervisor input.

### Watchdog rules

- Maintain `last_valid_frame_at` in the camera ingress loop.
- If no valid frame arrives for a deadline such as **2 seconds**, emit `camera_stalled`.
- Do not clear the stall immediately on a single recovered frame; require a short recovery confirmation window before emitting `camera_recovered`.

### On `camera_stalled`

The supervisor should immediately force a safe state:

- laser off,
- narration off,
- any motion outputs or rails off if present,
- play session paused or torn down,
- presence state marked stale,
- and cadence/power reduced until recovery is confirmed.

### On `camera_recovered`

- re-arm resident watch mode,
- clear stale-presence latches,
- and require normal wake validation again instead of jumping directly back to full play.

## Presence Hysteresis and Debounce

Single-frame changes should not trigger state transitions.

### Suggested hysteresis rules

#### Cat presence
- Enter-present threshold: detected in **3 of last 5** watch frames.
- Exit-present threshold: absent in **10 of last 12** watch frames.

#### Human presence
- Enter-present threshold: detected in **3 of last 5** watch frames.
- Exit-present threshold: absent in **10 of last 12** watch frames.

#### Wave gesture
- Trigger only if gesture confidence remains above threshold for **2–4 consecutive frames** or an equivalent short temporal window.

#### Room empty
- Only emit `room_empty_confirmed` if both cat and human presence scores remain below threshold for a sustained window, such as **5–10 seconds** depending on cadence.

#### Cat interest
Use motion + gaze/body-orientation heuristics over a short sliding window:
- high interest: repeated target orientation, stalking, approach, or pounce-like bursts,
- low interest: looking away, loafing, leaving the play zone, or no approach behavior for a configured period.

These windows should be state-dependent: more responsive in active play, more conservative in idle.

## State Machine

The requested high-level state flow is:

- `idle`
- `detect cat/human`
- `wake`
- `play`
- `disengage timeout`
- `sleep again`

To make it operational as a resident system, this proposal maps those concepts to explicit runtime states:

### State A: `IDLE`

Resident, low-cost watch mode.

#### Purpose
- keep camera loop alive,
- run low-rate presence and wave detection,
- keep laser off,
- keep narration off,
- minimize accelerator usage.

#### Entry actions
- laser off,
- narration off,
- switch camera and detectors to low cadence,
- optionally lower accelerator clocks or power state,
- clear transient play-only timers.

#### Exit conditions
- cat presence confirmed,
- or human wave gesture confirmed,
- or human presence confirmed if product policy allows human-only waking.

### State B: `DETECT`

Short validation state between idle and active wake.

#### Purpose
- confirm that a wake signal is real,
- avoid false wake from a single bad frame,
- decide whether activation source is cat-led or human-led.

#### Entry actions
- temporarily raise detector cadence,
- hold outputs off,
- start a short validation timer (e.g. 1–3 seconds).

#### Exit conditions
- if cat presence persists or wave remains valid -> `WAKE`,
- if human-only wake is enabled and human presence persists -> `WAKE`,
- if detections collapse -> return to `IDLE`.

### State C: `WAKE`

Bring the system into active readiness.

#### Purpose
- power up higher-cost inference path,
- initialize the play session controller,
- enable narration if the room is occupied,
- decide if laser should arm immediately.

#### Entry actions
- raise camera cadence,
- request accelerator active clocks/power,
- load or prepare active models,
- create or resume session context,
- narration on only if room is occupied,
- laser remains off until cat presence policy passes,
- start a short wake dwell timer (for example 1–3 seconds).

#### Exit conditions
- if cat present and ready -> `PLAY`,
- if no cat but human remains -> `PLAY` in lure mode or a `PLAY_PREP` behavior,
- if the original wake signal collapses before play criteria are met -> `DISENGAGE_TIMEOUT` or `IDLE`,
- if wake dwell expires without a stable play-start condition -> rollback to `DISENGAGE_TIMEOUT` or `IDLE`,
- if occupancy vanishes quickly -> `DISENGAGE_TIMEOUT` or `IDLE`.

### State D: `PLAY`

Active play and engagement state.

#### Purpose
- run the play loop,
- keep laser gated to cat presence,
- keep narration reactive while room occupied,
- measure engagement and participation.

#### Entry actions
- start play timers,
- enable laser if cat is present,
- narration on if room occupied,
- switch to active detector cadence,
- enable richer play or gesture detectors as needed.

#### Active policies
- **Turn laser on if cat is present.**
- If cat interest drops but human remains, **shift to lure mode** rather than shutting down immediately.
- If room becomes empty, **disable laser immediately** and **turn narration off**.
- If repeated no-detection frames accumulate, reduce inference cadence and consider lowering accelerator clocks.

#### Exit conditions
- cat/human absent long enough -> `DISENGAGE_TIMEOUT`,
- manual pause -> `IDLE` or safe paused mode,
- safety stop -> immediate safe idle.

### State E: `DISENGAGE_TIMEOUT`

Graceful cooldown before sleep.

#### Purpose
- avoid abrupt shutdown after short occlusions or brief disengagement,
- support fast resume if cat or human reappears,
- drain active effects cleanly.

#### Entry actions
- laser off,
- narration off if room empty,
- reduce detector cadence from full active rate to medium watch rate,
- keep session context alive,
- start disengage timer.

#### Exit conditions
- if cat or person reappears before timer expires -> `WAKE` or directly back to `PLAY`,
- if timer expires and room remains empty -> `SLEEP`/`IDLE`.

### State F: `SLEEP`

Logical low-power return state. In many implementations this can collapse back into `IDLE`.

#### Purpose
- restore low-power steady state,
- release active play resources,
- optionally unload high-cost models.

#### Entry actions
- reduce accelerator clocks or power down accelerator path if supported,
- keep only resident watch path alive,
- clear play-only state while preserving long-term session stats.

#### Exit conditions
- cat detection,
- wave gesture,
- or occupancy re-entry according to product policy.

## Transition Summary

### Required behaviors from the request

- **No cat/human for 30–60s -> drop from engaged to cooldown.**
  - In `PLAY`, if cat and human absence is sustained for the configured timeout, transition to `DISENGAGE_TIMEOUT`.

- **Cat or person reappears -> wake back to engaged.**
  - In `DISENGAGE_TIMEOUT`, any validated cat detection or wave/person reappearance transitions to `WAKE` and then `PLAY` if conditions hold.

- **Cat interest drops but person remains -> shift to lure mode.**
  - In `PLAY`, keep the session alive but switch the play controller from active chase patterns to lure/re-engagement patterns.

- **Room empty -> disable laser, reduce inference cadence.**
  - On `room_empty_confirmed`, immediately force `laser = off`, `narration = off`, and step cadence down.

- **Repeated no-detection frames -> possibly power down accelerator or lower clocks.**
  - In `DISENGAGE_TIMEOUT` and `SLEEP`, accumulate absence windows and issue power-management requests once sustained emptiness is confirmed.

- **Camera ingress stalls -> force deadman safe state.**
  - On `camera_stalled`, immediately disable laser and narration, clear stale presence, tear down active play, and transition to `IDLE` or a recovery-safe cooldown path until `camera_recovered` is validated.

## Laser Policy

The laser controller should be explicit and conservative.

### Laser allowed only when
- state is `PLAY`,
- cat presence is currently confirmed,
- no safety inhibit is active,
- and room is not marked empty.

### Laser disabled when
- entering `IDLE`, `DETECT`, `DISENGAGE_TIMEOUT`, or `SLEEP`,
- room empty is confirmed,
- cat presence falls below threshold for too long,
- safety stop or manual pause is asserted.

### Lure mode
If a person remains present but cat interest is low:
- keep session alive,
- use slower, simpler, more re-engaging motion plans,
- reduce narration density,
- drop back to cooldown if no cat re-engagement arrives within the lure window.

## Narration Policy

Narration should be occupancy and state gated.

### Narration on when
- in `WAKE` or `PLAY`,
- and room occupancy is non-empty,
- and audio policy permits it.

### Narration off when
- room empty is confirmed,
- camera stall is active,
- device is in `IDLE` or `SLEEP`,
- or disengage timeout is entered with no occupant present.

This directly implements: **turn narration off if room empty**.

## Persistent Camera Loop

The camera loop should never fully stop while the product is resident.

### Idle cadence example
- 2–5 FPS or equivalent sparse sampling,
- low-cost person/cat/wave watch path only.

### Active cadence example
- 10–30 FPS depending on model budget,
- play-state detectors and engagement scoring enabled.

### Ring buffer behavior
- keep a short rolling buffer for wake validation and quick replay,
- let the wake path inspect a few frames preceding the trigger so it can avoid blind starts.

### Camera health behavior
- update `last_valid_frame_at` on every good frame,
- emit `camera_stalled` if no good frame arrives before the watchdog deadline,
- and keep the supervisor in a safe non-playing state until `camera_recovered` is confirmed.

## Session State Machine

Inside the outer supervisor, the play controller can maintain a smaller session machine:

- `SESSION_NONE`
- `SESSION_ARMED`
- `SESSION_ACTIVE`
- `SESSION_LURE`
- `SESSION_COOLDOWN`

This inner machine should own:
- play pattern selection,
- cat interest measurement,
- narration trigger generation,
- session scoring,
- and premium 8 GB features.

The outer supervisor decides whether the inner session machine is allowed to run at all.

## Power and Performance Policy

### In idle/sleep
- reduce detector cadence,
- run only watch-path models,
- consider lowering accelerator clocks,
- optionally move some watch logic to CPU if cheaper.

### In detect/wake
- temporarily increase cadence,
- ramp accelerator clocks,
- prepare active inference graphs.

### In play
- maintain full cadence,
- run gesture and engagement models as needed,
- keep power elevated only while engagement remains valid.

### On sustained emptiness
- lower clocks or power down accelerator resources if the runtime supports it,
- keep only resident watch path alive.

## Debouncing Active/Inactive Flapping

To prevent noisy transitions:

1. **Require confirmation windows** before every active->inactive or inactive->active transition.
2. **Use asymmetric thresholds** so it is easier to stay active than to wake on a single accidental frame, and harder to go inactive on a short occlusion.
3. **Use minimum dwell times** per state.
   - Example: once `PLAY` begins, remain there for at least a short dwell unless safety requires exit.
4. **Remember the wake source.**
   - A wave-driven wake may tolerate brief cat absence while lure mode runs.
   - A cat-driven wake may require cat reappearance sooner.
5. **Separate room empty from cat interest low.**
   - Low cat interest is not the same as an empty room.

## Example Event-Driven Flow

### Flow 1: Cat-led wake
1. `IDLE`: low-rate watch path running.
2. Cat seen in 3/5 frames -> `cat_detected`.
3. Supervisor enters `DETECT` for validation.
4. Cat remains present -> `WAKE`.
5. Active cadence enabled; play controller arms.
6. Cat still present -> `PLAY`; laser on.
7. Cat disengages briefly but human still present -> switch to `SESSION_LURE` while remaining in `PLAY`.
8. No cat/human for 45 seconds -> `DISENGAGE_TIMEOUT`.
9. Still empty after timeout -> `SLEEP`/`IDLE`; laser off, narration off, cadence reduced.

### Flow 2: Human wave wake
1. `IDLE`: room not occupied enough to activate.
2. Human wave confidence exceeds threshold for validation window -> `wave_detected`.
3. Supervisor enters `DETECT`.
4. Wave remains valid -> `WAKE`.
5. Human present but cat absent -> narration may greet, laser remains off.
6. Cat appears -> transition into `PLAY`; laser on.
7. Room empties -> laser off immediately, narration off, cadence reduced.
8. Cooldown expires -> `SLEEP`/`IDLE`.

## Proposed Event Handling Pseudocode

```text
on event:
  update_presence_scores()
  update_interest_scores()

  switch supervisor_state:
    IDLE:
      if cat_present_confirmed or wave_confirmed or (human_only_wake_enabled and human_present_confirmed):
        transition(DETECT)

    DETECT:
      if validation_passed:
        transition(WAKE)
      elif validation_failed:
        transition(IDLE)

    WAKE:
      request_active_power()
      if cat_present_confirmed:
        transition(PLAY)
      elif wake_source == HUMAN_WAVE and human_present_confirmed:
        arm_lure_or_wait_for_cat()
        transition(PLAY)
      elif human_only_wake_enabled and human_present_confirmed:
        arm_lure_or_wait_for_cat()
        transition(PLAY)
      elif wake_signal_collapsed or wake_dwell_expired:
        transition(DISENGAGE_TIMEOUT)
      elif room_empty_confirmed:
        transition(DISENGAGE_TIMEOUT)

    PLAY:
      laser_enabled = cat_present_confirmed and not room_empty
      narration_enabled = not room_empty

      if cat_interest_low and human_present_confirmed:
        session_mode = LURE

      if room_empty_confirmed:
        laser_off()
        narration_off()
        reduce_cadence()
        transition(DISENGAGE_TIMEOUT)

      if no_cat_and_no_human_for_30_to_60s:
        transition(DISENGAGE_TIMEOUT)

      if camera_stalled:
        laser_off()
        narration_off()
        clear_presence_latches()
        transition(IDLE)

    DISENGAGE_TIMEOUT:
      laser_off()
      if room_empty_confirmed:
        narration_off()
      reduce_cadence_medium()

      if cat_present_confirmed or human_present_confirmed or wave_confirmed:
        transition(WAKE)
      elif timeout_elapsed:
        transition(SLEEP)

    SLEEP:
      request_low_power()
      narration_off()
      laser_off()
      transition(IDLE)
```

## Suggested Defaults

These are initial tuning targets only.

- Idle watch cadence: 2–5 FPS
- Detect validation window: 1–3 s
- Active cadence: 10–30 FPS
- Cat/human absence to cooldown: 30–60 s
- Lure mode retry window: 10–20 s
- Cooldown duration before sleep: 10–30 s
- Minimum play dwell: 5–10 s unless safety stop

## What Should Be Implemented First

1. Persistent camera loop
2. Presence event bus
3. Supervisor state machine
4. Laser and narration gating hooks
5. Hysteresis/debounce windows
6. Power cadence control
7. Session controller with lure mode
8. Premium narration/LLM integration on top

## Reference Implementation Sketch

A concrete state-machine sketch has been added under `examples/yolov5/`:

- `supervisor_state_machine.h`
- `supervisor_state_machine.cpp`

That sketch codifies the proposal into:

- explicit supervisor/session/wake/cadence enums,
- a `SupervisorConfig` with concrete timing and hysteresis parameters,
- a `SupervisorContext` carrying timestamps, windows, and current policy state,
- a `DetectionSample` input shape for each resident loop tick,
- and a `supervisor_step(...)` function that computes transitions and output gating.

The sketch is intended as the handoff point between this proposal and a future integration into the live YOLOv5 example loop.

## Non-Goals for First Cut

- Full premium narration stack
- Household memory features
- High-complexity multi-model orchestration
- Any logic that lets narration or the LLM bypass supervisor safety gating

## Summary

The proposed architecture turns the device into a resident, event-driven system that watches the room continuously but spends compute only when justified. The camera loop stays alive; low-cost detectors publish semantic events; a supervisor state machine validates wake conditions, arms active play, gates the laser and narration, applies hysteresis, manages disengagement, and returns the device to a low-power idle/sleep mode when the room goes empty.

This provides the requested stay-resident behavior while making cat presence, human wave gesture, room occupancy, and cat engagement the key gates for waking, playing, luring, cooling down, and sleeping again.
