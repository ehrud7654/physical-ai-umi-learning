# Backend role and gate policy

## Request boundary

The API must carry semantic roles. Upload order, ZIP layout, filename, duration and `manifest.outcome` are not role classifiers.

- exactly one Mapping source: `mapping_session_id` or external Atlas fields
- at least one Demonstration session
- a session cannot be both Mapping and Demonstration
- unknown or duplicate IDs reject the request before preprocessing

## Mapping state

1. Prepare recorded camera timestamps and IMU telemetry.
2. Build the shared Atlas with `--save_map`.
3. Detect `DICT_4X4_50` ID 13 with measured black-square side `0.16 m`.
4. Compute the Atlas-to-table transform.
5. Require `PASS_ARUCO_ALIGNMENT` without post-result threshold relaxation.

Mapping `outcome` is ignored because it is not a training episode label. The fixed marker is mandatory in this state.

## Demonstration state

1. Require `manifest.outcome == "success"` for training candidacy.
2. Prepare telemetry and gap independently of the Mapping outcome.
3. Load the aligned shared Atlas with `--load_map` only. Process-local mapping is allowed, but
   `--localization_only` and `--save_map` are forbidden for demonstrations.
4. Verify that the shared Atlas SHA-256 is unchanged after every demonstration.
5. Keep attempt count and the selected complete trajectory in the episode report. Never splice pose rows across runs.

Direct ID 13 visibility in a Demonstration is not required. If measured, it is a diagnostic field only. The shared aligned Atlas supplies the table frame.

## Episode gates

- pose coverage >= 0.95
- first valid pose <= 0.5 s for shared-Atlas localization
- timestamp error <= 2 ms
- no tracking-loss interval after initialization
- final frame tracked
- gripper missing run <= 15 frames unless an explicit reviewed exception matches the measured run exactly
- pose continuity must be checked; `state=2` and `is_lost=false` are insufficient
- this package's provisional catastrophic-jump bounds are 0.25 m and 45 deg per consecutive tracked frame;
  the report records both the bound and measured maximum

An ORB process exit code of zero is not a quality pass.

## External Atlas state

An external Atlas is accepted only with:

- Atlas SHA-256
- rigid `T_slam_tag` alignment JSON with passing status when a status field exists, plus SHA-256
- ORB binary/settings/mask provenance from the receipts and bundle provenance
- the same physical scene and fixed-marker layout

If the scene or marker placement changes, reject reuse and require a new Mapping recording.

## Required report semantics

`pipeline_report.json` must record:

- explicit Mapping ID or external Atlas identity
- total Demonstrations submitted
- included and excluded counts
- per-episode outcome, ORB metrics, gap metrics and every rejection reason
- Atlas/alignment/binary/settings hashes
- whether continuity was checked and its result
- `stopped` reason when no episode passes

Do not remove failed episodes from the denominator and do not convert a warning into success.
