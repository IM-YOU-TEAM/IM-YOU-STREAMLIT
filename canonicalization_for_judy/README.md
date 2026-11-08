# Shank + thigh models — 100 Hz

This is the completed **canonical100_v1** experiment. It is separate from the
earlier 60 Hz model. Both models were trained from scratch on Camargo data only.
No phone recordings were used to fit weights, scalers, or select training epochs.

## Files to use

- `model.pt`: BioMAT joint-angle weights, saved input scaler, and metadata; epoch 12.
- `classifier_model.pt`: activity classifier weights, scaler, and metadata; epoch 30.
- `model.py`, `classifier.py`: matching network definitions.
- `preprocessing.py`: required 100 Hz input preparation.
- `predict.py`: CPU inference and numerical verification.
- `verification_examples.npz`: 15 processed Camargo examples and original saved predictions.

Do not load these weights into the old 60 Hz/150-sample architecture.

## Install and verify

Use Python 3.11 and a virtual environment. A GPU is not required for inference.

```sh
python -m pip install -r requirements.txt
python predict.py --verify
```

Verification compares against original saved Camargo test predictions. It does
not measure phone accuracy. Numerical tolerance is 0.001 degrees for angle
outputs and 0.00001 for classifier scores to allow CPU/GPU rounding differences.

## Test a new recording

Put these files in a session folder:

- `phone_1.csv`: RIGHT THIGH.
- `phone_2.csv`: RIGHT SHANK.
- Optional `metadata.json` with `{"start": <recording-start-in-common-clock-seconds>}`.
  Without it, output time is relative to the first overlapping sample.

```sh
python predict.py --session "path/to/session" --output "results"
```

CSV columns required:

```text
common_timestamp,ax,ay,az,gx,gy,gz,accel_units,gyro_units
```

Timestamps must be in seconds on a shared, synchronized clock. Sensor rates must
be approximately 100 Hz (the loader accepts 90–110 Hz), with strictly increasing
timestamps, finite data, at least 250 samples per sensor, and no gaps above 50 ms.
The loader rejects a 60 Hz recording; it does not silently upsample one to 100 Hz.
Acceleration must include gravity. Each file must explicitly label acceleration
as `g` or `m/s^2`, and gyro as `rad/s` or `deg/s`. Mixed or missing unit labels
are rejected. g is converted using 9.80665; degrees/s uses pi/180. rad/s is not
converted again. This API requires the column names above; other native exports
must be mapped to that schema with the correct axes and clock first.

The runner aligns both recordings to a 100 Hz grid, filters at 20 Hz, builds
250-sample windows every 50 samples, and uses the same signal-derived frame and
median-acceleration normalization as training. It then applies each checkpoint's
fixed training mean and standard deviation. No scaler is fitted to phone data.

## Input and output

Network input: float32 `[batch, 250, 12]`, SHANK then THIGH, each with acceleration
along three derived axes followed by gyro along those axes. All final inputs
are standardized and dimensionless. The verification examples already have
canonicalization applied; do not canonicalize them again or standardize twice.

BioMAT output: `[batch, 250, 3]`, hip/knee/ankle angles in degrees. The CSV retains
the original Camargo target signs. The earlier phone plot negated knee angles
to display knee flexion as positive; this runner does not change the CSV sign.

Classifier output: walking, ramp_ascent, ramp_descent, stair_ascent, stair_descent.
Softmax scores are uncalibrated and are not probabilities of correctness.

Outputs include `activity_windows.csv`, `joint_angles_experimental.csv`, raw
window predictions, and `summary.json`. The gate requires each sensor to have
at least 75% of samples above 0.25 rad/s angular-speed magnitude and a principal
transverse-gyro eigenvalue fraction of at least 0.60. Raw NPZ outputs include
rejected windows: use their `usable` mask. Angle CSVs average accepted overlapping
windows; unsupported times are blank. The gate is not a standing classifier or
a validated detector of unfamiliar movements.

## What has been evaluated

Held-out Camargo classification accuracy: 95.71%.
Hip/knee/ankle RMSE: 9.24 / 7.10 / 5.26 degrees.
These lab results do not establish phone performance. The new 100 Hz phone
recording has not yet been evaluated. Compared with the earlier 60 Hz experiment,
classification improved, hip/knee errors increased, and ankle error decreased.

Training used AB06–AB21, five participant-disjoint validation folds, and 1,910
test windows from AB23/24/25. Those test participants were used in earlier
experiments, so this is not a fresh untouched cohort. See `training_results.md`
and `training_protocol.json` for exact scope and limitations. Repeating full
training/evaluation requires the Camargo dataset and original training pipeline.

This is an offline runner: forward/backward filtering and full-window processing
use future samples. It is not a zero-delay live implementation. Canonical axes
are signal-derived, not an anatomical calibration. Hip and ankle predictions
use learned relationships without a pelvis or foot sensor.

The architecture is copied from the local BioMAT port, which identifies upstream
MohsenSharifi1991/BioMAT commit 0ca003db. Dependencies match the tested environment
(torch 2.11.0+cu128 running on CPU; portable requirement torch==2.11.0).
