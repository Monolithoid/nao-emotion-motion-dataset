---
license: cc-by-4.0
task_categories:
- robotics
tags:
- nao
- emotion
- gesture-generation
- kinesthetic-teaching
- human-robot-interaction
pretty_name: nao-emotion-motion-dataset
---

# nao-emotion-motion-dataset

Version 2026-11-24. Expressive gestures demonstrated on a real NAO robot (NAO V4, NAOqi 2.1), recorded with [nao-emotion-motion](https://github.com/Monolithoid/nao-emotion-motion). Each take is a short movement performed for a prompt: an emotion, an intensity in [0, 1], and an optional event (a point the robot reacts to, given as azimuth, elevation and distance in the torso frame). The goal is to train generators (emotion + intensity + event -> gesture), not emotion recognisers.

Author: Monolithoid

## Content

| Group | Takes | With event | Total duration (s) | Joints |
|---|---|---|---|---|
| arms | 168 | 57 | 801.7 | LShoulderPitch, LShoulderRoll, LElbowYaw, LElbowRoll, LWristYaw, RShoulderPitch, RShoulderRoll, RElbowYaw, RElbowRoll, RWristYaw |
| head | 78 | 60 | 382.8 | HeadYaw, HeadPitch |
| hands | 17 | 7 | 75.4 | LHand, RHand |

Takes per emotion:

- arms: anger 28, calm 27, fear 29, joy 28, sadness 28, surprise 28
- hands: anger 3, calm 3, fear 2, joy 3, sadness 3, surprise 3
- head: anger 15, calm 13, fear 13, joy 12, sadness 12, surprise 13

Groups:

- `arms`: arms guided by hand (head held stiff)
- `head`: head guided by hand (arms held stiff)
- `hands`: hands driven by the operator from the interface (opening 0..1, not an angle)

Selection: status in ['accepted'], recorded on the real robot (no simulated take), complete recordings only. Emotion labels are working labels in French (see `configs/emotions.yaml`), not a scientific taxonomy. All takes come from a single operator and a single robot.

## Layout

```
takes/<take_id>/meta.json     prompt (emotion, intensity, event), useful segment, status, calibration
takes/<take_id>/samples.npz   t (N,) in s ; q (N, J) MEASURED positions in rad ; joints (J,)
                              hands: opening in [0, 1] ; ref_t / ref_q / ref_joints = command sent
index.csv                     one row per take
calibration/*.yaml            joint limits measured on the robot (cited by meta.calibration_version)
configs/                      emotions.yaml, joints.yaml
models/<group>/               trained models (PyTorch and numpy)
generator/                    nao_gesture.py + nao_gesture.json: the models, ready to use
dataset.json                  manifest with sha256 of every file
```

Sample rates: 50 Hz. Only the `segment` [start, end] of each take is the useful gesture. Raw samples are never resampled.

```python
import json, numpy as np
meta = json.load(open('takes/<take_id>/meta.json'))
z = np.load('takes/<take_id>/samples.npz')
t, q, joints = z['t'], z['q'], list(z['joints'])
```

To use the takes with the project: `nao-motion import-dataset <this folder>`, then `nao-motion train`.

## Ready-to-use gesture generator

`generator/` holds the trained models as two files to copy into another project, nothing to install: `nao_gesture.py` (generator + safety guardian, Python 2.7 and 3, standard library only: no PyTorch, no numpy, runs on the robot's own Python) and `nao_gesture.json` (network weights, settings, guardian limits). Models: arms (run `arms-20261120-105302-826469`), head (run `head-20261120-094747-792560`), hands (run `hands-20261120-094804-474648`). Guardian limits: calibration `nao-20261118-183149` of the robot the takes were recorded on; another NAO should be calibrated and the generator re-exported with the project (`nao-motion export-generator`).

```python
from nao_gesture import GestureGenerator
gen = GestureGenerator.load('generator/nao_gesture.json')
print(gen.emotions)
g = gen.generate('joy', 0.7,                       # emotion (or a mix {'joy': 0.7, 'surprise': 0.3}), intensity 0..1
                 event={'azimuth_deg': 30, 'elevation_deg': 0, 'distance_m': 1.2},  # optional, torso frame
                 current_pose=measured_posture)     # {joint: rad}, adds the transitions
if g.ok:                                            # accepted by the guardian
    names, angles, times = g.angle_interpolation()
    motion.angleInterpolation(names, angles, times, True)   # ALMotion, absolute angles
else:
    print(g.report.summary())                       # never send a refused gesture
```

Azimuth is positive to the robot's left, elevation positive up, 0 = straight ahead. Without `current_pose`, only the gesture is returned and the robot must be brought to the start position first. The library never commands the robot itself. Quick check from a shell: `python generator/nao_gesture.py generator/nao_gesture.json joy 0.7 30 0 1.2` (exit code 0 accepted, 1 refused). Full documentation: [Gesture generator library](https://github.com/Monolithoid/nao-emotion-motion#gesture-generator-library-export-for-another-project).

## Calibrations

- `nao-20261104-183718`
- `nao-20261116-185555-takes`

156 take(s) were recorded before the first calibration (`calibration_version` is the placeholder name of `configs/joints.yaml`): their positions are measured like the others, only no calibration file goes with them.

## Models

`models/<group>/model.pt` (PyTorch) and `model_numpy.npz` (numpy only), last training of each group at export time:

- arms: run `arms-20261120-105302-826469`
- head: run `head-20261120-094747-792560`
- hands: run `hands-20261120-094804-474648`

## License

Data: CC-BY-4.0 (https://creativecommons.org/licenses/by/4.0/). Code: MIT.

