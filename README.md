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

Version 2026-11-11. Expressive gestures demonstrated on a real NAO robot (NAO V4, NAOqi 2.1), recorded with [nao-emotion-motion](https://github.com/Monolithoid/nao-emotion-motion). Each take is a short movement performed for a prompt: an emotion, an intensity in [0, 1], and an optional event (a point the robot reacts to, given as azimuth, elevation and distance in the torso frame). The goal is to train generators (emotion + intensity + event -> gesture), not emotion recognisers.

Author: Joachim

## Content

| Group | Takes | With event | Total duration (s) | Joints |
|---|---|---|---|---|
| arms | 42 | 39 | 188.9 | LShoulderPitch, LShoulderRoll, LElbowYaw, LElbowRoll, LWristYaw, RShoulderPitch, RShoulderRoll, RElbowYaw, RElbowRoll, RWristYaw |
| head | 40 | 40 | 199.6 | HeadYaw, HeadPitch |
| hands | 7 | 7 | 33.9 | LHand, RHand |

Takes per emotion:

- arms: calme 6, colere 8, joie 7, peur 8, surprise 6, tristesse 7
- hands: calme 2, colere 1, joie 1, peur 1, surprise 1, tristesse 1
- head: calme 6, colere 7, joie 7, peur 7, surprise 7, tristesse 6

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

## Calibrations

- `nao-20261104-183718`

## License

Data: CC-BY-4.0 (https://creativecommons.org/licenses/by/4.0/). Code: MIT.

