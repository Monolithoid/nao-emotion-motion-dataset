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

Version 2026-11-18. Expressive gestures demonstrated on a real NAO robot (NAO V4, NAOqi 2.1), recorded with [nao-emotion-motion](https://github.com/Monolithoid/nao-emotion-motion). Each take is a short movement performed for a prompt: an emotion, an intensity in [0, 1], and an optional event (a point the robot reacts to, given as azimuth, elevation and distance in the torso frame). The goal is to train generators (emotion + intensity + event -> gesture), not emotion recognisers.



## Content

| Group | Takes | With event | Total duration (s) | Joints |
|---|---|---|---|---|
| arms | 168 | 57 | 801.7 | LShoulderPitch, LShoulderRoll, LElbowYaw, LElbowRoll, LWristYaw, RShoulderPitch, RShoulderRoll, RElbowYaw, RElbowRoll, RWristYaw |
| head | 78 | 60 | 382.8 | HeadYaw, HeadPitch |
| hands | 17 | 7 | 75.4 | LHand, RHand |

Takes per emotion:

- arms: calme 27, colere 28, joie 28, peur 29, surprise 28, tristesse 28
- hands: calme 3, colere 3, joie 3, peur 2, surprise 3, tristesse 3
- head: calme 13, colere 15, joie 12, peur 13, surprise 13, tristesse 12

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
- `nao-20261116-185555-takes`

## Models

`models/<group>/model.pt` (PyTorch) and `model_numpy.npz` (numpy only), last training of each group at export time:

- arms: run `arms-20261118-113406-202159`
- head: run `head-20261116-192305-033451`

## License

Data: CC-BY-4.0 (https://creativecommons.org/licenses/by/4.0/). Code: MIT.

