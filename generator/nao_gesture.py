# -*- coding: utf-8 -*-
"""Expressive gesture generator for NAO: emotion + intensity + event -> validated movement.

A SINGLE standalone file, produced by `nao-motion export-generator` together with
its data file `nao_gesture.json` (the 3 arms / head / hands models, their settings
and the guardian limits). Compatible with Python 2.7 (NAOqi 2.1) and Python 3,
standard library only: no PyTorch, no numpy, no NAOqi.

To use it in another project, copy `nao_gesture.py` and `nao_gesture.json`.

    from nao_gesture import GestureGenerator
    gen = GestureGenerator.load("nao_gesture.json")
    g = gen.generate("joie", 0.7,
                     event={"azimuth_deg": 30, "elevation_deg": 0, "distance_m": 1.2},
                     current_pose=measured_posture)      # recommended, see below
    if g.ok:                                             # accepted by the guardian
        names, angles, times = g.angle_interpolation()
        motion.angleInterpolation(names, angles, times, True)
    else:
        print(g.report.summary())                        # send nothing

Inputs
  - emotion: a name from `gen.emotions`, or a mix {"joie": 0.7, "surprise": 0.3};
  - intensity: between 0 and 1;
  - event (optional): None = no event; otherwise a dictionary
    {"azimuth_deg" | "azimuth_rad", "elevation_deg" | "elevation_rad", "distance_m" (optional)}
    in the torso frame (FRAME_TORSO): azimuth positive to the robot's LEFT, elevation
    positive UP, 0 = straight ahead. A direction seen by the camera is given with
    "frame": "head" plus the HeadYaw / HeadPitch measured at detection time
    ("head_yaw_rad", "head_pitch_rad").

Output: a `Gesture` (joints, times in s, positions in rad; LHand / RHand are
openings 0..1) and the guardian report. The generator commands NOTHING and the
guardian never modifies a movement: it accepts or rejects it.

`current_pose` ({joint: measured value}): the learned gesture starts from the
calibrated start position. Given the measured posture, the generator adds the
transitions (current posture -> start -> gesture -> start, and to rest if asked)
and validates the whole, like the project UI. Without it, only the gesture is
returned and the caller must first bring the robot to the start position.
`from_rest=True`: the robot is at rest (possibly outside the movement zone, in
contact); the rest -> start segment then follows the "rest" rules.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import bisect
import io
import json
import math
import random
import struct
import sys

# ======================================================================================
# Guardian (copy of the <guardian-core> block of nao_guardian.py, same rules)
# ======================================================================================
# <guardian-core> (copied as is into nao_gesture.py by `nao-motion export-generator`)
KIND = "nao_motion.guardian_export"
SCHEMA_VERSION = 1
LEG_PREFIXES = ("LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle")
MAX_LISTED = 50

class GuardianError(ValueError):
    """Invalid guardian file or incorrect usage."""


class Violation(object):
    def __init__(self, code, message, joint=None, time_s=None, value=None, limit=None):
        self.code, self.message = code, message
        self.joint, self.time_s, self.value, self.limit = joint, time_s, value, limit

    def to_dict(self):
        d = {"code": self.code, "message": self.message}
        for k in ("joint", "time_s", "value", "limit"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        return d

    def __repr__(self):
        return "Violation(%r)" % (self.to_dict(),)


class Report(object):
    def __init__(self, ok, violations=None, stats=None):
        self.ok = bool(ok)
        self.violations = list(violations or [])
        self.stats = dict(stats or {})

    def to_dict(self):
        return {"ok": self.ok, "violations": [v.to_dict() for v in self.violations[:MAX_LISTED]],
                "n_violations": len(self.violations), "stats": self.stats}

    def summary(self):
        if self.ok:
            return "mouvement accepté"
        codes = sorted(set(v.code for v in self.violations))
        return "mouvement refusé (%d violations : %s)" % (len(self.violations), ", ".join(codes))

    def __bool__(self):
        return self.ok

    __nonzero__ = __bool__  # Python 2


# -- numeric helpers (no numpy) ----------------------------------------------------
def _finite(x):
    return not (math.isinf(x) or math.isnan(x))


def _linspace(a, b, n):
    """Like numpy.linspace: a + i * step, last point exactly b."""
    if n == 1:
        return [float(a)]
    step = (b - a) / (n - 1)
    out = [a + i * step for i in range(n)]
    out[-1] = float(b)
    return out


def _interp(x, xs, ys):
    """Like numpy.interp (xs strictly increasing)."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    i = bisect.bisect_right(xs, x)
    x0, x1 = xs[i - 1], xs[i]
    y0, y1 = ys[i - 1], ys[i]
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)


def _sample(ts, cols, tt):
    """cols: one list of values per joint -> sampled rows."""
    return [[_interp(x, ts, c) for c in cols] for x in tt]


# -- guardian ----------------------------------------------------------------------------
class Guardian(object):
    def __init__(self, data, allow_uncalibrated=False):
        if not isinstance(data, dict) or data.get("kind") != KIND:
            raise GuardianError("Ce n'est pas un fichier guardian exporté par nao-emotion-motion")
        if data.get("schema_version") != SCHEMA_VERSION:
            raise GuardianError("Version de schéma non prise en charge : %r" % (data.get("schema_version"),))
        if data.get("unit") != "rad" or data.get("time_unit") != "s":
            raise GuardianError("Unités attendues : rad et s")
        src = data.get("source") or {}
        self.calibrated = bool(src.get("calibrated"))
        if not self.calibrated and not allow_uncalibrated:
            raise GuardianError("Limites de remplacement (aucune calibration mesurée) : refusé pour un robot réel. "
                                "Exporter après une calibration, ou passer allow_uncalibrated=True pour un essai.")
        self.data = data
        self.source = src
        self.limits = {}
        for j, lh in data["limits"].items():
            lo, hi = float(lh[0]), float(lh[1])
            if not (_finite(lo) and _finite(hi) and lo < hi):
                raise GuardianError("Limite invalide pour %s" % j)
            if j.startswith(LEG_PREFIXES):
                raise GuardianError("Les jambes ne peuvent pas figurer dans le guardian : %s" % j)
            self.limits[j] = (lo, hi)
        self.margin = float(data["limit_margin_rad"])
        # optional overrides (LHand/RHand hands: opening 0..1, full range)
        self.per_joint_margin = dict((k, float(v)) for k, v in (data.get("per_joint_margin") or {}).items())
        self.per_joint_acceleration = dict((k, float(v))
                                           for k, v in (data.get("per_joint_acceleration") or {}).items())
        vel = data["velocity_rad_s"]
        self.max_velocity = float(vel["default"])
        self.per_joint_velocity = dict((k, float(v)) for k, v in (vel.get("per_joint") or {}).items())
        self.max_acceleration = float(data["max_acceleration_rad_s2"])
        self.check_rate_hz = float(data["check_rate_hz"])
        self.min_duration_s, self.max_duration_s = [float(x) for x in data["duration_s"]]
        self.start_tolerance = float(data["start_tolerance_rad"])
        self.constraints = []
        for c in data.get("coupled_constraints") or []:
            terms = dict((k, float(v)) for k, v in c["terms"].items())
            for k in terms:
                if k not in self.limits:
                    raise GuardianError("Contrainte %s : articulation inconnue %s" % (c["name"], k))
            self.constraints.append({"name": c["name"], "terms": terms, "bound": float(c["bound"])})
        self.groups = data.get("groups") or {}
        self.poses = data.get("poses") or {}
        # rest <-> start segments: rest may be outside the zone (contact); bounded by the
        # factory limits and a reduced velocity (same rules as the project guardian)
        self.rest_limits = dict((j, (float(lh[0]), float(lh[1])))
                                for j, lh in (data.get("rest_limits") or data["limits"]).items())
        self.rest_velocity = float(data.get("rest_velocity_rad_s", self.max_velocity))
        self.transition_velocity = float(data.get("transition_velocity_rad_s", self.max_velocity))

    @classmethod
    def load(cls, path, allow_uncalibrated=False):
        with io.open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f), allow_uncalibrated=allow_uncalibrated)

    # -- accessors -----------------------------------------------------------------------
    @property
    def joints(self):
        return sorted(self.limits)

    def allowed_range(self, joint):
        """[min, max] actually allowed (margin included)."""
        lo, hi = self.limits[joint]
        m = self.per_joint_margin.get(joint, self.margin)
        return lo + m, hi - m

    def velocity_limit(self, joint):
        return self.per_joint_velocity.get(joint, self.max_velocity)

    def acceleration_limit(self, joint):
        return self.per_joint_acceleration.get(joint, self.max_acceleration)

    def pose(self, name):
        """Calibrated "rest" or "start" pose: {joint: rad}."""
        return dict(self.poses.get(name) or {})

    def min_duration(self, start, target):
        """Minimum duration (s) of a linear move start -> target within the allowed velocities."""
        d = self.min_duration_s
        for j, v in target.items():
            if j in start and j in self.limits:
                d = max(d, abs(float(v) - float(start[j])) / self.velocity_limit(j))
        return d

    # -- checks -----------------------------------------------------------------
    def _joint_names(self, joints):
        v = []
        for j in joints:
            if j.startswith(LEG_PREFIXES):
                v.append(Violation("legs", "Les jambes sont hors périmètre", joint=j))
            elif j not in self.limits:
                v.append(Violation("unknown_joint", "Articulation non configurée", joint=j))
        return v

    def _coupled(self, joints, rows, tt, context, v):
        ctx = dict(context or {})
        idx = dict((j, k) for k, j in enumerate(joints))
        for c in self.constraints:
            usable = all(j in idx or j in ctx for j in c["terms"])
            if not usable:
                continue
            first, count, first_val = None, 0, None
            for i, row in enumerate(rows):
                total = 0.0
                for j, coef in c["terms"].items():
                    total += coef * (row[idx[j]] if j in idx else float(ctx[j]))
                if total > c["bound"]:
                    count += 1
                    if first is None:
                        first, first_val = i, total
            if count:
                v.append(Violation("coupled", "Contrainte couplée « %s » violée sur %d points" % (c["name"], count),
                                   time_s=None if tt is None else float(tt[first]), value=float(first_val),
                                   limit=c["bound"]))

    def check_pose(self, pose, context=None):
        """A target posture {joint: rad}: limits (margin included) and coupled constraints.
        `context`: angles of the joints that are not commanded (e.g. head held)."""
        joints = list(pose)
        if not all(hasattr(j, "startswith") for j in joints):
            return Report(False, [Violation("format", "Noms d'articulations invalides")])
        v = self._joint_names(joints)
        if v:
            return Report(False, v)
        row = []
        for j in joints:
            x = float(pose[j])
            if not _finite(x):
                return Report(False, [Violation("format", "Valeur non finie", joint=j)])
            row.append(x)
            lo, hi = self.allowed_range(j)
            if x < lo or x > hi:
                v.append(Violation("limit", "Hors limites (marge incluse)", joint=j, value=x, limit=lo if x < lo else hi))
        self._coupled(joints, [row], None, context, v)
        return Report(not v, v)

    def check_move(self, start, target, duration_s, context=None, check_duration=True):
        """Linear move from `start` to `target` in `duration_s` seconds
        (joints of `target`; `start` = measured posture). See min_duration()."""
        joints = list(target)
        missing = [j for j in joints if j not in start]
        if missing:
            return Report(False, [Violation("format", "Position de départ inconnue pour %s" % ", ".join(missing))])
        times = [0.0, float(duration_s)]
        positions = [[float(start[j]) for j in joints], [float(target[j]) for j in joints]]
        return self.check_trajectory(times, joints, positions, context=context, check_duration=check_duration)

    def check_trajectory(self, times, joints, positions, start_state=None, context=None, check_duration=True,
                         expected_joints=None):
        """times: N strictly increasing timestamps (s); joints: J names;
        positions: N rows of J angles (rad). Same check as the project's guardian."""
        try:
            ts = [float(x) for x in times]
            joints = list(joints)
            qs = [[float(x) for x in row] for row in positions]
        except (TypeError, ValueError) as e:
            return Report(False, [Violation("format", "Trajectoire invalide : %s" % e)])
        err = None
        if len(qs) != len(ts) or any(len(r) != len(joints) for r in qs):
            err = "Forme incohérente : %d horodatages, %d lignes, %d articulations" % (len(ts), len(qs), len(joints))
        elif len(ts) < 2:
            err = "Au moins deux échantillons requis"
        elif not all(_finite(x) for x in ts) or not all(_finite(x) for r in qs for x in r):
            err = "Valeurs non finies"
        elif any(b - a <= 0 for a, b in zip(ts, ts[1:])):
            err = "Horodatages non strictement croissants"
        elif not all(hasattr(j, "startswith") for j in joints):
            err = "Noms d'articulations invalides"
        elif len(set(joints)) != len(joints):
            err = "Noms d'articulations dupliqués"
        if err:
            return Report(False, [Violation("format", err)])

        v = self._joint_names(joints)
        if expected_joints is not None and set(expected_joints) != set(joints):
            v.append(Violation("incompatible", "Articulations attendues %s, reçues %s"
                               % (sorted(expected_joints), sorted(joints))))
        if v:
            return Report(False, v)

        J = len(joints)
        cols = [[r[k] for r in qs] for k in range(J)]
        dur = ts[-1] - ts[0]
        if check_duration and not (self.min_duration_s <= dur <= self.max_duration_s):
            v.append(Violation("duration", "Durée hors bornes", value=dur,
                               limit=self.max_duration_s if dur > self.max_duration_s else self.min_duration_s))

        # full path: original samples + fine grid
        n = max(2, int(math.ceil(dur * self.check_rate_hz)) + 1)
        tt = sorted(set(ts) | set(_linspace(ts[0], ts[-1], n)))
        rows = _sample(ts, cols, tt)
        for k, j in enumerate(joints):
            lo, hi = self.allowed_range(j)
            bad = [i for i, r in enumerate(rows) if r[k] < lo or r[k] > hi]
            if bad:
                i = bad[0]
                x = rows[i][k]
                v.append(Violation("limit", "Hors limites sur %d points (marge incluse)" % len(bad), joint=j,
                                   time_s=float(tt[i]), value=float(x), limit=lo if x < lo else hi))

        # velocities and accelerations of the executed piecewise-linear path
        dt = [b - a for a, b in zip(ts, ts[1:])]
        vel = [[(qs[i + 1][k] - qs[i][k]) / dt[i] for k in range(J)] for i in range(len(dt))]
        max_v = [max(abs(r[k]) for r in vel) for k in range(J)]
        for k, j in enumerate(joints):
            lim = self.velocity_limit(j)
            if max_v[k] > lim:
                i = max(range(len(vel)), key=lambda m: abs(vel[m][k]))
                v.append(Violation("velocity", "Vitesse excessive", joint=j, time_s=float(ts[i]),
                                   value=float(max_v[k]), limit=lim))
        max_a = [0.0] * J
        if len(vel) >= 2:
            acc = [[(vel[i + 1][k] - vel[i][k]) / ((dt[i + 1] + dt[i]) / 2) for k in range(J)]
                   for i in range(len(vel) - 1)]
            max_a = [max(abs(r[k]) for r in acc) for k in range(J)]
            for k, j in enumerate(joints):
                alim = self.acceleration_limit(j)
                if max_a[k] > alim:
                    i = max(range(len(acc)), key=lambda m: abs(acc[m][k]))
                    v.append(Violation("acceleration", "Accélération excessive", joint=j, time_s=float(ts[i + 1]),
                                       value=float(max_a[k]), limit=alim))

        # coupled constraints over the whole path (including head <-> arm through `context`)
        self._coupled(joints, rows, tt, context, v)

        if start_state is not None:
            for k, j in enumerate(joints):
                if j in start_state:
                    d = abs(qs[0][k] - float(start_state[j]))
                    if d > self.start_tolerance:
                        v.append(Violation("start_mismatch", "Premier point éloigné de l'état mesuré", joint=j,
                                           value=d, limit=self.start_tolerance))

        stats = {"duration_s": dur, "n_checked_points": len(tt),
                 "max_velocity": dict((j, round(max_v[k], 4)) for k, j in enumerate(joints)),
                 "max_acceleration": dict((j, round(max_a[k], 4)) for k, j in enumerate(joints))}
        if any(x.code in ("velocity", "acceleration") for x in v):
            # slow-down factor that would be enough (information only, never applied)
            sv = max(max_v[k] / self.velocity_limit(j) for k, j in enumerate(joints))
            sa = math.sqrt(max(max_a[k] / self.acceleration_limit(j) for k, j in enumerate(joints))) if max_a else 0.0
            stats["suggested_time_scale"] = round(float(max(sv, sa, 1.0)), 3)
        return Report(not v, v, stats)

    def check_rest_transition(self, times, joints, positions, start_state=None, time_offset=0.0):
        """Rest <-> start segment, the ONLY one allowed to leave the movement zone (rest in
        contact, arms on the knees...): factory limits, reduced velocity (rest_velocity_rad_s)
        and acceleration; no margin, no coupled constraints. That this path is free of
        collisions is up to the calibration (start reachable from rest)."""
        try:
            ts = [float(x) for x in times]
            joints = list(joints)
            qs = [[float(x) for x in row] for row in positions]
        except (TypeError, ValueError) as e:
            return Report(False, [Violation("format", "Invalid trajectory: %s" % e)])
        if (len(qs) != len(ts) or len(ts) < 2 or any(len(r) != len(joints) for r in qs)
                or not all(_finite(x) for x in ts) or not all(_finite(x) for r in qs for x in r)
                or any(b - a <= 0 for a, b in zip(ts, ts[1:])) or len(set(joints)) != len(joints)):
            return Report(False, [Violation("format", "Invalid trajectory")])
        for j in joints:
            if not hasattr(j, "startswith") or j.startswith(LEG_PREFIXES) or j not in self.rest_limits:
                return Report(False, [Violation("unknown_joint", "Joint not configured", joint=j)])
        v = []
        J = len(joints)
        cols = [[r[k] for r in qs] for k in range(J)]
        dur = ts[-1] - ts[0]
        n = max(2, int(math.ceil(dur * self.check_rate_hz)) + 1)
        tt = sorted(set(ts) | set(_linspace(ts[0], ts[-1], n)))
        rows = _sample(ts, cols, tt)
        for k, j in enumerate(joints):
            lo, hi = self.rest_limits[j]
            bad = [i for i, r in enumerate(rows) if r[k] < lo or r[k] > hi]
            if bad:
                x = rows[bad[0]][k]
                v.append(Violation("rest_limit", "Rest <-> start segment outside factory limits", joint=j,
                                   time_s=float(tt[bad[0]]) + time_offset, value=float(x), limit=lo if x < lo else hi))
        dt = [b - a for a, b in zip(ts, ts[1:])]
        vel = [[(qs[i + 1][k] - qs[i][k]) / dt[i] for k in range(J)] for i in range(len(dt))]
        for k, j in enumerate(joints):
            i = max(range(len(vel)), key=lambda m: abs(vel[m][k]))
            if abs(vel[i][k]) > self.rest_velocity:
                v.append(Violation("rest_velocity", "Rest <-> start segment too fast", joint=j,
                                   time_s=float(ts[i]) + time_offset, value=float(abs(vel[i][k])),
                                   limit=self.rest_velocity))
        if len(vel) >= 2:
            acc = [[(vel[i + 1][k] - vel[i][k]) / ((dt[i + 1] + dt[i]) / 2) for k in range(J)]
                   for i in range(len(vel) - 1)]
            for k, j in enumerate(joints):
                i = max(range(len(acc)), key=lambda m: abs(acc[m][k]))
                if abs(acc[i][k]) > self.acceleration_limit(j):
                    v.append(Violation("acceleration", "Excessive acceleration (rest <-> start segment)", joint=j,
                                       time_s=float(ts[i + 1]) + time_offset, value=float(abs(acc[i][k])),
                                       limit=self.acceleration_limit(j)))
        if start_state is not None:
            for k, j in enumerate(joints):
                if j in start_state and abs(qs[0][k] - float(start_state[j])) > self.start_tolerance:
                    v.append(Violation("start_mismatch", "First point far from the measured state", joint=j,
                                       value=abs(qs[0][k] - float(start_state[j])), limit=self.start_tolerance))
        return Report(not v, v, {"duration_s": dur, "rest_transition": True})

    def check_trajectory_dict(self, d, **kw):
        """nao-emotion-motion trajectory format: {"joints", "t", "q", "unit": "rad", "time_unit": "s"}."""
        if not isinstance(d, dict):
            return Report(False, [Violation("format", "Trajectoire invalide : dictionnaire attendu")])
        if d.get("unit", "rad") != "rad" or d.get("time_unit", "s") != "s":
            return Report(False, [Violation("format", "Trajectoire invalide : unités attendues rad et s")])
        try:
            return self.check_trajectory(d["t"], d["joints"], d["q"], **kw)
        except KeyError as e:
            return Report(False, [Violation("format", "Trajectoire invalide : champ manquant %s" % e)])


# ======================================================================================
# Generator
# ======================================================================================
BUNDLE_KIND = "nao_motion.gesture_bundle"
BUNDLE_SCHEMA = 1
EVENT_FEATURES = 6  # [present, sin(az), cos(az), elevation, distance known, 1/(1+d)]
COMPONENTS = ("arms", "head", "hands")
TIMINGS = ("hold", "stretch")
DEGREE = 3
# segments checked on the factory limits at reduced speed: rest and the measured posture may lie
# outside the movement zone (same rule as the workbench controller)
REST_SEGMENTS = ("rest_to_start", "to_start", "start_to_rest")


class GestureError(ValueError):
    """Invalid input (emotion, intensity, event, posture) or invalid data file."""


def _f32(x):
    """Rounds to float32, like the model inputs during training."""
    return struct.unpack(str("f"), struct.pack(str("f"), float(x)))[0]


def _matvec(W, x):
    return [sum(w * v for w, v in zip(row, x)) for row in W]


def _knots(n_ctrl):
    if n_ctrl < DEGREE + 1:
        raise GestureError("n_control_points >= 4 required")
    inner = _linspace(0.0, 1.0, n_ctrl - DEGREE + 1)
    return [0.0] * DEGREE + inner + [1.0] * DEGREE


def _bspline_row(k, n_ctrl, u):
    """One row of the clamped cubic B-spline basis (Cox-de Boor), same computation as the project."""
    u = min(1.0, max(0.0, float(u)))
    m = len(k) - 1
    if u >= 1.0:
        last = max(i for i in range(m) if k[i + 1] - k[i] != 0)
        B = [0.0] * m
        B[last] = 1.0
    else:
        B = [1.0 if k[i] <= u < k[i + 1] else 0.0 for i in range(m)]
    for d in range(1, DEGREE + 1):
        Bn = []
        for i in range(m - d):
            den1, den2 = k[i + d] - k[i], k[i + d + 1] - k[i + 1]
            a = (u - k[i]) / den1 * B[i] if den1 > 0 else 0.0
            b = (k[i + d + 1] - u) / den2 * B[i + 1] if den2 > 0 else 0.0
            Bn.append(a + b)
        B = Bn
    return B


def _head_to_torso(az_h, el_h, head_yaw, head_pitch):
    """Direction in the head frame -> torso frame (neck translation neglected)."""
    d = [math.cos(el_h) * math.cos(az_h), math.cos(el_h) * math.sin(az_h), math.sin(el_h)]
    cp, sp = math.cos(head_pitch), math.sin(head_pitch)
    d = [cp * d[0] + sp * d[2], d[1], -sp * d[0] + cp * d[2]]
    cy, sy = math.cos(head_yaw), math.sin(head_yaw)
    d = [cy * d[0] - sy * d[1], sy * d[0] + cy * d[1], d[2]]
    return math.atan2(d[1], d[0]), math.asin(max(-1.0, min(1.0, d[2])))


def event_features(event):
    """Event encoding for the models (see the module documentation)."""
    if event is None or event is False or (isinstance(event, dict) and not event.get("present", True)):
        return [0.0] * EVENT_FEATURES
    if not isinstance(event, dict):
        raise GestureError("Event: dictionary or None expected")

    def angle(name):
        if name + "_rad" in event and event[name + "_rad"] is not None:
            return float(event[name + "_rad"])
        if name + "_deg" in event and event[name + "_deg"] is not None:
            return math.radians(float(event[name + "_deg"]))
        raise GestureError("Event: %s_deg or %s_rad required" % (name, name))

    az, el = angle("azimuth"), angle("elevation")
    frame = event.get("frame", "torso")
    if frame == "head":
        if event.get("head_yaw_rad") is None or event.get("head_pitch_rad") is None:
            raise GestureError("frame='head' requires head_yaw_rad and head_pitch_rad measured at detection time")
        az, el = _head_to_torso(az, el, float(event["head_yaw_rad"]), float(event["head_pitch_rad"]))
    elif frame != "torso":
        raise GestureError("Unknown frame %r (torso | head)" % (frame,))
    if not (_finite(az) and _finite(el)) or not -math.pi / 2 <= el <= math.pi / 2:
        raise GestureError("Event: finite azimuth and elevation in [-90°, 90°] expected")
    az = math.atan2(math.sin(az), math.cos(az))
    dist = event.get("distance_m")
    if dist is not None and dist != "":
        dist = float(dist)
        if not _finite(dist) or dist <= 0:
            raise GestureError("Invalid distance: a strictly positive number of metres is expected")
        return [1.0, math.sin(az), math.cos(az), el, 1.0, 1.0 / (1.0 + dist)]
    return [1.0, math.sin(az), math.cos(az), el, 0.0, 0.0]


# -- transitions (not learned), same profiles as the project controller ------------------
def _traj(times, rows):
    return {"t": list(times), "q": [list(r) for r in rows]}


def _min_jerk(q0, q1, v_max, a_max, rate_hz=50.0, min_duration=0.4):
    """v_max / a_max: one value or one per joint."""
    J = len(q0)
    vm = v_max if isinstance(v_max, (list, tuple)) else [v_max] * J
    am = a_max if isinstance(a_max, (list, tuple)) else [a_max] * J
    T = min_duration
    for a, b, v, ac in zip(q0, q1, vm, am):
        d = abs(b - a)
        T = max(T, 1.875 * d / v, math.sqrt(5.774 * d / ac))
    n = max(3, int(math.ceil(T * rate_hz)) + 1)
    s = _linspace(0.0, 1.0, n)
    rows = []
    for x in s:
        h = 10 * x ** 3 - 15 * x ** 4 + 6 * x ** 5
        rows.append([a + h * (b - a) for a, b in zip(q0, q1)])
    return _traj([x * T for x in s], rows)


def _quintic(q0, v0, a0, q1, v1, a1, v_max, a_max, rate_hz=50.0, min_duration=0.4, max_duration=6.0,
             accept=None):
    """Transition matched in position, velocity and acceleration at both ends. v_max / a_max:
    one cap per joint. Among the durations within the caps, the shortest one that `accept`
    approves (else the shortest); when none is within the caps, the acceleration match is
    dropped, and failing that the least bad is kept. Same as
    nao_motion.control.transitions.quintic_matched."""
    z = [0.0] * len(q0)
    leasts = []
    for a_s, a_e in ((a0, a1), (z, z)):
        cands, least = _quintic_candidates(q0, v0, a_s, q1, v1, a_e, v_max, a_max, rate_hz,
                                           min_duration, max_duration)
        if cands:
            if accept is not None:
                for c in cands:
                    if accept(c):
                        return c
            return cands[0]
        leasts.append(least)
    return min(leasts, key=lambda x: x[1])[0]


def _quintic_candidates(q0, v0, a0, q1, v1, a1, v_max, a_max, rate_hz, min_duration, max_duration):
    J = len(q0)
    # caps raised to the matched boundary values: the gesture may enter or leave faster
    vcap = [max(v_max[j], abs(v0[j]), abs(v1[j])) for j in range(J)]
    acap = [max(a_max[j], abs(a0[j]), abs(a1[j])) for j in range(J)]
    T = float(min_duration)
    ok, least, least_score = [], None, float("inf")
    while T <= max_duration + 1e-9:
        n = max(3, int(math.ceil(T * rate_hz)) + 1)
        tt = _linspace(0.0, T, n)
        T2, T3, T4, T5 = T ** 2, T ** 3, T ** 4, T ** 5
        C = []
        for j in range(J):
            C.append([q0[j], v0[j], a0[j] / 2,
                      (20 * (q1[j] - q0[j]) - (8 * v1[j] + 12 * v0[j]) * T - (3 * a0[j] - a1[j]) * T2) / (2 * T3),
                      (30 * (q0[j] - q1[j]) + (14 * v1[j] + 16 * v0[j]) * T + (3 * a0[j] - 2 * a1[j]) * T2) / (2 * T4),
                      (12 * (q1[j] - q0[j]) - (6 * v1[j] + 6 * v0[j]) * T - (a0[j] - a1[j]) * T2) / (2 * T5)])
        rows, score = [], 0.0
        for t in tt:
            row = []
            for j, c in enumerate(C):
                row.append(sum(c[k] * t ** k for k in range(6)))
                score = max(score, abs(sum(k * c[k] * t ** (k - 1) for k in range(1, 6))) / vcap[j],
                            abs(sum(k * (k - 1) * c[k] * t ** (k - 2) for k in range(2, 6))) / acap[j])
            rows.append(row)
        tr = _traj(tt, rows)
        if score <= 1.0:
            ok.append(tr)
        if score < least_score - 1e-9:
            least, least_score = tr, score
        T *= 1.15
    return ok, (least, least_score)


def _ease(tr, ease_in, ease_out):
    """Same path, retimed to start and/or end at rest (time warp t = Tw(u^3 - u^4/2)).
    Same as nao_motion.control.transitions.ease."""
    t0, ts, q = tr["t"][0], tr["t"], tr["q"]
    h = ts[1] - ts[0]
    D = ts[-1] - t0
    ti = math.ceil(ease_in / h - 1e-9) * h if ease_in > 0 else 0.0
    to = math.ceil(ease_out / h - 1e-9) * h if ease_out > 0 else 0.0
    N = D + ti + to
    taus = _linspace(0.0, N, int(round(N / h)) + 1)
    tt = []
    for tau in taus:
        x = tau - ti
        if ti > 0 and tau < 2 * ti:
            u = tau / (2 * ti)
            x = 2 * ti * (u ** 3 - u ** 4 / 2)
        if to > 0 and tau > N - 2 * to:
            u = (N - tau) / (2 * to)
            x = D - 2 * to * (u ** 3 - u ** 4 / 2)
        tt.append(t0 + min(max(x, 0.0), D))
    cols = [[r[k] for r in q] for k in range(len(q[0]))]
    return _traj(taus, _sample(ts, cols, tt))


def _ease_duration(v, a_max, min_s=0.15):
    return max([min_s] + [1.5 * abs(x) / a for x, a in zip(v, a_max)])


def _boundary(tr, at_end=False):
    """Velocity and acceleration (finite differences) at the start or the end."""
    t, q = tr["t"], tr["q"]
    if at_end:
        t, q = t[::-1], q[::-1]
    J = len(q[0])
    if len(t) < 3:
        return [0.0] * J, [0.0] * J
    dt1, dt2 = t[1] - t[0], t[2] - t[1]
    v1 = [(q[1][j] - q[0][j]) / dt1 for j in range(J)]
    v2 = [(q[2][j] - q[1][j]) / dt2 for j in range(J)]
    a = [(v2[j] - v1[j]) / ((dt1 + dt2) / 2) for j in range(J)]
    return [v1[j] - a[j] * dt1 / 2 for j in range(J)], a


def _resampled(tr, rate_hz):
    t = tr["t"]
    n = max(2, int(math.ceil((t[-1] - t[0]) * rate_hz)) + 1)
    tt = _linspace(t[0], t[-1], n)
    cols = [[r[k] for r in tr["q"]] for k in range(len(tr["q"][0]))]
    return _traj(tt, _sample(t, cols, tt))


def _concat(parts):
    ts, qs, offset = [], [], 0.0
    for i, p in enumerate(parts):
        t = [x - p["t"][0] + offset for x in p["t"]]
        q = p["q"]
        if i > 0:
            t, q = t[1:], q[1:]
        ts += t
        qs += q
        offset = ts[-1]
    return _traj(ts, qs)


# -- playback adaptation (same as nao_motion/guardian/project.py and playable.py) -------------
# Before the guardian checks a generated gesture, the project makes it playable: control points
# outside the zone are moved to the nearest point inside, then the decoded gesture is brought
# back inside the limits with a smooth saturation and slowed down just enough (locally by
# default) for the velocity and acceleration limits. The guardian stays the only judge.
SOFT_BAND_RAD = 0.06
TIME_SAFETY = 1.05
WARP_WINDOW_S = 0.2
SLOW_DOWN_MODES = ("local", "uniform")


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _zone(guardian, joints):
    """(lo, hi, half-spaces [(a, b)]) of the guardian zone for one group, margins included."""
    lo = [guardian.allowed_range(j)[0] for j in joints]
    hi = [guardian.allowed_range(j)[1] for j in joints]
    half = []
    for c in guardian.constraints:
        if all(j in joints for j in c["terms"]):
            a = [0.0] * len(joints)
            for j, coef in c["terms"].items():
                a[joints.index(j)] = float(coef)
            half.append((a, float(c["bound"]) - 1e-3))
    return lo, hi, half


def _inside(x, lo, hi, half, tol=1e-9):
    return (all(v >= l - tol for v, l in zip(x, lo)) and all(v <= h + tol for v, h in zip(x, hi))
            and all(_dot(a, x) <= b + tol for a, b in half))


def _clip_box(x, lo, hi):
    return [min(h, max(l, v)) for v, l, h in zip(x, lo, hi)]


def _project_point(x, lo, hi, half, iters=200):
    """Nearest point of {lo <= y <= hi, a.y <= b} to x (Dykstra's alternating projections)."""
    x = [float(v) for v in x]
    if _inside(x, lo, hi, half):
        return list(x)
    sets = [None] + list(half)
    y = list(x)
    inc = [[0.0] * len(x) for _ in sets]
    for _ in range(iters):
        prev = list(y)
        for k, h in enumerate(sets):
            z = [a + b for a, b in zip(y, inc[k])]
            if h is None:
                p = _clip_box(z, lo, hi)
            else:
                a, b = h
                s = _dot(a, z) - b
                p = [zi - (s / _dot(a, a)) * ai for zi, ai in zip(z, a)] if s > 0 else list(z)
            inc[k] = [zi - pi for zi, pi in zip(z, p)]
            y = p
        if max(abs(a - b) for a, b in zip(y, prev)) < 1e-10:
            break
    return _clip_box(y, lo, hi)  # the box is never left, even if the half-spaces could not all be met


def _project_control_points(guardian, ctrl, joints):
    """(projected control points, max translation in rad, number of points moved)."""
    lo, hi, half = _zone(guardian, joints)
    out, moved, n = [], 0.0, 0
    for p in ctrl:
        q = _project_point(p, lo, hi, half)
        d = math.sqrt(sum((a - b) ** 2 for a, b in zip(p, q)))
        moved, n = max(moved, d), n + (d > 1e-9)
        out.append(q)
    return out, moved, n


def _soft_upper(xs, hi, band):
    """Identity below hi - band; above, compressed smoothly (C1) so that it never exceeds hi."""
    knee = hi - band
    return [knee + band * math.tanh((x - knee) / band) if x > knee else x for x in xs]


def _soft_box(xs, lo, hi, band):
    band = max(1e-4, min(band, (hi - lo) / 4))
    return [-v for v in _soft_upper([-v for v in _soft_upper(xs, hi, band)], -lo, band)]


def _gradient(t, q):
    """Like numpy.gradient(q, t, axis=0): second order inside, first order at the ends."""
    n, J = len(t), len(q[0])
    g = [[0.0] * J for _ in range(n)]
    if n < 2:
        return g
    for j in range(J):
        g[0][j] = (q[1][j] - q[0][j]) / (t[1] - t[0])
        g[-1][j] = (q[-1][j] - q[-2][j]) / (t[-1] - t[-2])
    for i in range(1, n - 1):
        hs, hd = t[i] - t[i - 1], t[i + 1] - t[i]
        den = hs * hd * (hd + hs)
        for j in range(J):
            g[i][j] = (hs * hs * q[i + 1][j] + (hd * hd - hs * hs) * q[i][j] - hd * hd * q[i - 1][j]) / den
    return g


def _cubic_resample(t, q, tt):
    """Cubic Hermite interpolation (tangents from _gradient): no kinks, so no artificial
    acceleration peaks (linear interpolation would add some)."""
    m = _gradient(t, q)
    n, out = len(t), []
    for x in tt:
        i = min(max(bisect.bisect_right(t, x) - 1, 0), n - 2)
        h = t[i + 1] - t[i]
        u = (x - t[i]) / h
        h00, h10 = 2 * u ** 3 - 3 * u ** 2 + 1, u ** 3 - 2 * u ** 2 + u
        h01, h11 = -2 * u ** 3 + 3 * u ** 2, u ** 3 - u ** 2
        out.append([h00 * a + h10 * h * ma + h01 * b + h11 * h * mb
                    for a, b, ma, mb in zip(q[i], q[i + 1], m[i], m[i + 1])])
    return out


def _grid(duration, rate_hz):
    return _linspace(0.0, duration, max(2, int(math.ceil(duration * rate_hz)) + 1))


def _velocities(t, q):
    return [[(b - a) / (t[i + 1] - t[i]) for a, b in zip(q[i], q[i + 1])] for i in range(len(t) - 1)]


def _accelerations(t, vel):
    return [[(b - a) / ((t[i + 2] - t[i]) / 2) for a, b in zip(vel[i], vel[i + 1])] for i in range(len(vel) - 1)]


def _time_scale(guardian, t, q, joints):
    if len(t) < 2:
        return 1.0
    vel = _velocities(t, q)
    s = 1.0
    for k, j in enumerate(joints):
        s = max(s, max(abs(v[k]) for v in vel) / guardian.velocity_limit(j))
    if len(vel) >= 2:
        acc = _accelerations(t, vel)
        for k, j in enumerate(joints):
            s = max(s, math.sqrt(max(abs(a[k]) for a in acc) / guardian.acceleration_limit(j)))
    return s


def _local_need(guardian, t, q, joints):
    """Slow-down needed on each interval: velocity ratio, and the square root of the
    acceleration ratio at both ends."""
    vel = _velocities(t, q)
    vlim = [guardian.velocity_limit(j) for j in joints]
    need = [max(abs(v) / l for v, l in zip(row, vlim)) for row in vel]
    if len(vel) >= 2:
        alim = [guardian.acceleration_limit(j) for j in joints]
        for i, row in enumerate(_accelerations(t, vel)):
            ra = math.sqrt(max(abs(a) / l for a, l in zip(row, alim)))
            need[i] = max(need[i], ra)
            need[i + 1] = max(need[i + 1], ra)
    return need


def _warp_rate(need, window):
    """Smooth local slow-down factor >= need (and >= 1): running max, then running mean."""
    r = [1.0 if x <= 1.0 else max(x * TIME_SAFETY, 1.0) for x in need]
    n = len(r)
    w = max(1, min(window, n))
    pad = [r[0]] * w + r + [r[-1]] * w
    mx = [max(pad[i:i + 2 * w + 1]) for i in range(n)]
    pad = [mx[0]] * w + mx + [mx[-1]] * w
    return [sum(pad[i:i + 2 * w + 1]) / (2 * w + 1) for i in range(n)]


def _stretch(t, q, s, rate_hz):
    """Same path, played s times slower (uniform)."""
    tn = _grid((t[-1] - t[0]) * s, rate_hz)
    return tn, _cubic_resample(t, q, [min(x / s, t[-1]) for x in tn])


def _local_warp(guardian, t, q, joints, rate_hz, rounds=8):
    """Slow down only where a limit is exceeded (smooth time warp, same path). None if the
    result would exceed the maximum duration."""
    w = int(round(WARP_WINDOW_S * rate_hz))
    for _ in range(rounds):
        need = _local_need(guardian, t, q, joints)
        if max(need) <= 1.0:
            break
        rate = _warp_rate(need, w)
        tw = [0.0]
        for i in range(len(t) - 1):
            tw.append(tw[-1] + (t[i + 1] - t[i]) * rate[i])
        if tw[-1] > guardian.max_duration_s:
            return None
        tn = _grid(tw[-1], rate_hz)
        q = _cubic_resample(tw, q, [min(x, tw[-1]) for x in tn])
        t = tn
    return t, q


def _shape(guardian, joints, q, context, rounds=6):
    """Joints past their bound and violated coupled constraints brought back smoothly; what
    stays inside is left exactly as the model made it."""
    q = [list(r) for r in q]
    bounds = [guardian.allowed_range(j) for j in joints]
    for _ in range(rounds):
        for c in guardian.constraints:
            terms = c["terms"]
            idx = [(joints.index(j), coef) for j, coef in terms.items() if j in joints]
            if not idx or any(j not in joints and j not in context for j in terms):
                continue
            fixed = sum(coef * float(context[j]) for j, coef in terms.items() if j not in joints)
            cvec = [coef for _, coef in idx]
            cols = [k for k, _ in idx]
            bound = c["bound"] - 1e-3
            s = [sum(r[k] * cv for k, cv in zip(cols, cvec)) + fixed for r in q]
            if not any(x > bound for x in s):
                continue
            s_new = _soft_upper(s, bound, SOFT_BAND_RAD)
            nn = _dot(cvec, cvec)
            for r, a, b in zip(q, s, s_new):
                d = b - a
                if d:
                    for k, cv in zip(cols, cvec):
                        r[k] += d * cv / nn
        for k, (lo, hi) in enumerate(bounds):
            col = [r[k] for r in q]
            if min(col) < lo or max(col) > hi:
                for r, v in zip(q, _soft_box(col, lo, hi, SOFT_BAND_RAD)):
                    r[k] = v
    return q


def _make_playable(guardian, joints, t, q, context=None, rate_hz=50.0, mode="local"):
    """Adapted copy of the gesture (joints, times, rows) + a description of the change. The
    guardian must still validate it. Same as nao_motion.guardian.playable.make_playable."""
    ctx = dict(context or {})
    t0 = [x - t[0] for x in t]
    tg = _grid(t0[-1], rate_hz)
    src = _cubic_resample(t0, q, tg)
    shaped = _shape(guardian, joints, src, ctx)
    diff = [max(abs(a[k] - b[k]) for a, b in zip(shaped, src)) for k in range(len(joints))]
    changed = [j for j, d in zip(joints, diff) if d > 1e-3]
    notes = []
    dur = tg[-1]
    bt, bq = tg, shaped
    s0 = max(1.0, guardian.min_duration_s / dur)
    if s0 > 1.0:
        bt, bq = _stretch(bt, bq, s0, rate_hz)
    local = 1.0
    if mode == "local" and _time_scale(guardian, bt, bq, joints) > 1.0:
        warped = _local_warp(guardian, bt, bq, joints, rate_hz)
        if warped is not None:
            local = warped[0][-1] / bt[-1]
            bt, bq = warped
    s, st, sq = 1.0, bt, bq
    for _ in range(6):
        need = _time_scale(guardian, st, sq, joints)
        if need <= 1.0:
            break
        s_new = s * need * TIME_SAFETY
        if bt[-1] * s_new > guardian.max_duration_s:
            s_new = max(s, guardian.max_duration_s / bt[-1])
            if s_new == s:
                break
            notes.append("slow-down capped by max duration (%s s): the guardian may refuse"
                         % guardian.max_duration_s)
            s = s_new
            st, sq = _stretch(bt, bq, s, rate_hz)
            break
        s = s_new
        st, sq = _stretch(bt, bq, s, rate_hz)
    if changed:
        notes.append("brought back inside limits / coupled zone: %s (max %.3f rad)" % (changed, max(diff)))
    if local > 1.0 + 1e-9:
        notes.append("fast moments slowed down locally: duration x%.2f" % local)
    if s > 1.0 + 1e-9:
        notes.append("slowed down uniformly x%.2f for velocity / acceleration limits" % s)
    bounds = [guardian.allowed_range(j) for j in joints]
    sq = [[min(hi, max(lo, v)) for v, (lo, hi) in zip(r, bounds)] for r in sq]
    total = st[-1] / dur
    info = {"changed": total > 1.0 + 1e-9 or max(diff) > 1e-6,
            "time_scale": round(total if total > 1.0 + 1e-9 else 1.0, 3),
            "max_shape_change_rad": round(max(diff), 4), "joints_changed": changed, "notes": notes}
    return st, sq, info


class Gesture(object):
    """Proposed movement. `ok` = accepted by the guardian; nothing has been sent to the robot."""

    def __init__(self, joints, times, positions, report, gesture_report, segments, inputs, notes, durations,
                 adaptation=None, raw_report=None):
        self.joints = list(joints)
        self.times = list(times)            # s, from 0
        self.positions = positions          # one row per time step, one column per joint
        self.report = report                # guardian report on everything returned
        self.gesture_report = gesture_report  # report on the gesture alone
        self.segments = segments            # [(name, start s, end s)]
        self.inputs = inputs
        self.notes = notes
        self.group_durations_s = durations
        self.adaptation = adaptation        # what was changed to make the gesture playable (None: off)
        self.raw_report = raw_report        # report on the raw learned gesture, before the adaptation

    @property
    def ok(self):
        return self.report.ok

    @property
    def duration(self):
        return self.times[-1] - self.times[0]

    def __bool__(self):
        return self.ok

    __nonzero__ = __bool__  # Python 2

    def angle_interpolation(self, rate_hz=None):
        """(names, angles, times) for ALMotion.angleInterpolation(names, angles, times, True).

        The first sample (t = 0, start posture) is dropped: NAOqi starts from the current
        posture. `rate_hz` thins out the points if needed (default: all of them, ~50 Hz).
        Refuses a movement the guardian rejected."""
        if not self.ok:
            raise GuardianError("Movement rejected by the guardian: " + self.report.summary())
        idx = list(range(1, len(self.times)))
        if rate_hz:
            step = max(1, int(round((len(self.times) - 1) / max(self.duration * float(rate_hz), 1.0))))
            idx = idx[step - 1::step]
            if idx[-1] != len(self.times) - 1:
                idx.append(len(self.times) - 1)
        names = list(self.joints)
        angles = [[float(self.positions[i][k]) for i in idx] for k in range(len(names))]
        times = [[float(self.times[i]) for i in idx] for _ in names]
        return names, angles, times

    def to_dict(self):
        return {"ok": self.ok, "inputs": self.inputs, "duration_s": self.duration, "notes": self.notes,
                "segments": self.segments, "group_durations_s": self.group_durations_s,
                "guardian": self.report.to_dict(), "gesture_guardian": self.gesture_report.to_dict(),
                "adaptation": self.adaptation,
                "raw_guardian": None if self.raw_report is None else self.raw_report.to_dict(),
                "trajectory": {"joints": self.joints, "unit": "rad", "time_unit": "s",
                               "t": self.times, "q": self.positions}}


class _Model(object):
    def __init__(self, name, d, n_inputs, K):
        self.name = name
        # control points per joint: the model's own (they can differ between groups), else the
        # file-wide value of exports made before this field
        K = int(d.get("n_control_points") or K)
        self.K = K
        self.knots = _knots(K)
        self.joints = list(d["joints"])
        self.center = [float(x) for x in d["center"]]
        self.scale = [float(x) for x in d["scale"]]
        self.latent_dim = int(d.get("latent_dim", 0))
        self.layers = [([[float(x) for x in row] for row in L["W"]], [float(x) for x in L["b"]])
                       for L in d["layers"]]
        self.clip = d.get("clip")
        J = len(self.joints)
        if not self.layers or len(self.layers[0][0][0]) != n_inputs + self.latent_dim:
            raise GestureError("Model %s: input does not match the emotion list" % name)
        if len(self.layers[-1][1]) != K * J + 1 or len(self.center) != J or len(self.scale) != J:
            raise GestureError("Model %s: output does not match (%d points x %d joints)" % (name, K, J))

    def predict(self, x):
        h = list(x) + [0.0] * self.latent_dim
        for i, (W, b) in enumerate(self.layers):
            h = [s + bb for s, bb in zip(_matvec(W, h), b)]
            if i < len(self.layers) - 1:
                h = [math.tanh(v) for v in h]
        return h


class GestureGenerator(object):
    def __init__(self, data, allow_uncalibrated=False):
        if not isinstance(data, dict) or data.get("kind") != BUNDLE_KIND:
            raise GestureError("Not a generator file exported by nao-emotion-motion")
        if data.get("schema_version") != BUNDLE_SCHEMA:
            raise GestureError("Unsupported schema version: %r" % (data.get("schema_version"),))
        self.data = data
        self.emotions = list(data["emotions"])
        self.K = int(data["n_control_points"])
        self.rate_hz = float(data.get("rate_hz", 50.0))
        self.timing = data.get("timing", "hold")
        if self.timing not in TIMINGS:
            raise GestureError("Unknown timing: %r" % (self.timing,))
        self.guardian = Guardian(data["guardian"], allow_uncalibrated=allow_uncalibrated)
        self.groups = dict((k, list(v)) for k, v in data["groups"].items())
        n_in = len(self.emotions) + 1 + EVENT_FEATURES
        self.models = {}
        for name in COMPONENTS:
            if (data.get("models") or {}).get(name):
                self.models[name] = _Model(name, data["models"][name], n_in, self.K)
                if self.models[name].joints != self.groups[name]:
                    raise GestureError("Model %s: joints differ from the group" % name)
        if not self.models:
            raise GestureError("No model in the file")
        self._knots = _knots(self.K)
        # playback adaptation, as the project UI does (robot.yaml execution.*); files exported
        # before this setting existed get the project defaults
        pb = data.get("playback") or {}
        self.project_control_points = bool(pb.get("project_control_points", True))
        self.adapt_generated = bool(pb.get("adapt_generated", True))
        self.slow_down = pb.get("slow_down", "local")
        if self.slow_down not in SLOW_DOWN_MODES:
            self.slow_down = "local"
        for j in self.groups["arms"] + self.groups["head"] + self.groups.get("hands", []):
            if j not in self.guardian.limits:
                raise GestureError("Joint %s missing from the guardian" % j)

    @classmethod
    def load(cls, path, allow_uncalibrated=False):
        with io.open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f), allow_uncalibrated=allow_uncalibrated)

    @property
    def info(self):
        """Provenance: export date, calibration, source runs of the models."""
        return {"exported_at": self.data.get("exported_at"), "calibration": self.guardian.source,
                "models": dict((k, (self.data["models"][k] or {}).get("source")) for k in self.models),
                "emotions": self.emotions}

    # -- inputs ----------------------------------------------------------------------
    def _emotion_vector(self, emotion):
        v = [0.0] * len(self.emotions)
        if not isinstance(emotion, dict):
            if emotion not in self.emotions:
                raise GestureError("Unknown emotion: %s (known: %s)" % (emotion, ", ".join(self.emotions)))
            v[self.emotions.index(emotion)] = 1.0
            return v
        total = 0.0
        for e, w in emotion.items():
            if e not in self.emotions:
                raise GestureError("Unknown emotion: %s" % e)
            w = float(w)
            if not _finite(w) or w < 0:
                raise GestureError("Invalid emotion weights")
            v[self.emotions.index(e)] = _f32(w)
            total += _f32(w)
        if total <= 0:
            raise GestureError("Empty emotion mix")
        return [_f32(x / _f32(total)) for x in v]

    def model_input(self, emotion, intensity, event=None):
        x = float(intensity)
        if not _finite(x) or not 0.0 <= x <= 1.0:
            raise GestureError("Intensity outside [0, 1]")
        return [_f32(v) for v in self._emotion_vector(emotion) + [x] + event_features(event)]

    # -- generation ------------------------------------------------------------------
    def _plan(self, model, x, variation, seed, project=False, notes=None):
        out = model.predict(x)
        J, K = len(model.joints), model.K
        ctrl = [out[i * J:(i + 1) * J] for i in range(K)]
        if variation > 0:
            rng = random.Random(seed)
            n = [[rng.gauss(0.0, variation) for _ in range(J)] for _ in range(K)]
            sm = [[(n[i][j] + n[i - 1][j] + n[(i + 1) % K][j]) / 3 for j in range(J)] for i in range(K)]
            sm[0], sm[-1] = [0.0] * J, [0.0] * J
            ctrl = [[c + s for c, s in zip(cr, sr)] for cr, sr in zip(ctrl, sm)]
        dur = min(self.guardian.max_duration_s, max(self.guardian.min_duration_s, math.exp(out[-1])))
        ctrl = [[c * model.scale[j] + model.center[j] for j, c in enumerate(row)] for row in ctrl]
        if project:
            # control points outside the guardian zone -> nearest point inside; by the convex
            # hull property the decoded curve then stays in the zone for these position limits
            ctrl, moved, n_moved = _project_control_points(self.guardian, ctrl, model.joints)
            if n_moved and notes is not None:
                notes.append("%s: %d control point(s) moved into the zone (max %.3f rad)"
                             % (model.name, n_moved, moved))
        if model.clip:
            lo, hi = model.clip
            ctrl = [[min(hi[j], max(lo[j], c)) for j, c in enumerate(row)] for row in ctrl]
        return ctrl, dur

    def gesture(self, emotion, intensity, event=None, variation=0.0, seed=None, project=False):
        """Gesture alone (arms + head + hands), no transitions: (joints, times, positions, info).
        project=True moves the control points that fall outside the guardian zone inside it
        (raw model output otherwise)."""
        x = self.model_input(emotion, intensity, event)
        variation = float(variation or 0.0)
        plans, proj_notes = {}, []
        for k, name in enumerate(COMPONENTS):
            if name in self.models:
                plans[name] = self._plan(self.models[name], x, variation, None if seed is None else int(seed) + k,
                                         project=project, notes=proj_notes)
        T = max(d for _, d in plans.values())
        n = max(4, int(math.ceil(T * self.rate_hz)) + 1)
        tt = _linspace(0.0, T, n)
        joints, cols, notes = [], [], list(proj_notes)
        start = self.guardian.pose("start")
        for name in COMPONENTS:
            js = self.groups.get(name) or []
            if not js:
                continue
            if name in plans:
                ctrl, dur = plans[name]
                d = T if self.timing == "stretch" else dur
                K = len(ctrl)
                rows = [_bspline_row(self.models[name].knots, K, min(1.0, max(0.0, t / d))) for t in tt]
                for j in range(len(js)):
                    cols.append([sum(B[i] * ctrl[i][j] for i in range(K)) for B in rows])
                joints += js
            elif name in ("arms", "head"):
                missing = [j for j in js if j not in start]
                if missing:
                    raise GestureError("Unknown start pose for %s" % ", ".join(missing))
                cols += [[float(start[j])] * n for j in js]
                joints += js
                notes.append("No %s model: held in the start position" % name)
            else:
                notes.append("No hands model: hands not commanded")
        rows = [[c[i] for c in cols] for i in range(n)]
        durations = dict((k, round(d, 3)) for k, (_, d) in plans.items())
        return joints, tt, rows, {"notes": notes, "durations": durations}

    def generate(self, emotion, intensity, event=None, current_pose=None, from_rest=False, return_to="start",
                 variation=0.0, seed=None, adapt=None):
        """Generates and validates. See the module documentation.

        current_pose : measured posture {joint: value}; adds the transitions.
        from_rest    : the robot is at rest (rest -> start segment under the "rest" rules).
        return_to    : "start" (end in the start position) or "rest" (go back to rest);
                       ignored without current_pose.
        variation    : small random variation (0 = learned gesture; seed `seed`).
        adapt        : make the gesture playable before the check, like the project UI
                       (control points into the zone, smooth saturation, slow-down where a
                       velocity or acceleration limit is exceeded). None = the file's setting
                       (on by default), False = the raw learned gesture."""
        if adapt is None:
            adapt = self.adapt_generated
        project = bool(adapt) and self.project_control_points
        joints, tt, rows, extra = self.gesture(emotion, intensity, event, variation, seed, project=project)
        notes = list(extra["notes"])
        g = self.guardian
        ctx = {}
        if current_pose is not None:
            current_pose = dict((k, float(v)) for k, v in current_pose.items())
            ctx = dict((j, v) for j, v in current_pose.items() if j not in joints and j in g.limits)
        adaptation = raw_report = None
        if adapt:
            raw_report = g.check_trajectory(tt, joints, rows, context=ctx)
            tt, rows, adaptation = _make_playable(g, joints, tt, rows, mode=self.slow_down)
            notes += adaptation["notes"]
        gesture_report = g.check_trajectory(tt, joints, rows, context=ctx)
        inputs = {"emotion": emotion, "intensity": float(intensity), "event": event, "variation": variation,
                  "seed": seed, "timing": self.timing}
        if current_pose is None:
            notes.append("No measured posture: bring the robot to the start position before sending this gesture")
            return Gesture(joints, tt, rows, gesture_report, gesture_report, [("gesture", 0.0, round(tt[-1], 4))],
                           inputs, notes, extra["durations"], adaptation, raw_report)
        if return_to not in ("start", "rest"):
            raise GestureError("return_to: 'start' or 'rest'")
        missing = [j for j in joints if j not in current_pose]
        if missing:
            raise GestureError("Incomplete measured posture: %s" % ", ".join(missing))
        start_p, rest_p = g.pose("start"), g.pose("rest")
        for name, p in (("start", start_p), ("rest", rest_p)):
            if any(j not in p for j in joints):
                raise GestureError("Incomplete %s pose in the file" % name)
        q_cur = [current_pose[j] for j in joints]
        start = [float(start_p[j]) for j in joints]
        rest = [float(rest_p[j]) for j in joints]
        hands = set(self.groups.get("hands") or [])
        violations = []
        if from_rest:
            far = [j for j, q, r in zip(joints, q_cur, rest)
                   if j not in hands and abs(q - r) > 2 * g.start_tolerance]
            if far:
                violations.append(Violation("not_at_rest", "Robot not measured at rest: %s" % ", ".join(far)))
        motion = _resampled(_traj(tt, rows), 50.0)
        v0, a0 = _boundary(motion)
        v1, a1 = _boundary(motion, at_end=True)
        z = [0.0] * len(joints)
        # per-joint caps (factory speeds or the takes'), not the global default
        vlim = [g.velocity_limit(j) * 0.8 for j in joints]
        alim = [g.acceleration_limit(j) * 0.8 for j in joints]

        def in_zone(seg):  # transitions must stay inside the guardian zone (limits, coupled)
            r = g.check_trajectory(seg["t"], joints, seg["q"], check_duration=False, context=ctx)
            return not any(x.code in ("limit", "coupled") for x in r.violations)
        if from_rest:
            segs = [("rest_to_start", _min_jerk(q_cur, start, g.rest_velocity * 0.8, g.max_acceleration * 0.25))]
        else:
            segs = [("to_start", _min_jerk(q_cur, start, g.rest_velocity * 0.8, g.max_acceleration * 0.25))]
        t_in = _quintic(start, z, z, motion["q"][0], v0, a0, vlim, alim, accept=in_zone)
        t_out = _quintic(motion["q"][-1], v1, a1, start, z, z, vlim, alim, accept=in_zone)
        # no velocity-matched transition inside the zone: ease the gesture in/out (same path,
        # from/to rest) and join it with a straight segment, inside the (convex) zone
        ti = 0.0 if in_zone(t_in) else _ease_duration(v0, alim)
        to = 0.0 if in_zone(t_out) else _ease_duration(v1, alim)
        if ti or to:
            scale = min(1.0, 0.6 * (motion["t"][-1] - motion["t"][0]) / (ti + to))
            motion = _ease(motion, ti * scale, to * scale)
            if ti:
                t_in = _min_jerk(start, motion["q"][0], vlim, alim)
            if to:
                t_out = _min_jerk(motion["q"][-1], start, vlim, alim)
            notes.append("Straight transition, gesture eased %s: the velocity-matched transition left the zone"
                         % " and ".join(x for x, y in (("in", ti), ("out", to)) if y))
        segs += [("start_to_gesture", t_in), ("gesture", motion), ("gesture_to_start", t_out)]
        if return_to == "rest":
            segs.append(("start_to_rest", _min_jerk(start, rest, g.rest_velocity * 0.8,
                                                    g.max_acceleration * 0.25)))
        # validation: rest segments on their own, everything else on the continuous path it forms
        t, reports, main, main_t0, segments = 0.0, [], [], None, []
        for name, s in segs:
            d = s["t"][-1] - s["t"][0]
            segments.append((name, round(t, 4), round(t + d, 4)))
            if name in REST_SEGMENTS:
                reports.append(g.check_rest_transition(s["t"], joints, s["q"],
                                                       start_state=current_pose if t == 0 else None, time_offset=t))
            else:
                main_t0 = t if main_t0 is None else main_t0
                main.append(s)
            t += d
        m = _concat(main)
        r = g.check_trajectory(m["t"], joints, m["q"], check_duration=False, context=ctx,
                               start_state=current_pose if main_t0 == 0 else None)
        for x in r.violations:
            if x.time_s is not None:
                x.time_s += main_t0
        reports.insert(0, r)
        violations += gesture_report.violations
        for rep in reports:
            violations += rep.violations
        full = _concat([s for _, s in segs])
        notes.append("Rest <-> start segments: factory limits and reduced velocity only; that they are free of "
                     "collisions is up to the calibration.")
        report = Report(not violations, violations, reports[0].stats)
        return Gesture(joints, full["t"], full["q"], report, gesture_report, segments, inputs, notes,
                       extra["durations"], adaptation, raw_report)


def _main(argv=None):
    """python nao_gesture.py nao_gesture.json joie 0.7 [azimuth_deg elevation_deg [distance_m]]"""
    argv = list(sys.argv[1:] if argv is None else argv)
    allow = "--allow-uncalibrated" in argv
    raw = "--raw" in argv
    args = [a for a in argv if not a.startswith("--")]
    if len(args) not in (3, 5, 6):
        print("usage: python nao_gesture.py nao_gesture.json EMOTION INTENSITY [AZIMUTH_DEG ELEVATION_DEG "
              "[DISTANCE_M]] [--allow-uncalibrated] [--raw]")
        return 2
    gen = GestureGenerator.load(args[0], allow_uncalibrated=allow)
    event = None
    if len(args) >= 5:
        event = {"azimuth_deg": float(args[3]), "elevation_deg": float(args[4])}
        if len(args) == 6:
            event["distance_m"] = float(args[5])
    g = gen.generate(args[1], float(args[2]), event, adapt=False if raw else None)
    out = json.dumps({"ok": g.ok, "duration_s": round(g.duration, 3), "joints": g.joints,
                      "guardian": g.report.to_dict(), "adaptation": g.adaptation, "notes": g.notes}, indent=1, ensure_ascii=False)
    if not isinstance(out, str):  # Python 2: unicode -> bytes
        out = out.encode("utf-8")
    print(out)
    return 0 if g.ok else 1


if __name__ == "__main__":
    sys.exit(_main())
