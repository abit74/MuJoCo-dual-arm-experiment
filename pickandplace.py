# -*- coding: utf-8 -*-
"""
Dual Franka Panda Pick-and-Place Control

Features:
- Two independent Panda robots in one MuJoCo scene.
- Same Tkinter control interface style for left hand, right hand, and both hands.
- No leader/follower mode: robot1 and robot2 are controlled independently.
- Robot 1 is allowed to use only red_box, yellow_box, and box.
- Robot 2 is allowed to use only green_box and blue_box.
- Workspace limits keep the two robots separated to reduce collision risk.

Run:
    py pickandplace.py
    py pickandplace.py --terminal
    py pickandplace.py --viewer-console
"""

from __future__ import annotations

import argparse
import time
import threading
from threading import Thread
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Callable, Optional

import glfw
import mujoco
import numpy as np

from nl_interface import parse_command


def _now() -> float:
    return time.time()


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class Demo:
    qpos0 = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
    K = np.array([900.0, 900.0, 900.0, 40.0, 40.0, 40.0])

    height, width = 600, 900
    fps = 30

    ctrl_hz = 400
    hold_hz = 500

    max_speed_xy = 0.30
    max_speed_z = 0.25
    min_move_dur = 0.25
    max_move_dur = 2.80

    hover_clear = 0.15
    pre_clear = 0.020
    near_clear = 0.010
    touch_clear = 0.002

    verify_lift = 0.06
    verify_min_rise = 0.018
    verify_max_dxy = 0.08
    max_grasp_attempts = 3

    align_yaw_to_object = True
    track_object_during_descend = True
    retreat_z_min = 0.22
    retreat_settle_s = 0.25

    ROBOTS = {
        "robot1": {
            "label": "Left Hand",
            "hand": "panda_hand",
            "joint_prefix": "panda_joint",
            "finger1": "pos_panda_finger_joint1",
            "finger2": "pos_panda_finger_joint2",
            "allowed": ["red_box", "yellow_box", "box"],
            "workspace": (-0.90, 0.05, -0.75, 0.45),  # xmin, xmax, ymin, ymax
            "default_place": (-0.58, 0.18, 0.25),
        },
        "robot2": {
            "label": "Right Hand",
            "hand": "panda2_panda_hand",
            "joint_prefix": "panda2_panda_joint",
            "finger1": "panda2_pos_panda_finger_joint1",
            "finger2": "panda2_pos_panda_finger_joint2",
            "allowed": ["green_box", "blue_box"],
            "workspace": (0.25, 1.10, -0.75, 0.45),
            "default_place": (0.58, 0.18, 0.25),
        },
    }

    INITIAL_BLOCKS = {
        "red_box_free": [-0.65, -0.35, 0.03, 1, 0, 0, 0],
        "yellow_box_free": [-0.45, -0.35, 0.03, 1, 0, 0, 0],
        "box_free": [-0.25, -0.35, 0.03, 1, 0, 0, 0],
        "green_box_free": [0.45, -0.35, 0.03, 1, 0, 0, 0],
        "blue_box_free": [0.65, -0.35, 0.03, 1, 0, 0, 0],
    }

    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path("world.xml")
        self.data = mujoco.MjData(self.model)

        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam.fixedcamid = 0
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)

        self.run = True
        self.stop_flag = threading.Event()
        self.motion_busy = threading.Event()
        self._hold_running = True
        self._act_id_cache: dict[str, int] = {}
        self._state_lock = threading.RLock()

        self.robot_state: dict[str, dict[str, object]] = {}
        self.held_claims: dict[str, str] = {}  # object -> robot

        self.console_input = ""
        self.console_history: list[str] = []
        self.console_status = "Ready"
        self.viewer_console_enabled = False
        self._console_lock = threading.Lock()
        self._console_busy = False

        # Reset both arms and grippers.
        for robot in self.ROBOTS:
            self.gripper(robot, True)
            for i in range(1, 8):
                self.data.joint(self._joint_name(robot, i)).qpos = self.qpos0[i - 1]
        self.reset_blocks()
        mujoco.mj_forward(self.model, self.data)

        for robot in self.ROBOTS:
            hand = self.data.body(self._hand_name(robot))
            self.robot_state[robot] = {
                "target_pos": hand.xpos.copy(),
                "target_quat": hand.xquat.copy(),
                "home_pos": hand.xpos.copy(),
                "home_quat": hand.xquat.copy(),
                "held_obj": None,
            }

    # ---------------- name helpers ----------------
    def _robot(self, robot: str) -> str:
        r = str(robot).lower().replace(" ", "")
        aliases = {"left": "robot1", "r1": "robot1", "1": "robot1", "robot1": "robot1",
                   "right": "robot2", "r2": "robot2", "2": "robot2", "robot2": "robot2"}
        r = aliases.get(r, r)
        if r not in self.ROBOTS:
            raise KeyError(f"Unknown robot '{robot}'. Use robot1/left or robot2/right.")
        return r

    def _hand_name(self, robot: str) -> str:
        return self.ROBOTS[self._robot(robot)]["hand"]  # type: ignore[return-value]

    def _joint_name(self, robot: str, i: int) -> str:
        return f"{self.ROBOTS[self._robot(robot)]['joint_prefix']}{i}"

    def _finger_act(self, robot: str, n: int) -> str:
        return self.ROBOTS[self._robot(robot)][f"finger{n}"]  # type: ignore[return-value]

    def _state(self, robot: str) -> dict[str, object]:
        return self.robot_state[self._robot(robot)]

    def _target_pos(self, robot: str) -> np.ndarray:
        return self._state(robot)["target_pos"]  # type: ignore[return-value]

    def _set_target_pos(self, robot: str, pos: np.ndarray) -> None:
        self._state(robot)["target_pos"] = pos.copy()

    def _target_quat(self, robot: str) -> np.ndarray:
        return self._state(robot)["target_quat"]  # type: ignore[return-value]

    def _set_target_quat(self, robot: str, quat: np.ndarray) -> None:
        self._state(robot)["target_quat"] = quat.copy()

    def _home_pos(self, robot: str) -> np.ndarray:
        return self._state(robot)["home_pos"]  # type: ignore[return-value]

    def _home_quat(self, robot: str) -> np.ndarray:
        return self._state(robot)["home_quat"]  # type: ignore[return-value]

    def _held_obj(self, robot: str) -> Optional[str]:
        return self._state(robot)["held_obj"]  # type: ignore[return-value]

    def _set_held_obj(self, robot: str, obj: Optional[str]) -> None:
        r = self._robot(robot)
        old = self._held_obj(r)
        if old and old in self.held_claims:
            self.held_claims.pop(old, None)
        self._state(r)["held_obj"] = obj
        if obj:
            self.held_claims[obj] = r

    # ---------------- actuator helpers ----------------
    def _act_id(self, name: str) -> int:
        if name in self._act_id_cache:
            return self._act_id_cache[name]
        aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            raise KeyError(f"Actuator '{name}' not found in model.")
        self._act_id_cache[name] = int(aid)
        return int(aid)

    def _set_act(self, name: str, value: float) -> None:
        self.data.ctrl[self._act_id(name)] = float(value)

    # ---------------- math helpers ----------------
    @staticmethod
    def _quat_conj(q: np.ndarray) -> np.ndarray:
        return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)

    @staticmethod
    def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ], dtype=float)

    def _quat_rotate_vec(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        qv = np.array([0.0, v[0], v[1], v[2]], dtype=float)
        return self._quat_mul(self._quat_mul(q, qv), self._quat_conj(q))[1:]

    def _yaw_from_quat(self, q: np.ndarray) -> float:
        vx = self._quat_rotate_vec(q, np.array([1.0, 0.0, 0.0], dtype=float))
        return float(np.arctan2(vx[1], vx[0]))

    @staticmethod
    def _wrap_pi(a: float) -> float:
        while a > np.pi:
            a -= 2*np.pi
        while a < -np.pi:
            a += 2*np.pi
        return float(a)

    @staticmethod
    def _quat_from_yaw(yaw: float) -> np.ndarray:
        half = 0.5 * yaw
        return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=float)

    @staticmethod
    def _quat_err(q: np.ndarray, r: np.ndarray) -> float:
        dot = abs(float(np.dot(q, r)))
        dot = max(min(dot, 1.0), -1.0)
        return 2.0 * np.arccos(dot)

    # ---------------- scene helpers ----------------
    def _valid_object_names(self) -> list[str]:
        out: list[str] = []
        for bid in range(self.model.nbody):
            nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if nm in {"red_box", "yellow_box", "box", "green_box", "blue_box"}:
                out.append(nm)
        return sorted(out)

    def _allowed_objects(self, robot: str) -> list[str]:
        return list(self.ROBOTS[self._robot(robot)]["allowed"])  # type: ignore[arg-type]

    def _require_allowed_body(self, robot: str, name: str) -> str:
        r = self._robot(robot)
        if name not in self._allowed_objects(r):
            raise KeyError(
                f"{self.ROBOTS[r]['label']} is not allowed to use '{name}'. "
                f"Allowed: {self._allowed_objects(r)}"
            )
        # Prevent both robots from picking/placing the same object.
        claimed_by = self.held_claims.get(name)
        if claimed_by is not None and claimed_by != r:
            raise RuntimeError(f"'{name}' is already being handled by {claimed_by}.")
        try:
            _ = self.data.body(name)
            return name
        except KeyError:
            raise KeyError(f"Invalid object '{name}'. Valid objects: {self._valid_object_names()}")

    def _body_xy(self, body_name: str) -> np.ndarray:
        return self.data.body(body_name).xpos[:2].copy()

    def _body_top_z(self, body_name: str) -> float:
        b = self.data.body(body_name)
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        z_top = float(b.xpos[2])
        for g in range(self.model.ngeom):
            if self.model.geom_bodyid[g] != bid:
                continue
            size = np.array(self.model.geom_size[g])
            gtype = int(self.model.geom_type[g])
            if gtype == mujoco.mjtGeom.mjGEOM_BOX and size.size >= 3:
                return float(b.xpos[2] + size[2])
            if gtype == mujoco.mjtGeom.mjGEOM_CYLINDER and size.size >= 2:
                return float(b.xpos[2] + size[1])
        return z_top

    def _body_yaw(self, body_name: str) -> float:
        b = self.data.body(body_name)
        R = np.array(b.xmat).reshape(3, 3)
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def _safe_xy(self, robot: str, x: float, y: float) -> tuple[float, float]:
        xmin, xmax, ymin, ymax = self.ROBOTS[self._robot(robot)]["workspace"]  # type: ignore[misc]
        return float(_clamp(x, xmin, xmax)), float(_clamp(y, ymin, ymax))

    def _safe_pos(self, robot: str, pos: np.ndarray) -> np.ndarray:
        x, y = self._safe_xy(robot, float(pos[0]), float(pos[1]))
        z = float(_clamp(float(pos[2]), 0.08, 0.65))
        return np.array([x, y, z], dtype=float)

    # ---------------- controller ----------------
    def gripper(self, robot: str, open: bool = True) -> None:
        v = 0.04 if open else 0.0
        self._set_act(self._finger_act(robot, 1), v)
        self._set_act(self._finger_act(robot, 2), v)

    def gripper_both(self, open: bool = True) -> None:
        for robot in self.ROBOTS:
            self.gripper(robot, open)

    def control(self, robot: str, xpos_d: np.ndarray, xquat_d: np.ndarray) -> None:
        hand_name = self._hand_name(robot)
        xpos = self.data.body(hand_name).xpos
        xquat = self.data.body(hand_name).xquat

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        bodyid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, hand_name)
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, bodyid)

        error = np.zeros(6)
        error[:3] = xpos_d - xpos
        res = np.zeros(3)
        mujoco.mju_subQuat(res, xquat, xquat_d)
        mujoco.mju_rotVecQuat(res, res, xquat)
        error[3:] = -res

        J = np.concatenate((jacp, jacr))
        v = J @ self.data.qvel
        Kp = np.diag(self.K)
        Kd = np.diag(2.0 * np.sqrt(self.K))

        for i in range(1, 8):
            joint_name = self._joint_name(robot, i)
            dofadr = int(self.model.joint(joint_name).dofadr)

            torque = float(self.data.qfrc_bias[dofadr])
            torque += float(J[:, dofadr].T @ Kp @ error)
            torque -= float(J[:, dofadr].T @ Kd @ v)

            self._set_act(joint_name, torque)

    def _hold_loop(self) -> None:
        dt = 1.0 / float(self.hold_hz)
        while self.run and self._hold_running:
            with self._state_lock:
                for robot in self.ROBOTS:
                    self.control(robot, self._target_pos(robot), self._target_quat(robot))
            mujoco.mj_step(self.model, self.data)
            time.sleep(dt)

    # ---------------- motion primitives ----------------
    def _reach_pose(self, robot: str, pos_goal: np.ndarray, quat_goal: np.ndarray,
                    pos_tol: float = 0.004, ang_tol: float = 0.06,
                    timeout: float = 2.2) -> bool:
        pos_goal = self._safe_pos(robot, pos_goal)
        t0 = _now()
        self._set_target_pos(robot, pos_goal)
        self._set_target_quat(robot, quat_goal)
        dt = 1.0 / float(self.ctrl_hz)

        while _now() - t0 < timeout and not self.stop_flag.is_set():
            hand = self.data.body(self._hand_name(robot))
            p_err = float(np.linalg.norm(pos_goal - hand.xpos))
            a_err = float(self._quat_err(quat_goal, hand.xquat))
            if p_err < pos_tol and a_err < ang_tol:
                return True
            time.sleep(dt)
        return False

    def _adaptive_duration(self, start: np.ndarray, goal: np.ndarray) -> float:
        d = goal - start
        dxy = float(np.linalg.norm(d[:2]))
        dz = abs(float(d[2]))
        t_xy = dxy / max(self.max_speed_xy, 1e-6)
        t_z = dz / max(self.max_speed_z, 1e-6)
        return float(_clamp(max(t_xy, t_z, self.min_move_dur), self.min_move_dur, self.max_move_dur))

    def move_to(self, robot: str, x: float, y: float, z: float, duration_s: float | None = None) -> None:
        robot = self._robot(robot)
        hand = self.data.body(self._hand_name(robot))
        target = self._safe_pos(robot, np.array([x, y, z], dtype=float))
        self._move_linear(robot, target, hand.xquat.copy(), duration_s=duration_s)
        self._reach_pose(robot, target, hand.xquat.copy(), pos_tol=0.012, ang_tol=0.14, timeout=2.4)

    def _move_linear(self, robot: str, target_pos: np.ndarray, xquat_ref: np.ndarray,
                     duration_s: float | None = None) -> None:
        robot = self._robot(robot)
        target_pos = self._safe_pos(robot, target_pos)
        start = self.data.body(self._hand_name(robot)).xpos.copy()
        if duration_s is None:
            duration_s = self._adaptive_duration(start, target_pos)
        steps = max(1, int(duration_s * self.ctrl_hz))
        dt = 1.0 / float(self.ctrl_hz)
        self._set_target_quat(robot, xquat_ref)
        for k in range(steps):
            if self.stop_flag.is_set():
                return
            a = (k + 1) / steps
            pos = (1.0 - a) * start + a * target_pos
            self._set_target_pos(robot, self._safe_pos(robot, pos))
            time.sleep(dt)

    def _descend_with_xy_lock(self, robot: str, xy_ref, z_from: float, z_to: float,
                              xquat_ref: np.ndarray, duration_s: float,
                              xy_alpha: float = 0.45) -> None:
        robot = self._robot(robot)
        steps = max(1, int(duration_s * self.ctrl_hz))
        dt = 1.0 / float(self.ctrl_hz)

        def _xy() -> np.ndarray:
            return xy_ref() if callable(xy_ref) else xy_ref

        for k in range(steps):
            if self.stop_flag.is_set():
                return
            a = (k + 1) / steps
            z = (1.0 - a) * z_from + a * z_to
            xy = np.asarray(_xy(), dtype=float)
            cur_xy = self._target_pos(robot)[:2]
            new_xy = (1.0 - xy_alpha) * cur_xy + xy_alpha * xy
            x, y = self._safe_xy(robot, float(new_xy[0]), float(new_xy[1]))
            self._set_target_pos(robot, np.array([x, y, z], dtype=float))
            self._set_target_quat(robot, xquat_ref)
            time.sleep(dt)

    def wait(self, seconds: float) -> None:
        t0 = _now()
        dt = 1.0 / float(self.ctrl_hz)
        while _now() - t0 < seconds:
            if self.stop_flag.is_set():
                return
            time.sleep(dt)

    # ---------------- home/reset ----------------
    def reset_home(self, robot: str) -> None:
        robot = self._robot(robot)
        self.stop_flag.clear()
        self._set_held_obj(robot, None)
        self.gripper(robot, True)
        for i in range(1, 8):
            self.data.joint(self._joint_name(robot, i)).qpos = self.qpos0[i - 1]
        mujoco.mj_forward(self.model, self.data)
        hand = self.data.body(self._hand_name(robot))
        self._state(robot)["target_pos"] = hand.xpos.copy()
        self._state(robot)["target_quat"] = hand.xquat.copy()
        self._state(robot)["home_pos"] = hand.xpos.copy()
        self._state(robot)["home_quat"] = hand.xquat.copy()

    def reset_home_both(self) -> None:
        for robot in self.ROBOTS:
            self.reset_home(robot)

    def return_home_smooth(self, robot: str, duration_s: float = 1.8) -> None:
        robot = self._robot(robot)
        self.stop_flag.clear()
        hand = self.data.body(self._hand_name(robot))
        start = hand.xpos.copy()
        goal = self._home_pos(robot)
        quat = self._home_quat(robot)
        steps = max(1, int(duration_s * self.ctrl_hz))
        dt = 1.0 / float(self.ctrl_hz)
        for k in range(steps):
            if self.stop_flag.is_set():
                return
            a = (k + 1) / steps
            self._set_target_pos(robot, (1.0 - a) * start + a * goal)
            self._set_target_quat(robot, quat)
            time.sleep(dt)
        self._reach_pose(robot, goal, quat, pos_tol=0.006, ang_tol=0.08, timeout=2.2)

    def reset_blocks(self) -> None:
        for joint_name, qpos in self.INITIAL_BLOCKS.items():
            try:
                self.data.joint(joint_name).qpos[:] = np.array(qpos, dtype=float)
                self.data.joint(joint_name).qvel[:] = 0
            except Exception:
                pass
        self.held_claims.clear()
        for r in self.ROBOTS:
            if r in self.robot_state:
                self._state(r)["held_obj"] = None
        mujoco.mj_forward(self.model, self.data)

    def _retreat_safe(self, robot: str) -> None:
        hand = self.data.body(self._hand_name(robot))
        pos = hand.xpos.copy()
        pos[2] = max(float(pos[2]), self.retreat_z_min)
        self._move_linear(robot, pos, self._home_quat(robot), duration_s=0.6)
        self._reach_pose(robot, pos, self._home_quat(robot), pos_tol=0.02, ang_tol=0.25, timeout=1.6)
        self.wait(self.retreat_settle_s)

    # ---------------- pick/place ----------------
    def _make_pick_orientation(self, robot: str, obj_name: str) -> np.ndarray:
        base_q = self._home_quat(robot).copy()
        if not self.align_yaw_to_object:
            return base_q
        try:
            obj_yaw = float(self._body_yaw(obj_name))
            base_yaw = float(self._yaw_from_quat(base_q))
            dyaw = float(self._wrap_pi(obj_yaw - base_yaw))
            qz = self._quat_from_yaw(dyaw)
            return self._quat_mul(qz, base_q)
        except Exception:
            return base_q

    def _holding_object_now(self, robot: str, obj_name: str) -> bool:
        hand = self.data.body(self._hand_name(robot))
        obj = self.data.body(obj_name)
        dxy = float(np.linalg.norm(obj.xpos[:2] - hand.xpos[:2]))
        return dxy < self.verify_max_dxy

    def _lift_verify_grasp(self, robot: str, obj_name: str, xquat_ref: np.ndarray) -> bool:
        hand = self.data.body(self._hand_name(robot))
        obj = self.data.body(obj_name)
        obj_z0 = float(obj.xpos[2])
        lift = hand.xpos.copy()
        lift[2] += self.verify_lift
        self._move_linear(robot, lift, xquat_ref, duration_s=0.9)
        self._reach_pose(robot, lift, xquat_ref, pos_tol=0.012, ang_tol=0.12, timeout=2.0)
        obj_z1 = float(obj.xpos[2])
        dxy = float(np.linalg.norm(obj.xpos[:2] - hand.xpos[:2]))
        return bool((obj_z1 - obj_z0) > self.verify_min_rise and dxy < self.verify_max_dxy)

    def pick_only_once(self, robot: str, obj_name: str) -> bool:
        robot = self._robot(robot)
        obj = self._require_allowed_body(robot, obj_name)
        self.stop_flag.clear()
        self.held_claims[obj] = robot

        xy0 = self._body_xy(obj)
        x0, y0 = self._safe_xy(robot, float(xy0[0]), float(xy0[1]))
        z_top0 = self._body_top_z(obj)
        xquat_ref = self._make_pick_orientation(robot, obj)
        hover = np.array([x0, y0, z_top0 + self.hover_clear], dtype=float)

        self.gripper(robot, True)
        self.wait(0.10)
        self._move_linear(robot, hover, xquat_ref)
        self._reach_pose(robot, hover, xquat_ref, 0.010, 0.12, 2.4)

        xy1 = self._body_xy(obj)
        x1, y1 = self._safe_xy(robot, float(xy1[0]), float(xy1[1]))
        z_top1 = self._body_top_z(obj)
        pregrasp = np.array([x1, y1, z_top1 + self.pre_clear], dtype=float)
        self._move_linear(robot, pregrasp, xquat_ref)
        self._reach_pose(robot, pregrasp, xquat_ref, 0.010, 0.12, 2.4)

        xy2 = self._body_xy(obj)
        x2, y2 = self._safe_xy(robot, float(xy2[0]), float(xy2[1]))
        z_top2 = self._body_top_z(obj)
        near = np.array([x2, y2, z_top2 + self.near_clear], dtype=float)

        if self.track_object_during_descend:
            xy_live: Callable[[], np.ndarray] = lambda: self._body_xy(obj)
        else:
            xy_live = lambda: xy2

        self._descend_with_xy_lock(robot, xy_live, z_from=pregrasp[2], z_to=near[2],
                                   xquat_ref=xquat_ref, duration_s=0.8, xy_alpha=0.45)
        self._reach_pose(robot, near, xquat_ref, 0.012, 0.14, 2.2)

        z_top3 = self._body_top_z(obj)
        touch = np.array([self._body_xy(obj)[0], self._body_xy(obj)[1], z_top3 + self.touch_clear], dtype=float)
        self._descend_with_xy_lock(robot, xy_live, z_from=float(self._target_pos(robot)[2]), z_to=touch[2],
                                   xquat_ref=xquat_ref, duration_s=1.2, xy_alpha=0.60)
        self._reach_pose(robot, touch, xquat_ref, 0.014, 0.16, 2.2)
        self.wait(0.10)

        self.gripper(robot, False)
        self.wait(0.28)
        ok = self._lift_verify_grasp(robot, obj, xquat_ref)
        if not ok:
            self.gripper(robot, True)
            self.wait(0.18)
            self.held_claims.pop(obj, None)
            self._retreat_safe(robot)
            return False

        self._set_held_obj(robot, obj)
        xy_c = self._body_xy(obj)
        x_c, y_c = self._safe_xy(robot, float(xy_c[0]), float(xy_c[1]))
        z_top_c = self._body_top_z(obj)
        carry = np.array([x_c, y_c, z_top_c + self.hover_clear], dtype=float)
        self._move_linear(robot, carry, xquat_ref, duration_s=1.0)
        self._reach_pose(robot, carry, xquat_ref, 0.014, 0.16, 2.0)
        return True

    def pick_only(self, robot: str, obj_name: str, attempts: int | None = None) -> bool:
        robot = self._robot(robot)
        obj = self._require_allowed_body(robot, obj_name)
        attempts = int(attempts or self.max_grasp_attempts)
        for k in range(attempts):
            with self._console_lock:
                self.console_status = f"{self.ROBOTS[robot]['label']} picking {obj} ({k+1}/{attempts})"
            if self.pick_only_once(robot, obj):
                return True
            self.wait(0.15)
        with self._console_lock:
            self.console_status = f"Failed to grasp {obj}"
        return False

    def place_xy(self, robot: str, x: float, y: float) -> bool:
        robot = self._robot(robot)
        x, y = self._safe_xy(robot, float(x), float(y))
        self.stop_flag.clear()
        hand = self.data.body(self._hand_name(robot))
        xquat_ref = hand.xquat.copy()
        z_place = 0.035
        z_hover = z_place + 0.15
        hover = np.array([x, y, z_hover], dtype=float)
        place = np.array([x, y, z_place], dtype=float)

        self._move_linear(robot, hover, xquat_ref)
        self._reach_pose(robot, hover, xquat_ref, 0.010, 0.12, 2.4)

        held = self._held_obj(robot)
        if held is not None and not self._holding_object_now(robot, held):
            self._set_held_obj(robot, None)
            return False

        self._descend_with_xy_lock(robot, np.array([x, y]), z_from=hover[2], z_to=place[2],
                                   xquat_ref=xquat_ref, duration_s=1.1, xy_alpha=0.55)
        self._reach_pose(robot, place, xquat_ref, 0.012, 0.14, 2.2)
        self.wait(0.10)
        self.gripper(robot, True)
        self.wait(0.22)
        self._set_held_obj(robot, None)
        self._move_linear(robot, hover, xquat_ref, duration_s=1.0)
        self._reach_pose(robot, hover, xquat_ref, 0.012, 0.14, 2.0)
        return True

    def pick_place_xy(self, robot: str, obj: str, x: float, y: float) -> None:
        robot = self._robot(robot)
        if not self.pick_only(robot, obj):
            return
        ok = self.place_xy(robot, x, y)
        if not ok:
            self.wait(0.3)
            if self.pick_only(robot, obj, attempts=2):
                self.place_xy(robot, x, y)

    def pick_place_default(self, robot: str, obj: str) -> None:
        x, y, _ = self.ROBOTS[self._robot(robot)]["default_place"]  # type: ignore[misc]
        self.pick_place_xy(robot, obj, x, y)

    def pick_place_both(self, left_obj: str, right_obj: str,
                        left_xy: tuple[float, float] | None = None,
                        right_xy: tuple[float, float] | None = None) -> None:
        # Run concurrently, but each robot is locked to its own workspace and its own object list.
        if left_xy is None:
            lx, ly, _ = self.ROBOTS["robot1"]["default_place"]  # type: ignore[misc]
        else:
            lx, ly = left_xy
        if right_xy is None:
            rx, ry, _ = self.ROBOTS["robot2"]["default_place"]  # type: ignore[misc]
        else:
            rx, ry = right_xy
        t1 = Thread(target=self.pick_place_xy, args=("robot1", left_obj, lx, ly), daemon=True)
        t2 = Thread(target=self.pick_place_xy, args=("robot2", right_obj, rx, ry), daemon=True)
        t1.start(); t2.start()
        t1.join(); t2.join()
        self.return_home_smooth("robot1")
        self.return_home_smooth("robot2")

    def list_objects(self) -> str:
        lines = ["Objects:"]
        for nm in self._valid_object_names():
            b = self.data.body(nm)
            lines.append(f"  - {nm:10s} pos=({b.xpos[0]:+.3f},{b.xpos[1]:+.3f},{b.xpos[2]:+.3f})")
        lines.append("Allowed: robot1 -> red_box/yellow_box/box; robot2 -> green_box/blue_box")
        return "\n".join(lines)

    # ---------------- command execution ----------------
    def _execute_parsed_command(self, cmd: dict, raw: str | None = None) -> None:
        if raw:
            with self._console_lock:
                self.console_history.append(raw)
                self.console_history = self.console_history[-10:]
        task = cmd.get("task", "unknown")
        robot = cmd.get("robot", "robot1")

        def run() -> None:
            try:
                if task == "quit":
                    self.run = False
                    self._hold_running = False
                    self.stop_flag.set()
                    return
                if task == "help":
                    print("Commands: robot1 pick red place bin | robot2 pick green place bin | both pick red and green | open/close/reset")
                    return
                if task == "list_objects":
                    print(self.list_objects())
                    return
                if task == "reset":
                    if robot == "both":
                        self.reset_home_both()
                    else:
                        self.reset_home(robot)
                    return
                if task == "gripper":
                    if robot == "both":
                        self.gripper_both(open=(cmd.get("mode") == "open"))
                    else:
                        self.gripper(robot, open=(cmd.get("mode") == "open"))
                    return
                if task in {"pick_place", "pick_place_xy"}:
                    obj = cmd.get("obj", "box")
                    if task == "pick_place_xy":
                        x = float(cmd["x"]); y = float(cmd["y"])
                    else:
                        x, y, _ = self.ROBOTS[self._robot(robot)]["default_place"]  # type: ignore[misc]
                    self.pick_place_xy(robot, obj, x, y)
                    self.return_home_smooth(robot)
                    return
                with self._console_lock:
                    self.console_status = f"Unknown/unhandled command: {cmd}"
            except Exception as e:
                with self._console_lock:
                    self.console_status = f"Error: {e}"
                print("ERROR:", e)

        Thread(target=run, daemon=True).start()

    def start_terminal_control(self) -> None:
        def loop() -> None:
            print("\n--- Dual Panda Terminal Control ---")
            print("Examples:")
            print("  robot1 pick red place bin")
            print("  robot2 pick green place bin")
            print("  robot1 open gripper | robot2 close gripper | both open gripper")
            print("  list objects | reset | quit\n")
            while self.run:
                try:
                    text = input("NL> ").strip()
                except (EOFError, KeyboardInterrupt):
                    text = "quit"
                cmd = parse_command(text)
                self._execute_parsed_command(cmd, raw=text)
        Thread(target=loop, daemon=True).start()

    # ---------------- viewer ----------------
    def render(self) -> None:
        glfw.init()
        glfw.window_hint(glfw.SAMPLES, 8)
        window = glfw.create_window(self.width, self.height, "Dual Panda Control", None, None)
        glfw.make_context_current(window)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100)
        opt = mujoco.MjvOption()
        pert = mujoco.MjvPerturb()
        viewport = mujoco.MjrRect(0, 0, self.width, self.height)

        def on_char(win, codepoint):
            if not self.viewer_console_enabled:
                return
            ch = chr(codepoint)
            if ch.isprintable():
                with self._console_lock:
                    self.console_input += ch

        def on_key(win, key, scancode, action, mods):
            if not self.viewer_console_enabled or action not in (glfw.PRESS, glfw.REPEAT):
                return
            if key == glfw.KEY_BACKSPACE:
                with self._console_lock:
                    self.console_input = self.console_input[:-1]
            elif key == glfw.KEY_ESCAPE:
                with self._console_lock:
                    self.console_input = ""
            elif key in (glfw.KEY_ENTER, glfw.KEY_KP_ENTER):
                with self._console_lock:
                    text = self.console_input.strip()
                    self.console_input = ""
                if text:
                    self._execute_parsed_command(parse_command(text), raw=text)

        glfw.set_char_callback(window, on_char)
        glfw.set_key_callback(window, on_key)

        while not glfw.window_should_close(window):
            w, h = glfw.get_framebuffer_size(window)
            viewport.width, viewport.height = w, h
            mujoco.mjv_updateScene(self.model, self.data, opt, pert, self.cam,
                                   mujoco.mjtCatBit.mjCAT_ALL, self.scene)
            mujoco.mjr_render(viewport, self.scene, self.context)
            with self._console_lock:
                overlay = (
                    f"Active robots: Independent | Status: {self.console_status}\n"
                    f"Left holding: {self._held_obj('robot1') or '-'}\n"
                    f"Right holding: {self._held_obj('robot2') or '-'}"
                )
                if self.viewer_console_enabled:
                    overlay += "\n" + "\n".join(self.console_history[-5:]) + f"\n> {self.console_input}_"
            mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_100,
                               mujoco.mjtGridPos.mjGRID_TOPLEFT,
                               viewport, overlay, "", self.context)
            time.sleep(1.0 / self.fps)
            glfw.swap_buffers(window)
            glfw.poll_events()
        self.run = False
        self._hold_running = False
        self.stop_flag.set()
        glfw.terminate()

    def start(self) -> None:
        Thread(target=self._hold_loop, daemon=True).start()
        self.render()


# ---------------- GUI ----------------
def launch_gui(demo: Demo) -> None:
    root = tk.Tk()
    root.title("Dual Panda Advanced Control")
    root.geometry("860x610")

    def run_async(fn, *args):
        def worker():
            try:
                fn(*args)
            except Exception as e:
                root.after(0, lambda: messagebox.showerror("Error", str(e)))
        Thread(target=worker, daemon=True).start()

    speed_map = {"slow": 0.18, "normal": 0.30, "fast": 0.45}

    def set_speed(choice: str):
        demo.max_speed_xy = speed_map.get(choice, 0.30)
        demo.max_speed_z = max(0.15, demo.max_speed_xy * 0.85)

    # ---------- left/right panels ----------
    top = ttk.Frame(root)
    top.pack(fill="x", padx=12, pady=8)

    panels: dict[str, dict[str, tk.Variable]] = {}

    def make_robot_panel(parent, robot: str, title: str, defaults: tuple[str, float, float, float]):
        frame = ttk.LabelFrame(parent, text=title)
        frame.pack(side="left", fill="both", expand=True, padx=6)
        box_default, x0, y0, z0 = defaults
        speed = tk.StringVar(value="normal")
        box = tk.StringVar(value=box_default)
        x = tk.StringVar(value=f"{x0:.2f}")
        y = tk.StringVar(value=f"{y0:.2f}")
        z = tk.StringVar(value=f"{z0:.2f}")
        panels[robot] = {"speed": speed, "box": box, "x": x, "y": y, "z": z}

        ttk.Label(frame, text="Speed:").grid(row=0, column=0, sticky="e", padx=4, pady=4)
        ttk.Combobox(frame, values=["slow", "normal", "fast"], textvariable=speed, width=11, state="readonly").grid(row=0, column=1, sticky="w", padx=4)
        ttk.Label(frame, text="Target box:").grid(row=1, column=0, sticky="e", padx=4, pady=4)
        ttk.Combobox(frame, values=demo._allowed_objects(robot), textvariable=box, width=14, state="readonly").grid(row=1, column=1, sticky="w", padx=4)
        for i, (lab, var) in enumerate([("X:", x), ("Y:", y), ("Z:", z)], start=2):
            ttk.Label(frame, text=lab).grid(row=i, column=0, sticky="e", padx=4, pady=4)
            ttk.Entry(frame, width=12, textvariable=var).grid(row=i, column=1, sticky="w", padx=4)

        def vals():
            set_speed(speed.get())
            return box.get(), float(x.get()), float(y.get()), float(z.get())

        ttk.Button(frame, text="Move", command=lambda: run_async(lambda: demo.move_to(robot, vals()[1], vals()[2], vals()[3]))).grid(row=5, column=0, padx=4, pady=7, sticky="ew")
        ttk.Button(frame, text="Open", command=lambda: run_async(demo.gripper, robot, True)).grid(row=5, column=1, padx=4, pady=7, sticky="ew")
        ttk.Button(frame, text="Close", command=lambda: run_async(demo.gripper, robot, False)).grid(row=6, column=0, padx=4, pady=4, sticky="ew")
        ttk.Button(frame, text="Home", command=lambda: run_async(demo.return_home_smooth, robot)).grid(row=6, column=1, padx=4, pady=4, sticky="ew")
        ttk.Button(frame, text="Capture Block", command=lambda: run_async(lambda: demo.pick_only(robot, vals()[0]))).grid(row=7, column=0, padx=4, pady=4, sticky="ew")
        ttk.Button(frame, text="Release Block", command=lambda: run_async(lambda: demo.place_xy(robot, vals()[1], vals()[2]))).grid(row=7, column=1, padx=4, pady=4, sticky="ew")
        ttk.Button(frame, text="Capture / Return", command=lambda: run_async(lambda: (demo.pick_place_xy(robot, vals()[0], vals()[1], vals()[2]), demo.return_home_smooth(robot)))).grid(row=8, column=0, columnspan=2, padx=4, pady=7, sticky="ew")

    make_robot_panel(top, "robot1", "Left Hand Control", ("red_box", -0.58, 0.18, 0.25))
    make_robot_panel(top, "robot2", "Right Hand Control", ("green_box", 0.58, 0.18, 0.25))

    # ---------- both hands section ----------
    both = ttk.LabelFrame(root, text="Both Hands Control")
    both.pack(fill="x", padx=12, pady=8)

    both_speed = tk.StringVar(value="normal")
    left_box = tk.StringVar(value="red_box")
    right_box = tk.StringVar(value="green_box")
    lx = tk.StringVar(value="-0.58"); ly = tk.StringVar(value="0.18"); lz = tk.StringVar(value="0.25")
    rx = tk.StringVar(value="0.58"); ry = tk.StringVar(value="0.18"); rz = tk.StringVar(value="0.25")

    ttk.Label(both, text="Speed:").grid(row=0, column=0, sticky="e", padx=4, pady=4)
    ttk.Combobox(both, values=["slow", "normal", "fast"], textvariable=both_speed, width=12, state="readonly").grid(row=0, column=1, sticky="w", padx=4)
    ttk.Label(both, text="Left box:").grid(row=1, column=0, sticky="e", padx=4, pady=4)
    ttk.Combobox(both, values=demo._allowed_objects("robot1"), textvariable=left_box, width=14, state="readonly").grid(row=1, column=1, sticky="w", padx=4)
    ttk.Label(both, text="Right box:").grid(row=1, column=2, sticky="e", padx=4, pady=4)
    ttk.Combobox(both, values=demo._allowed_objects("robot2"), textvariable=right_box, width=14, state="readonly").grid(row=1, column=3, sticky="w", padx=4)

    ttk.Label(both, text="Left X/Y/Z:").grid(row=2, column=0, sticky="e", padx=4, pady=4)
    ttk.Entry(both, textvariable=lx, width=9).grid(row=2, column=1, sticky="w")
    ttk.Entry(both, textvariable=ly, width=9).grid(row=2, column=2, sticky="w")
    ttk.Entry(both, textvariable=lz, width=9).grid(row=2, column=3, sticky="w")
    ttk.Label(both, text="Right X/Y/Z:").grid(row=3, column=0, sticky="e", padx=4, pady=4)
    ttk.Entry(both, textvariable=rx, width=9).grid(row=3, column=1, sticky="w")
    ttk.Entry(both, textvariable=ry, width=9).grid(row=3, column=2, sticky="w")
    ttk.Entry(both, textvariable=rz, width=9).grid(row=3, column=3, sticky="w")

    def both_vals():
        set_speed(both_speed.get())
        return (left_box.get(), right_box.get(), float(lx.get()), float(ly.get()), float(lz.get()), float(rx.get()), float(ry.get()), float(rz.get()))

    ttk.Button(both, text="Move Both", command=lambda: run_async(lambda: [demo.move_to("robot1", both_vals()[2], both_vals()[3], both_vals()[4]), demo.move_to("robot2", both_vals()[5], both_vals()[6], both_vals()[7])])).grid(row=4, column=0, padx=4, pady=8, sticky="ew")
    ttk.Button(both, text="Pick and Place Both", command=lambda: run_async(lambda: demo.pick_place_both(both_vals()[0], both_vals()[1], (both_vals()[2], both_vals()[3]), (both_vals()[5], both_vals()[6])))).grid(row=4, column=1, padx=4, pady=8, sticky="ew")
    ttk.Button(both, text="Open Both", command=lambda: run_async(demo.gripper_both, True)).grid(row=4, column=2, padx=4, pady=8, sticky="ew")
    ttk.Button(both, text="Close Both", command=lambda: run_async(demo.gripper_both, False)).grid(row=4, column=3, padx=4, pady=8, sticky="ew")
    ttk.Button(both, text="Home Both", command=lambda: run_async(lambda: [demo.return_home_smooth("robot1"), demo.return_home_smooth("robot2")])).grid(row=5, column=0, padx=4, pady=4, sticky="ew")
    ttk.Button(both, text="Capture Blocks", command=lambda: run_async(lambda: [demo.pick_only("robot1", both_vals()[0]), demo.pick_only("robot2", both_vals()[1])])).grid(row=5, column=1, padx=4, pady=4, sticky="ew")
    ttk.Button(both, text="Release Blocks", command=lambda: run_async(lambda: [demo.place_xy("robot1", both_vals()[2], both_vals()[3]), demo.place_xy("robot2", both_vals()[5], both_vals()[6])])).grid(row=5, column=2, padx=4, pady=4, sticky="ew")
    ttk.Button(both, text="Demo", command=lambda: run_async(lambda: demo.pick_place_both("red_box", "green_box"))).grid(row=5, column=3, padx=4, pady=4, sticky="ew")
    ttk.Button(both, text="Reset Blocks", command=lambda: run_async(demo.reset_blocks)).grid(row=6, column=0, columnspan=4, padx=4, pady=8, sticky="ew")

    ttk.Label(root, text="Robot 1 can only use red/yellow/white boxes. Robot 2 can only use green/blue boxes. Workspaces are separated to avoid collision.").pack(anchor="w", padx=14, pady=4)
    root.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--terminal", action="store_true")
    parser.add_argument("--viewer-console", action="store_true")
    parser.add_argument("--robot", choices=["robot1", "robot2"], default="robot1")
    parser.add_argument("--task", choices=["pick_place_xy"], default=None)
    parser.add_argument("--obj", type=str, default="red_box")
    parser.add_argument("--x", type=float, default=None)
    parser.add_argument("--y", type=float, default=None)
    args = parser.parse_args()

    demo = Demo()
    demo.viewer_console_enabled = bool(args.viewer_console)

    if args.terminal:
        demo.start_terminal_control()
        demo.start()
    elif args.viewer_console:
        demo.start()
    elif args.task == "pick_place_xy":
        if args.x is None or args.y is None:
            raise ValueError("pick_place_xy requires --x and --y")
        Thread(target=lambda: (demo.pick_place_xy(args.robot, args.obj, args.x, args.y), demo.return_home_smooth(args.robot)), daemon=True).start()
        demo.start()
    else:
        Thread(target=launch_gui, args=(demo,), daemon=True).start()
        demo.start()
