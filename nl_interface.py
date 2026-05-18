from __future__ import annotations
import re
from typing import Dict, Any, Optional


def _normalize(s: str) -> str:
    s = s.strip().lower().replace(",", " ")
    return re.sub(r"\s+", " ", s)


COLOR_ALIASES = {
    "red": "red_box",
    "green": "green_box",
    "blue": "blue_box",
    "yellow": "yellow_box",
    "white": "box",
    "gray": "box",
    "grey": "box",
}

TARGET_ALIASES = {
    "bin": "bin_center",
    "basket": "bin_center",
    "bucket": "bin_center",
    "left": "zone_left",
    "right": "zone_right",
    "zone left": "zone_left",
    "zone right": "zone_right",
}

ROBOT_ALIASES = {
    "robot1": "robot1", "robot 1": "robot1", "left": "robot1", "left hand": "robot1", "r1": "robot1",
    "robot2": "robot2", "robot 2": "robot2", "right": "robot2", "right hand": "robot2", "r2": "robot2",
    "both": "both", "both hands": "both", "two robots": "both",
}


def _resolve_obj_token(obj_phrase: Optional[str]) -> Optional[str]:
    if not obj_phrase:
        return None
    p = obj_phrase.strip().lower()
    p = re.sub(r"\b(the|box|cube|block)\b", "", p).strip()
    p = re.sub(r"\s+", "_", p)
    if p in COLOR_ALIASES:
        return COLOR_ALIASES[p]
    if p in COLOR_ALIASES.values():
        return p
    return p or None


def _strip_robot_prefix(t: str) -> tuple[str, str]:
    for phrase, robot in sorted(ROBOT_ALIASES.items(), key=lambda x: -len(x[0])):
        if t == phrase:
            return robot, ""
        if t.startswith(phrase + " "):
            return robot, t[len(phrase):].strip()
    return "robot1", t


def _extract_pick_object(t: str) -> Optional[str]:
    m = re.search(r"\b(?:pick|grab|take|move|put)\b\s+(.+?)(?:\s+\b(place|to|in|into|at|on)\b|$)", t)
    if m:
        return m.group(1).strip()
    return None


def parse_command(text: str) -> Dict[str, Any]:
    raw = text
    t = _normalize(text)
    robot, t = _strip_robot_prefix(t)

    if t in {"q", "quit", "exit"}:
        return {"task": "quit"}
    if t in {"h", "help", "?"}:
        return {"task": "help"}
    if t in {"reset", "home"}:
        return {"task": "reset", "robot": robot}
    if t in {"open", "open gripper", "gripper open"} or "open gripper" in t:
        return {"task": "gripper", "robot": robot, "mode": "open"}
    if t in {"close", "close gripper", "gripper close"} or "close gripper" in t:
        return {"task": "gripper", "robot": robot, "mode": "close"}
    if t in {"list", "list objects", "objects", "what objects", "list robots"}:
        return {"task": "list_objects", "robot": robot}

    # simple both-hands spoken form: "both pick red and green"
    m_both = re.search(r"\bpick\b\s+(.+?)\s+\band\b\s+(.+)$", t)
    if robot == "both" and m_both:
        return {
            "task": "pick_place_both",
            "robot": "both",
            "left_obj": _resolve_obj_token(m_both.group(1)) or "red_box",
            "right_obj": _resolve_obj_token(m_both.group(2)) or "green_box",
        }

    if "pick" in t or "grab" in t or "take" in t or "move" in t or "put" in t:
        obj = _resolve_obj_token(_extract_pick_object(t)) or ("green_box" if robot == "robot2" else "red_box")
        mx = re.search(r"\bx\s*(-?\d+(?:\.\d+)?)\b", t)
        my = re.search(r"\by\s*(-?\d+(?:\.\d+)?)\b", t)
        if mx and my:
            return {"task": "pick_place_xy", "robot": robot, "obj": obj, "x": float(mx.group(1)), "y": float(my.group(1))}
        return {"task": "pick_place", "robot": robot, "obj": obj, "target": "bin_center"}

    return {"task": "unknown", "robot": robot, "text": raw}
