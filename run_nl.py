from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from nl_interface import parse_command

ROOT = Path(__file__).resolve().parent
PICK_SCRIPT = ROOT / "pickandplace.py"


def _call_pickandplace(args: list[str]) -> int:
    cmd = [sys.executable, str(PICK_SCRIPT)] + args
    print(f"\n>> Running: {' '.join(cmd)}\n")
    return subprocess.run(cmd).returncode


def main() -> None:
    if not PICK_SCRIPT.exists():
        raise FileNotFoundError(f"Cannot find {PICK_SCRIPT}")

    print("Dual Panda Natural Language Runner")
    print("Better option: py pickandplace.py --terminal")
    print("Examples:")
    print("  robot1 pick red place bin")
    print("  robot2 pick green place bin")
    print("  robot1 pick yellow place x -0.58 y 0.18")
    print("  robot2 pick blue place x 0.58 y 0.18")
    print("  robot1 open gripper | robot2 close gripper | both open gripper")
    print("  reset | quit\n")

    while True:
        text = input("NL> ").strip()
        if not text:
            continue
        cmd = parse_command(text)
        task = cmd.get("task", "unknown")
        robot = cmd.get("robot", "robot1")

        if task == "quit":
            print("Bye.")
            return
        if task == "help":
            print("Use: robot1/robot2 pick <color> place x <x> y <y>; open/close/reset")
            continue
        if task == "unknown":
            print(f"❌ Didn't understand: {cmd.get('text')}")
            continue
        if task == "reset":
            print("For continuous reset, use: py pickandplace.py --terminal")
            continue
        if task == "gripper":
            print("For continuous gripper control, use: py pickandplace.py --terminal")
            continue
        if task == "pick_place_xy":
            _call_pickandplace(["--robot", robot, "--task", "pick_place_xy", "--obj", cmd.get("obj", "red_box"), "--x", str(cmd["x"]), "--y", str(cmd["y"])])
            continue
        if task == "pick_place":
            # Use safe default placing coordinate for each robot.
            x, y = (-0.58, 0.18) if robot == "robot1" else (0.58, 0.18)
            _call_pickandplace(["--robot", robot, "--task", "pick_place_xy", "--obj", cmd.get("obj", "red_box"), "--x", str(x), "--y", str(y)])
            continue
        print(f"❌ Unhandled task: {cmd}")


if __name__ == "__main__":
    main()
