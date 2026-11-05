#!/usr/bin/env python3
"""Full passive/stand/RL round trip against a loopback-only simulated bridge."""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def child(port, directory):
    from l1w_teleop.rl import runtime as app

    class PlannedKeys:
        def __init__(self, _enabled):
            self.cycle = -1

        def read(self):
            self.cycle += 1
            return {10: "3", 120: "2", 230: "3", 260: "1", 275: "3", 540: "\x1b"}.get(
                self.cycle, ""
            )

        def close(self):
            pass

    with mock.patch.object(app, "Keyboard", PlannedKeys):
        return app.main(
            [
                "--mode",
                "mock",
                "--host",
                "127.0.0.1",
                "--bind",
                "127.0.0.1",
                "--port",
                str(port),
                "--log",
                str(directory / "trace.jsonl"),
                "--report",
                str(directory / "nx.json"),
            ]
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge", type=Path)
    parser.add_argument("--port", type=int, default=16751)
    parser.add_argument("--child", type=Path)
    args = parser.parse_args()
    if args.child:
        return child(args.port, args.child)
    if args.bridge is None:
        parser.error("--bridge is required")
    with tempfile.TemporaryDirectory() as directory:
        d = Path(directory)
        with (d / "bridge.log").open("w") as log:
            bridge = subprocess.Popen(
                [
                    str(args.bridge.resolve()),
                    "--mock",
                    "--control",
                    "--bind",
                    "127.0.0.1",
                    "--peer",
                    "127.0.0.1",
                    "--port",
                    str(args.port),
                    "--report",
                    str(d / "bridge.json"),
                ],
                stdout=log,
                stderr=log,
            )
            try:
                time.sleep(0.1)
                result = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--child",
                        str(d),
                        "--port",
                        str(args.port),
                    ],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                assert result.returncode == 0, result.stdout + result.stderr
                bridge.terminate()  # Test teardown explicitly stops the resident bridge.
                bridge.wait(timeout=10)
                assert bridge.returncode == 0, (d / "bridge.log").read_text()
                trace = [
                    json.loads(line)
                    for line in (d / "trace.jsonl").read_text().splitlines()
                ]
                for phase in (
                    "passive",
                    "hip_return",
                    "rise",
                    "stand",
                    "policy",
                    "descending",
                ):
                    assert any(row["phase"] == phase for row in trace), phase
                assert all(row["state"] == "passive" for row in trace[:10])
                assert any(row["state"] == "locomotion" for row in trace[100:120])
                assert all(row["state"] == "passive" for row in trace[275:400]), (
                    "rearm interrupted descent"
                )
                assert any(row["state"] == "locomotion" for row in trace[520:]), (
                    "did not rearm after passive: " + (d / "nx.json").read_text()
                )
                assert len({row["command_session"] for row in trace}) == 1
                assert len({row["bridge_boot"] for row in trace}) == 1
                assert all(row["bridge_flags"] & 4 for row in trace)
                states = []
                for row in trace:
                    if not states or states[-1] != row["state"]:
                        states.append(row["state"])
                print(
                    json.dumps(
                        {
                            "states": states,
                            "nx": json.loads((d / "nx.json").read_text()),
                            "bridge": json.loads((d / "bridge.json").read_text()),
                            "trace_frames": len(trace),
                        },
                        indent=2,
                    )
                )
            finally:
                if bridge.poll() is None:
                    bridge.terminate()
                    try:
                        bridge.wait(timeout=8)
                    except subprocess.TimeoutExpired:
                        bridge.kill()
                        bridge.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
