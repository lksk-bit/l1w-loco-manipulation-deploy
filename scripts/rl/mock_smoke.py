#!/usr/bin/env python3
"""Loopback-only integration check. Never maps SHM or talks to robot addresses."""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bridge", type=Path, required=True)
    p.add_argument("--port", type=int, default=16551)
    args = p.parse_args()
    with tempfile.TemporaryDirectory() as folder:
        d = Path(folder)
        report = d / "bridge.json"
        with (d / "bridge.log").open("w") as log:
            bridge = subprocess.Popen(
                [
                    str(args.bridge.resolve()),
                    "--mock",
                    "--control",
                    "--peer",
                    "127.0.0.1",
                    "--bind",
                    "127.0.0.1",
                    "--port",
                    str(args.port),
                    "--report",
                    str(report),
                ],
                stdout=log,
                stderr=log,
            )
            try:
                time.sleep(0.1)
                if bridge.poll() is not None:
                    raise RuntimeError((d / "bridge.log").read_text())
                result = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "import sys,runpy; sys.modules['mujoco']=None; runpy.run_module('l1w_teleop.rl',run_name='__main__')",
                        "--mode",
                        "mock",
                        "--host",
                        "127.0.0.1",
                        "--bind",
                        "127.0.0.1",
                        "--port",
                        str(args.port),
                        "--state",
                        "locomotion",
                        "--steps",
                        "160",
                        "--report",
                        str(d / "nx.json"),
                        "--log",
                        str(d / "trace.jsonl"),
                    ],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=20,
                    check=False,
                )
                if result.returncode:
                    raise RuntimeError(result.stdout + "\n" + result.stderr)
                bridge.terminate()  # Test teardown explicitly stops the resident bridge.
                bridge.wait(timeout=10)
                if bridge.returncode:
                    raise RuntimeError((d / "bridge.log").read_text())
                nx = json.loads((d / "nx.json").read_text())
                driver = json.loads(report.read_text())
                trace = [
                    json.loads(line)
                    for line in (d / "trace.jsonl").read_text().splitlines()
                ]
                assert nx["writes"] > 0 and not driver["fault"]
                assert any(row["state"] == "locomotion" for row in trace)
                print(
                    json.dumps(
                        {"nx": nx, "bridge": driver, "trace_frames": len(trace)},
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


if __name__ == "__main__":
    main()
