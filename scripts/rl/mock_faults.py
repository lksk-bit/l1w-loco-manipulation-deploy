#!/usr/bin/env python3
"""Loopback-only stale-packet, session and command-loss watchdog checks."""

import argparse
import json
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from l1w_teleop.rl import bridge as wire


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bridge", type=Path, required=True)
    p.add_argument("--port", type=int, default=16651)
    a = p.parse_args()
    with (
        tempfile.TemporaryDirectory() as folder,
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock,
    ):
        d = Path(folder)
        sock.bind(("127.0.0.1", a.port + 1))
        sock.settimeout(0.1)
        with (d / "bridge.log").open("w") as log:
            process = subprocess.Popen(
                [
                    str(a.bridge.resolve()),
                    "--mock",
                    "--control",
                    "--bind",
                    "127.0.0.1",
                    "--peer",
                    "127.0.0.1",
                    "--port",
                    str(a.port),
                    "--report",
                    str(d / "report.json"),
                ],
                stdout=log,
                stderr=log,
            )

            def frame():
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    try:
                        return wire.unpack_state(sock.recvfrom(2048)[0])
                    except TimeoutError:
                        pass
                raise RuntimeError(
                    "mock bridge did not produce state: "
                    + (d / "bridge.log").read_text()
                )

            targets = [
                [0, 0, 40, 1, 0] if i % 4 != 3 else [0, 0, 0, 0.7, 0] for i in range(16)
            ]

            def send(state, sequence=1, mode=wire.PASSIVE, session=999):
                sock.sendto(
                    wire.pack_command(
                        session,
                        sequence,
                        time.monotonic_ns(),
                        state,
                        targets,
                        mode=mode,
                    ),
                    ("127.0.0.1", a.port),
                )

            try:
                s = frame()
                send(
                    replace(s, boot=s.boot + 1)
                )  # Wrong controller boot must not latch a session.
                send(
                    replace(s, sequence=s.sequence + 99999)
                )  # Impossible/future observation.
                time.sleep(0.02)
                while True:
                    latest = frame()
                    if latest.sender_ns > s.sender_ns + 15000000:
                        break
                assert latest.accepted_command == 0
                send(latest)
                deadline = time.monotonic() + 1
                accepted = False
                while time.monotonic() < deadline:
                    s = frame()
                    if s.accepted_command == 1:
                        accepted = True
                        break
                assert accepted, (
                    "valid command was not accepted after malformed packets"
                )
                assert s.flags & wire.PASSIVE_READY
                send(s, sequence=2, mode=wire.ACTIVE)
                deadline = time.monotonic() + 1
                while time.monotonic() < deadline:
                    s = frame()
                    if s.accepted_command == 2:
                        break
                assert s.accepted_command == 2
                # Command loss holds support without descent or process exit.
                old_boot = s.boot
                holding_seen = False
                deadline = time.monotonic() + 0.4
                while time.monotonic() < deadline:
                    s = frame()
                    assert s.boot == old_boot
                    assert not s.flags & (wire.DESCENDING | wire.FAULT)
                    holding_seen |= bool(s.flags & wire.HOLDING)
                assert holding_seen and process.poll() is None
                # A new NX can claim an expired owner using an explicit
                # passive request. It cannot claim with an Active/Shutdown.
                send(s, sequence=1, mode=wire.ACTIVE, session=1001)
                send(s, sequence=1, mode=wire.SHUTDOWN, session=1001)
                send(s, sequence=1, mode=wire.PASSIVE, session=1000)
                deadline = time.monotonic() + 8
                while time.monotonic() < deadline:
                    s = frame()
                    if s.accepted_command == 1 and s.flags & wire.PASSIVE_READY:
                        break
                assert s.accepted_command == 1 and s.flags & wire.PASSIVE_READY
                # Explicit shutdown still retires the session.
                send(s, sequence=2, mode=wire.SHUTDOWN, session=1000)
                deadline = time.monotonic() + 8
                while time.monotonic() < deadline:
                    s = frame()
                    if s.boot != old_boot:
                        break
                assert s.boot != old_boot and process.poll() is None
                assert s.accepted_command == 0
                # An old NX's finalizer may echo the retired bridge's fresh
                # token in a late Shutdown. An idle bridge must not reserve
                # that client session before its motor thread rejects it.
                send(s, sequence=3, mode=wire.SHUTDOWN, session=999)
                time.sleep(0.02)
                s = frame()
                send(s, sequence=1, mode=wire.ACTIVE, session=1001)
                time.sleep(0.02)
                s = frame()
                # New NX session, resetting sequence to 1, same bridge PID.
                deadline = time.monotonic() + 1
                while time.monotonic() < deadline:
                    send(s, session=1000)
                    s = frame()
                    if s.accepted_command == 1:
                        break
                assert s.accepted_command == 1 and s.flags & wire.PASSIVE_READY, (
                    "late Shutdown/Active reserved the idle bridge and blocked NX restart"
                )
                send(s, sequence=2, mode=wire.ACTIVE, session=1000)
                deadline = time.monotonic() + 1
                while time.monotonic() < deadline:
                    s = frame()
                    if s.accepted_command == 2:
                        break
                assert s.accepted_command == 2
                process.terminate()  # Explicit operator termination still exits.
                process.wait(timeout=8)
                result = json.loads((d / "report.json").read_text())
                assert process.returncode == 0, result
                assert result["rejected_commands"] >= 2, result
                assert not result["fault"], result
                assert result["completed_sessions"] >= 1, result
                print(
                    json.dumps(
                        {
                            "stale_packets_rejected": True,
                            "command_loss_hold_without_descent": True,
                            "bridge": result,
                        },
                        indent=2,
                    )
                )
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=8)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()


if __name__ == "__main__":
    main()
