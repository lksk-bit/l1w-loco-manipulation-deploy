#!/usr/bin/env python3
"""Read-only SPLINE command/state recorder; run on the 3588 controller.

Never creates SHM, changes consumer flags or sends motor commands. The command
is the SHM publication, not proof of motor-side consumption. Matching repeated
copies reduces torn reads but is not a firmware sequence/freshness guarantee.
"""

import argparse
import json
import math
import mmap
import signal
import socket
import struct
import time
from pathlib import Path

COMMAND_LEG_BYTES = 84  # Four packed 5-float commands and one int32 flag.
STATE_OFFSET = 344
LAYOUT_BYTES = 608


def decode(data):
    if len(data) != LAYOUT_BYTES:
        raise ValueError("unexpected SPLINE layout size")
    targets, position, velocity, torque, flags = [], [], [], [], []
    for leg in range(4):
        for joint in range(4):
            targets.append(
                struct.unpack_from("<5f", data, leg * COMMAND_LEG_BYTES + joint * 20)
            )
            flag, q, dq, tau = struct.unpack_from(
                "<i3f", data, STATE_OFFSET + (leg * 4 + joint) * 16
            )
            flags.append(flag & 0xFFFFFFFF)
            position.append(q)
            velocity.append(dq)
            torque.append(tau)
    if not all(math.isfinite(v) for row in targets for v in row) or not all(
        math.isfinite(v) for row in (position, velocity, torque) for v in row
    ):
        raise ValueError("non-finite SPLINE data")
    return {
        "motor_targets": targets,
        "motor_pos_raw": position,
        "motor_vel_raw": velocity,
        "motor_torque_raw": torque,
        "motor_flags": flags,
        "command_leg_flags": [
            struct.unpack_from("<i", data, leg * COMMAND_LEG_BYTES + 80)[0]
            for leg in range(4)
        ],
        "command_consumer_flags": struct.unpack_from("<2I", data, 336),
        "state_consumer_flags": struct.unpack_from("<2I", data, 600),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shm", type=Path, default=Path("/dev/shm/spline_shm"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=600)
    parser.add_argument("--rate", type=float, default=50)
    args = parser.parse_args()
    if (
        not math.isfinite(args.duration)
        or args.duration <= 0
        or not math.isfinite(args.rate)
        or not 0 < args.rate <= 200
    ):
        parser.error("duration must be positive; rate must be in (0, 200]")
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (
        args.shm.open("rb") as source,
        mmap.mmap(source.fileno(), LAYOUT_BYTES, access=mmap.ACCESS_READ) as shm,
        args.output.open("x") as output,
    ):
        metadata = {
            "kind": "metadata",
            "host": socket.gethostname(),
            "rate_hz": args.rate,
            "duration_seconds": args.duration,
            "leg_order": ["FR", "FL", "RR", "RL"],
            "joint_order": ["abad", "hip", "knee", "wheel"],
            "command_source": "SPLINE shared-memory publication; consumption not confirmed",
        }
        output.write(json.dumps(metadata) + "\n")
        output.flush()
        print(f"Read-only recording: {args.output}", flush=True)
        started = next_tick = time.monotonic()
        samples = 0
        while not stopped and time.monotonic() - started < args.duration:
            stamp = time.monotonic_ns()
            data = shm[:]
            stable = False
            for _ in range(3):
                newer = shm[:]
                if data == newer:
                    stable = True
                    break
                data = newer
            try:
                record = decode(data)
                record.update(
                    kind="sample",
                    sample=samples,
                    monotonic_ns=stamp,
                    wall_time_ns=time.time_ns(),
                    read_stable=stable,
                )
            except ValueError as error:
                record = {
                    "kind": "invalid",
                    "sample": samples,
                    "monotonic_ns": stamp,
                    "error": str(error),
                }
            output.write(json.dumps(record, allow_nan=False) + "\n")
            samples += 1
            if samples % max(1, int(args.rate)) == 0:
                output.flush()
            next_tick += 1 / args.rate
            remaining = next_tick - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            else:
                next_tick = time.monotonic()
        print(f"Recording complete: {samples} samples", flush=True)


if __name__ == "__main__":
    main()
