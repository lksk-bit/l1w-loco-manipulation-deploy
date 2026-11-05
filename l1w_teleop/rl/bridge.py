"""UDP packet layout and latest-only bridge transport."""

import math
import struct
import select
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass

MAGIC = 0x31574C52
VERSION = 2
CONTROL_TIMEOUT_MS = 100
CONTROL_TIMEOUT_US = CONTROL_TIMEOUT_MS * 1000
CONTROL_TIMEOUT_NS = CONTROL_TIMEOUT_US * 1000
STATE = 1
COMMAND = 2
ACTIVE = 1
SHUTDOWN = 2
PASSIVE = 3  # Controller-local supported descent, then persistent damping; session stays owned.
MOTORS_VALID = 1
IMU_VALID = 2
CONTROL = 4
MOCK = 8
FAULT = 16
RT_READY = 32
PASSIVE_READY = 64
DESCENDING = 128
HOLDING = 256
HEADER = struct.Struct("<IHHIIQQQ")
STATE_BODY = struct.Struct("<48f16I10fQQfffII")
COMMAND_BODY = struct.Struct("<QQII80f")
STATE_SIZE = HEADER.size + STATE_BODY.size
COMMAND_SIZE = HEADER.size + COMMAND_BODY.size


@dataclass(frozen=True)
class StateFrame:
    flags: int
    boot: int
    sequence: int
    sender_ns: int
    q: tuple
    dq: tuple
    torque: tuple
    motor_flags: tuple
    quaternion: tuple  # Physical firmware order is configured on NX; mock uses wxyz.
    gyro: tuple
    acceleration: tuple
    imu_stamp: int
    accepted_command: int
    command_age_ms: float
    feedback_age_ms: float
    lateness_us: float
    missed_ticks: int
    profile_crc: int


def unpack_state(data):
    if len(data) != STATE_SIZE:
        raise ValueError("invalid state packet size")
    magic, version, kind, size, flags, boot, seq, stamp = HEADER.unpack_from(data)
    if (magic, version, kind, size) != (MAGIC, VERSION, STATE, STATE_SIZE):
        raise ValueError("invalid state packet header")
    b = STATE_BODY.unpack_from(data, HEADER.size)
    if not all(math.isfinite(x) for x in (*b[:48], *b[64:74], *b[76:79])):
        raise ValueError("nonfinite bridge feedback")
    return StateFrame(
        flags,
        boot,
        seq,
        stamp,
        b[:16],
        b[16:32],
        b[32:48],
        b[48:64],
        b[64:68],
        b[68:71],
        b[71:74],
        *b[74:],
    )


def pack_command(
    session, sequence, sender_ns, state, targets, mode=ACTIVE, ttl_us=CONTROL_TIMEOUT_US
):
    if (
        mode not in (ACTIVE, SHUTDOWN, PASSIVE)
        or not 1000 <= ttl_us <= CONTROL_TIMEOUT_US
    ):
        raise ValueError("invalid command mode/TTL")
    if len(targets) != 16 or any(len(joint) != 5 for joint in targets):
        raise ValueError("expected 16 [p,v,kp,kd,tff] targets")
    values = [float(v) for joint in targets for v in joint]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("nonfinite command")
    return HEADER.pack(
        MAGIC,
        VERSION,
        COMMAND,
        COMMAND_SIZE,
        state.profile_crc,
        session,
        sequence,
        sender_ns,
    ) + COMMAND_BODY.pack(state.boot, state.sequence, mode, ttl_us, *values)


RECEIVE_BATCH_LIMIT = 64


class BridgeLink:
    def __init__(self, host, port=15051, bind="0.0.0.0"):
        if not 1 <= port <= 65534:
            raise ValueError("bridge port must be 1..65534")
        self.peer = (socket.gethostbyname(host), port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((bind, port + 1))
        self.sock.setblocking(False)
        self.lock = threading.Lock()
        self.latest = None
        self.received_ns = 0
        self.invalid = 0
        self.boot_changed = False
        self.retired_boots = deque(maxlen=1024)
        self.ack_first = {}
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._receive, daemon=True)
        self.thread.start()

    def _receive(self):
        while not self.stop.is_set():
            try:
                if not select.select([self.sock], [], [], 0.01)[0]:
                    continue
            except OSError:
                break
            if not self._receive_batch():
                break

    def _receive_batch(self):
        # Drain queued feedback without repeating select for every old packet.
        # Still validate each frame: restart/fault and first-ACK semantics matter.
        for _ in range(RECEIVE_BATCH_LIMIT):
            try:
                data, address = self.sock.recvfrom(2048)
            except BlockingIOError:
                return True
            except OSError:
                return False
            if address != self.peer:
                self.invalid += 1
                continue
            try:
                frame = unpack_state(data)
            except ValueError:
                self.invalid += 1
                continue
            self.accept(frame, time.monotonic_ns())
        return True

    def accept(self, frame, received_ns):
        with self.lock:
            new_motor_sample = True
            if self.latest is not None:
                if frame.boot != self.latest.boot:
                    if frame.boot in self.retired_boots:
                        return False
                    self.retired_boots.append(self.latest.boot)
                    self.boot_changed = True
                    self.ack_first.clear()
                elif frame.sequence < self.latest.sequence:
                    return False
                elif frame.sequence == self.latest.sequence:
                    # The sequence counts motor feedback, not bridge ticks.
                    # A newer tick can report FAULT/invalid feedback or an ACK
                    # even while motors have stopped updating. Expose that
                    # status without making the old motor sample fresh again.
                    if frame.sender_ns <= self.latest.sender_ns:
                        return False
                    new_motor_sample = False
            if frame.accepted_command:
                self.ack_first.setdefault(frame.accepted_command, received_ns)
                if len(self.ack_first) > 64:
                    self.ack_first.pop(next(iter(self.ack_first)))
            self.latest = frame
            if new_motor_sample:
                self.received_ns = received_ns
            return True

    def ack_time(self, sequence):
        with self.lock:
            stamp = self.ack_first.get(sequence)
            self.ack_first = {
                seq: t for seq, t in self.ack_first.items() if seq > sequence
            }
            return stamp

    def snapshot(self):
        with self.lock:
            return self.latest, self.received_ns, self.boot_changed

    def confirm_restart(self, boot):
        """Explicitly acknowledge a new bridge session after manual rearming."""
        with self.lock:
            if self.latest is None or self.latest.boot != boot:
                return False
            self.boot_changed = False
            return True

    def send(self, packet):
        if self.sock.sendto(packet, self.peer) != len(packet):
            raise OSError("short UDP send")

    def close(self):
        self.stop.set()
        self.thread.join(timeout=0.1)
        self.sock.close()
