#pragma once
#include <cstdint>
#include <cmath>
#include <cstring>

namespace rl {
constexpr uint32_t Magic = 0x31574c52, ProfileMagic = 0x31504652;
constexpr uint32_t ProfileVersion = 3;
constexpr uint32_t CommandTtlUs = 100000;
constexpr uint64_t MaxStateLag = 100; // Motor state sequence advances at most once per 1 kHz tick.
enum : uint16_t { StateKind=1, CommandKind=2 };
enum : uint32_t { MotorsValid=1, ImuValid=2, Control=4, Mock=8, Fault=16, RtReady=32, PassiveReady=64, Descending=128, Holding=256 };
enum : uint32_t { Active=1, Shutdown=2, Passive=3 };
inline uint64_t age_ns(uint64_t now,uint64_t timestamp) { return now>=timestamp?now-timestamp:0; }
inline bool valid_state_reference(uint64_t current,uint64_t reference) {
    return reference<=current && current-reference<=MaxStateLag;
}
#pragma pack(push,1)
struct Header {
    uint32_t magic=Magic; uint16_t version=2, kind=0;
    uint32_t bytes=0, flags=0;
    uint64_t session=0, sequence=0, timestamp_ns=0;
};
struct StatePacket {
    Header h;
    float q[16]{}, dq[16]{}, torque[16]{};
    uint32_t motor_flags[16]{};
    // Physical quaternion is firmware SHM order, interpreted by the NX profile; mock uses wxyz.
    float quaternion[4]{1,0,0,0}, gyro[3]{}, acceleration[3]{};
    uint64_t imu_stamp=0, accepted_command=0;
    float command_age_ms=0, feedback_age_ms=0, lateness_us=0;
    uint32_t missed_ticks=0, profile_crc=0;
};
struct CommandPacket {
    Header h; uint64_t state_session=0, state_sequence=0;
    uint32_t mode=Active, ttl_us=CommandTtlUs;
    float joints[16][5]{};
};
struct Profile {
    uint32_t magic=ProfileMagic, version=ProfileVersion, bytes=0, verified=0;
    float lower[16], upper[16]; // Position soft stops only; targets are clipped, never slew limited.
    float shutdown_q[16], hold_kp[16], hold_kd[16], passive_kd[16];
    uint32_t wheel[16];
};
#pragma pack(pop)
static_assert(sizeof(Header)==40 && sizeof(StatePacket)==372 && sizeof(CommandPacket)==384);
static_assert(sizeof(Profile)==464);

inline uint32_t profile_crc(const Profile& p) {
    uint32_t crc=0xffffffff;
    const auto* bytes=reinterpret_cast<const uint8_t*>(&p);
    for(unsigned i=0;i<sizeof(p);++i) {
        crc^=bytes[i];for(unsigned bit=0;bit<8;++bit)crc=(crc>>1)^((crc&1)?0xedb88320:0);
    }
    return crc^0xffffffff;
}

inline bool valid(const CommandPacket& c, const Profile& p) {
    if (c.h.magic!=Magic || c.h.version!=2 || c.h.kind!=CommandKind || c.h.bytes!=sizeof(c) ||
        !c.h.session || !c.h.sequence || (c.mode!=Active && c.mode!=Shutdown && c.mode!=Passive) ||
        c.ttl_us<1000 || c.ttl_us>CommandTtlUs) return false;
    for (unsigned i=0;i<16;++i) {
        const float* j=c.joints[i];
        for (unsigned k=0;k<5;++k) if (!std::isfinite(j[k])) return false;
        if(c.mode==Shutdown||c.mode==Passive) continue; // Local profile owns descent/damping.
        if (j[2]<0 || j[3]<0 || (p.wheel[i] && j[2]!=0)) return false;
    }
    return true;
}
inline bool valid(const Profile& p) {
    if (p.magic!=ProfileMagic || p.version!=ProfileVersion || p.bytes!=sizeof(p) || p.verified!=1) return false;
    for (unsigned i=0;i<16;++i) {
        const float values[]{p.lower[i],p.upper[i],p.shutdown_q[i],p.hold_kp[i],p.hold_kd[i],p.passive_kd[i]};
        for (float f:values) if (!std::isfinite(f)) return false;
        if (p.hold_kp[i]<0 || p.hold_kd[i]<0 || p.passive_kd[i]<0 ||
            p.wheel[i]!=unsigned(i%4==3) ||
            (p.wheel[i] && p.hold_kp[i]!=0)) return false;
        if (!p.wheel[i] && (p.lower[i]>=p.upper[i] || p.shutdown_q[i]<p.lower[i] ||
                            p.shutdown_q[i]>p.upper[i])) return false;
    }
    return true;
}
} // namespace rl
