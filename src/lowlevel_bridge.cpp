// NX owns inference. This process owns motor publication and local shutdown.
#include "lowlevel/lowlevel.h"
#include "rl_protocol.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

#include <arpa/inet.h>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

namespace {

constexpr unsigned MotorCount = 16;
constexpr uint64_t MotorPeriodNs = 1'000'000;
constexpr float MotorPeriodSeconds = 0.001f;
constexpr uint64_t StatePeriodNs = 2'000'000;
constexpr uint64_t OwnershipCheckPeriodNs = 100'000'000;
constexpr uint64_t MotorFeedbackTimeoutNs = 20'000'000;
constexpr uint64_t ImuFeedbackTimeoutNs = 50'000'000;
constexpr uint64_t PublicationWarningNs = 10'000'000;
constexpr uint64_t PublicationTimeoutNs = 20'000'000;
constexpr std::array<uint64_t, 5> PublicationGapBucketsNs{
    2'000'000, 5'000'000, 10'000'000, 15'000'000, 20'000'000};
constexpr uint64_t HardLatenessNs = 10'000'000;
constexpr uint64_t PassiveExitDelayNs = 500'000'000;
constexpr unsigned ReceiveBatchLimit = 32;
constexpr unsigned MaxLatenessUs = 20'000;
constexpr unsigned RestStableTicks = 200;
constexpr float RestPositionTolerance = 0.08f;
constexpr float RestVelocityTolerance = 0.2f;
constexpr float ShutdownGainSeconds = 0.15f;
constexpr float ShutdownDescentSeconds = 3.0f;
constexpr float ShutdownTrajectorySeconds = 3.15f;
constexpr float ShutdownFaultSeconds = 8.0f;
constexpr float EmergencyExitSeconds = 0.3f;

volatile sig_atomic_t requested_stop = 0;

void request_stop(int) { requested_stop = 1; }

uint64_t now_ns() {
    timespec time{};
    clock_gettime(CLOCK_MONOTONIC, &time);
    return uint64_t(time.tv_sec) * 1'000'000'000 + time.tv_nsec;
}

// Startup resources and configuration. None of these runs in a motor tick.
struct Options {
    bool mock = false;
    bool control = false;
    int cpu = -1;
    int priority = 60;
    int port = 15051;
    double duration = 0;
    std::string peer = "192.168.234.234";
    std::string bind = "0.0.0.0";
    std::string profile;
    std::string report;
    std::string shm = "/dev/shm/spline_shm";
    std::string imu = "/dev/shm/imu_shm";
};

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        auto value = [&]() {
            if (++i >= argc) throw std::runtime_error("missing option value");
            return std::string(argv[i]);
        };
        if (argument == "--mock") options.mock = true;
        else if (argument == "--control") options.control = true;
        else if (argument == "--cpu") options.cpu = std::stoi(value());
        else if (argument == "--rt-priority") options.priority = std::stoi(value());
        else if (argument == "--port") options.port = std::stoi(value());
        else if (argument == "--peer") options.peer = value();
        else if (argument == "--bind") options.bind = value();
        else if (argument == "--profile") options.profile = value();
        else if (argument == "--report") options.report = value();
        else if (argument == "--duration") options.duration = std::stod(value());
        else if (argument == "--shm") options.shm = value();
        else if (argument == "--imu") options.imu = value();
        else if (argument == "--help") {
            std::cout
                << "lowlevel_bridge [--mock] [--control --profile FILE --cpu N] "
                   "[--peer NX_IP] [--port 15051] [--duration SEC] [--report JSON]\n"
                   "Default is read-only. No process is stopped or restarted automatically.\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown option: " + argument);
        }
    }
    if (options.port < 1 || options.port > 65534 || options.priority < 1 ||
        options.priority > 90 || options.cpu >= CPU_SETSIZE || options.cpu < -1 ||
        !std::isfinite(options.duration) || options.duration < 0) {
        throw std::runtime_error("invalid options");
    }
    if (options.control && !options.mock && (options.profile.empty() || options.cpu < 0)) {
        throw std::runtime_error("control requires verified profile and explicit CPU");
    }
    return options;
}

class MappedRegion {
public:
    MappedRegion(const std::string& path, size_t size, bool writable) : bytes_(size) {
        fd_ = open(path.c_str(), writable ? O_RDWR : O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("cannot open existing SHM: " + path);
        struct stat status{};
        if (fstat(fd_, &status) || status.st_size < off_t(size)) {
            close(fd_);
            throw std::runtime_error("SHM layout too short");
        }
        data_ = mmap(nullptr, size, PROT_READ | (writable ? PROT_WRITE : 0), MAP_SHARED, fd_, 0);
        if (data_ == MAP_FAILED) {
            close(fd_);
            throw std::runtime_error("mmap failed");
        }
        if (writable && flock(fd_, LOCK_EX | LOCK_NB)) {
            munmap(data_, bytes_);
            close(fd_);
            throw std::runtime_error("another bridge holds the SHM lock");
        }
    }
    ~MappedRegion() {
        munmap(data_, bytes_);
        close(fd_);
    }
    MappedRegion(const MappedRegion&) = delete;
    MappedRegion& operator=(const MappedRegion&) = delete;
    void* data() const { return data_; }

private:
    void* data_ = nullptr;
    size_t bytes_ = 0;
    int fd_ = -1;
};

class UdpSocket {
public:
    explicit UdpSocket(const Options& options) {
        fd_ = socket(AF_INET, SOCK_DGRAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
        if (fd_ < 0) throw std::runtime_error("socket failed");
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_port = htons(options.port);
        if (inet_pton(AF_INET, options.bind.c_str(), &address.sin_addr) != 1 ||
            bind(fd_, reinterpret_cast<sockaddr*>(&address), sizeof(address))) {
            close(fd_);
            throw std::runtime_error("UDP bind failed");
        }
    }
    ~UdpSocket() { close(fd_); }
    UdpSocket(const UdpSocket&) = delete;
    UdpSocket& operator=(const UdpSocket&) = delete;
    int fd() const { return fd_; }

private:
    int fd_ = -1;
};

bool vendor_controller_running() {
    for (const auto& entry : std::filesystem::directory_iterator("/proc")) {
        const auto name = entry.path().filename().string();
        if (name.empty() || !std::all_of(name.begin(), name.end(), [](char c) {
                return c >= '0' && c <= '9';
            })) {
            continue;
        }
        std::ifstream file(entry.path() / "comm");
        std::string command;
        std::getline(file, command);
        if (command == "mc_ctrl") return true;
    }
    return false;
}

rl::Profile load_profile(const std::string& path) {
    rl::Profile profile{};
    std::ifstream file(path, std::ios::binary);
    file.read(reinterpret_cast<char*>(&profile), sizeof(profile));
    if (!file || file.peek() != EOF || !rl::valid(profile)) {
        throw std::runtime_error("invalid/unchecked bridge profile");
    }
    return profile;
}

rl::Profile make_mock_profile() {
    rl::Profile profile{};
    profile.bytes = sizeof(profile);
    profile.verified = 1;
    for (unsigned i = 0; i < MotorCount; ++i) {
        profile.lower[i] = -10;
        profile.upper[i] = 10;
        profile.shutdown_q[i] = 0;
        profile.hold_kp[i] = i % 4 == 3 ? 0 : 40;
        profile.hold_kd[i] = 1;
        profile.passive_kd[i] = 0.3f;
        profile.wheel[i] = i % 4 == 3;
    }
    return profile;
}

joint_control_t& motor_command(spline_cmd_data_t& command, unsigned index) {
    auto* joints = reinterpret_cast<joint_control_t*>(&command.legs[index / 4]);
    return joints[index % 4];
}

const joint_state_t& motor_feedback(const spline_state_data_t& state, unsigned index) {
    const auto* joints = reinterpret_cast<const joint_state_t*>(&state.legs[index / 4]);
    return joints[index % 4];
}

void check_motor_feedback() {
    MappedRegion region("/dev/shm/spline_shm", sizeof(spline_data_t), false);
    const auto* data = static_cast<spline_data_p>(region.data());
    spline_state_data_t state{};
    std::memcpy(&state, &data->state, sizeof(state));
    for (unsigned i = 0; i < MotorCount; ++i) {
        const auto& joint = motor_feedback(state, i);
        if (!(joint.flags & 1) || (joint.flags & 0x3e) || !std::isfinite(joint.p) ||
            !std::isfinite(joint.v) || !std::isfinite(joint.t)) {
            throw std::runtime_error(
                "motor feedback must be finite, enabled and free of fault bits");
        }
    }
}

// Shared snapshots cross threads; MotorRuntime belongs only to the motor thread.
enum class SharedFailure : unsigned { None, VendorController, RtSetup, Network, OwnershipCheck };

struct SharedState {
    std::mutex mutex;
    rl::StatePacket state{};
    rl::CommandPacket command{};
    uint64_t command_received_ns = 0;
    std::atomic<bool> fault{false};
    std::atomic<SharedFailure> fault_reason{SharedFailure::None};
    std::atomic<bool> failed{false};
    std::atomic<bool> done{false};
    std::atomic<uint64_t> rejected{0};
    std::atomic<uint64_t> skipped_snapshots{0};
    std::atomic<uint64_t> publication_skips{0};
};

void set_shared_fault(SharedState& shared, SharedFailure reason) {
    auto expected = SharedFailure::None;
    shared.fault_reason.compare_exchange_strong(expected, reason);
    shared.fault = true;
}

const char* shared_fault_name(const SharedState& shared) {
    switch (shared.fault_reason.load()) {
    case SharedFailure::VendorController: return "vendor_controller_restarted";
    case SharedFailure::RtSetup: return "rt_thread_setup_failed";
    case SharedFailure::Network: return "network_thread_exception";
    case SharedFailure::OwnershipCheck: return "ownership_check_failed";
    default: return "shared_fault_unspecified";
    }
}

struct FaultDiagnostic {
    const char* reason = nullptr;
    uint64_t tick = 0;
    uint64_t at_ns = 0;
    uint64_t motor_age_ns = 0;
    uint64_t imu_age_ns = 0;
    uint64_t publication_gap_ns = 0;
    uint64_t lateness_ns = 0;
    std::array<uint32_t, MotorCount> motor_flags{};
};

struct MotorRuntime {
    rl::StatePacket feedback{};
    rl::CommandPacket current{};
    spline_cmd_data_t output{};
    uint64_t received_ns = 0;
    uint64_t seen_received_ns = 0;
    uint64_t feedback_at_ns = 0;
    uint64_t imu_at_ns = 0;
    uint64_t last_imu_stamp = 0;
    uint64_t claimed_session = 0;
    uint64_t accepted_sequence = 0;
    uint64_t started_ns = 0;
    uint64_t last_consumed_ns = 0;
    uint64_t shutdown_started_ns = 0;
    uint64_t passive_started_ns = 0;
    uint64_t shutdown_stable_ticks = 0;
    uint64_t recovery_stable_ticks = 0;
    uint64_t completed_sessions = 0;
    uint64_t command_hold_events = 0;
    uint64_t ticks = 0;
    uint64_t max_publication_gap_ns = 0;
    uint64_t publication_busy_periods = 0;
    uint64_t publication_recoveries = 0;
    uint64_t publication_warning_events = 0;
    uint64_t publication_warning_recoveries = 0;
    uint64_t publication_warnings_with_motor_gap = 0;
    std::array<uint64_t, PublicationGapBucketsNs.size() + 1> publication_recovery_histogram{};
    uint64_t max_motor_age_ns = 0;
    uint64_t max_imu_age_ns = 0;
    FaultDiagnostic fault_diagnostic{};
    std::array<float, MotorCount> shutdown_position{};
    std::array<float, MotorCount> shutdown_kp{};
    std::array<float, MotorCount> shutdown_kd{};
    std::array<float, MotorCount> shutdown_velocity{};
    std::array<float, MotorCount> shutdown_feedforward{};
    bool engaged = false;
    bool shutting_down = false;
    bool hard_fault = false;
    bool passive = false;
    bool command_hold = false;
    bool exit_after_shutdown = false;
    bool publication_busy = false;
    bool publication_warned = false;
};

void record_first_fault(MotorRuntime& runtime, const char* reason, uint64_t now,
                        uint64_t lateness_ns = 0) {
    auto& diagnostic = runtime.fault_diagnostic;
    if (diagnostic.reason) return;
    diagnostic.reason = reason; // Static literals only; no allocation in a motor tick.
    diagnostic.tick = runtime.ticks;
    diagnostic.at_ns = now;
    diagnostic.motor_age_ns = rl::age_ns(now, runtime.feedback_at_ns);
    diagnostic.imu_age_ns = rl::age_ns(now, runtime.imu_at_ns);
    diagnostic.publication_gap_ns = rl::age_ns(now, runtime.last_consumed_ns);
    diagnostic.lateness_ns = lateness_ns;
    std::copy(std::begin(runtime.feedback.motor_flags), std::end(runtime.feedback.motor_flags),
              diagnostic.motor_flags.begin());
}

void set_hard_fault(MotorRuntime& runtime, const char* reason, uint64_t now,
                    uint64_t lateness_ns = 0) {
    record_first_fault(runtime, reason, now, lateness_ns);
    runtime.hard_fault = true;
}

struct FeedbackStatus {
    bool motors = true;
    bool imu = true;
};

struct LoopStatistics {
    std::array<uint64_t, MaxLatenessUs + 1> lateness_histogram{};
    uint64_t max_lateness_ns = 0;
    uint64_t missed_ticks = 0;

    unsigned percentile(double fraction) const {
        uint64_t total = 0;
        for (auto count : lateness_histogram) total += count;
        uint64_t cumulative = 0;
        for (unsigned i = 0; i < lateness_histogram.size(); ++i) {
            cumulative += lateness_histogram[i];
            if (cumulative >= total * fraction) return i;
        }
        return MaxLatenessUs;
    }
};

bool configure_motor_thread(const Options& options, bool& rt_ready) {
    if (options.cpu >= 0) {
        cpu_set_t cpus;
        CPU_ZERO(&cpus);
        CPU_SET(options.cpu, &cpus);
        if (pthread_setaffinity_np(pthread_self(), sizeof(cpus), &cpus) && options.control) {
            return false;
        }
    }
    if (options.control && !options.mock) {
        sched_param scheduling{};
        scheduling.sched_priority = options.priority;
        rt_ready = pthread_setschedparam(pthread_self(), SCHED_FIFO, &scheduling) == 0;
        if (!rt_ready) return false;
    }
    return true;
}

uint64_t wait_for_motor_tick(uint64_t& next_ns, LoopStatistics& statistics, uint64_t& lateness_ns) {
    next_ns += MotorPeriodNs;
    timespec wake{time_t(next_ns / 1'000'000'000), long(next_ns % 1'000'000'000)};
    while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &wake, nullptr) == EINTR) {}
    const uint64_t now = now_ns();
    lateness_ns = now > next_ns ? now - next_ns : 0;
    statistics.max_lateness_ns = std::max(statistics.max_lateness_ns, lateness_ns);
    statistics.lateness_histogram[std::min<uint64_t>(lateness_ns / 1000, MaxLatenessUs)]++;
    if (lateness_ns >= MotorPeriodNs) {
        const auto skipped = lateness_ns / MotorPeriodNs;
        statistics.missed_ticks += skipped;
        next_ns += skipped * MotorPeriodNs;
    }
    return now;
}

void update_mock_feedback(MotorRuntime& runtime, const rl::Profile& profile, uint64_t now) {
    auto& state = runtime.feedback;
    if (runtime.engaged) {
        for (unsigned i = 0; i < MotorCount; ++i) {
            const auto& joint = motor_command(runtime.output, i);
            if (profile.wheel[i]) {
                state.dq[i] = joint.v_des;
                state.q[i] += state.dq[i] * MotorPeriodSeconds;
            } else {
                const float change = std::clamp(joint.p_des - state.q[i], -0.004f, 0.004f);
                state.q[i] += change;
                state.dq[i] = change / MotorPeriodSeconds;
            }
        }
    }
    for (auto& flags : state.motor_flags) flags = 1;
    runtime.feedback_at_ns = runtime.imu_at_ns = now;
    state.h.sequence++;
    state.imu_stamp = now;
}

void read_physical_feedback(MotorRuntime& runtime, spline_data_p spline, nav_imu_p imu,
                            bool control, uint64_t now) {
    // ABI has no seqlock/counter. Equal copies mitigate torn reads, not eliminate them.
    const uint32_t available =
        __atomic_load_n(&spline->state.consumer_flags[CONSUMER_CONTROL], __ATOMIC_ACQUIRE);
    if (available) {
        spline_state_data_t first{}, second{};
        std::memcpy(&first, &spline->state, sizeof(first));
        std::memcpy(&second, &spline->state, sizeof(second));
        if (!std::memcmp(first.legs, second.legs, sizeof(first.legs))) {
            for (unsigned i = 0; i < MotorCount; ++i) {
                const auto& joint = motor_feedback(first, i);
                runtime.feedback.q[i] = joint.p;
                runtime.feedback.dq[i] = joint.v;
                runtime.feedback.torque[i] = joint.t;
                runtime.feedback.motor_flags[i] = joint.flags;
            }
            if (control) {
                __atomic_store_n(&spline->state.consumer_flags[CONSUMER_CONTROL], 0u,
                                 __ATOMIC_RELEASE);
            }
            runtime.feedback_at_ns = now;
            runtime.feedback.h.sequence++;
        }
    }
    nav_imu_t first{}, second{};
    std::memcpy(&first, imu, sizeof(first));
    std::memcpy(&second, imu, sizeof(second));
    if (first.timestamp && first.timestamp == second.timestamp &&
        !std::memcmp(&first, &second, sizeof(first)) && first.timestamp != runtime.last_imu_stamp) {
        // Preserve firmware order. NX explicitly configures xyzw/wxyz.
        std::memcpy(runtime.feedback.quaternion, first.q, sizeof(first.q));
        std::memcpy(runtime.feedback.gyro, first.gyro, sizeof(first.gyro));
        std::memcpy(runtime.feedback.acceleration, first.acc, sizeof(first.acc));
        runtime.feedback.imu_stamp = first.timestamp;
        runtime.last_imu_stamp = first.timestamp;
        runtime.imu_at_ns = now;
    }
}

FeedbackStatus validate_feedback(const MotorRuntime& runtime, uint64_t now) {
    FeedbackStatus status;
    const auto& state = runtime.feedback;
    for (unsigned i = 0; i < MotorCount; ++i) {
        if (!std::isfinite(state.q[i]) || !std::isfinite(state.dq[i]) ||
            !std::isfinite(state.torque[i]) || !(state.motor_flags[i] & 1) ||
            (state.motor_flags[i] & 0x3e)) {
            status.motors = false;
        }
    }
    float norm = 0;
    for (float value : state.quaternion) {
        if (!std::isfinite(value)) status.imu = false;
        norm += value * value;
    }
    for (float value : state.gyro) {
        if (!std::isfinite(value)) status.imu = false;
    }
    if (norm < 0.5f || norm > 1.5f || now - runtime.imu_at_ns > ImuFeedbackTimeoutNs) {
        status.imu = false;
    }
    if (now - runtime.feedback_at_ns > MotorFeedbackTimeoutNs) status.motors = false;
    return status;
}

void accept_latest_command(SharedState& shared, MotorRuntime& runtime, const Options& options,
                           const rl::Profile& profile, FeedbackStatus feedback, uint64_t now) {
    // RT code only tries the lock; it never waits for the network thread.
    std::unique_lock<std::mutex> lock(shared.mutex, std::try_to_lock);
    if (!lock || shared.command_received_ns <= runtime.seen_received_ns) return;
    runtime.seen_received_ns = shared.command_received_ns;
    const auto candidate = shared.command;
    const bool new_owner = runtime.claimed_session && runtime.claimed_session != candidate.h.session;
    const bool takeover = new_owner && candidate.mode == rl::Passive &&
        rl::age_ns(now, runtime.received_ns) > uint64_t(runtime.current.ttl_us) * 1000;
    const bool valid =
        rl::valid(candidate, profile) && candidate.h.flags == runtime.feedback.profile_crc &&
        candidate.state_session == runtime.feedback.h.session &&
        rl::valid_state_reference(runtime.feedback.h.sequence, candidate.state_sequence) &&
        (!new_owner || takeover) &&
        (takeover || candidate.h.sequence > runtime.accepted_sequence) && feedback.motors && feedback.imu &&
        !runtime.hard_fault && (!runtime.shutting_down || candidate.mode != rl::Active) &&
        (!runtime.exit_after_shutdown || candidate.mode == rl::Shutdown);
    if (!valid) {
        shared.rejected++;
        return;
    }

    // A new session starts passive; pose and speed no longer gate activation.
    if (!runtime.engaged && candidate.mode != rl::Passive) {
        shared.rejected++;
        return;
    }

    if (!runtime.engaged) {
        for (unsigned i = 0; i < MotorCount; ++i) {
            motor_command(runtime.output, i).p_des = runtime.feedback.q[i];
        }
    }
    runtime.current = candidate;
    runtime.command_hold = false;
    runtime.received_ns = shared.command_received_ns;
    runtime.accepted_sequence = candidate.h.sequence;
    runtime.claimed_session = candidate.h.session;
    if (!runtime.engaged && candidate.mode == rl::Passive) {
        runtime.passive = true;
        runtime.passive_started_ns = now;
    }
    runtime.engaged = options.control;
    if (runtime.passive && candidate.mode == rl::Active) {
        runtime.passive = false;
        runtime.passive_started_ns = 0;
    }
}

void begin_shutdown(MotorRuntime& runtime, uint64_t now) {
    runtime.shutting_down = true;
    runtime.shutdown_started_ns = now;
    runtime.shutdown_stable_ticks = 0;
    for (unsigned i = 0; i < MotorCount; ++i) {
        const auto& joint = motor_command(runtime.output, i);
        runtime.shutdown_position[i] = joint.p_des;
        runtime.shutdown_kp[i] = joint.kp;
        runtime.shutdown_kd[i] = joint.kd;
        runtime.shutdown_velocity[i] = joint.v_des;
        runtime.shutdown_feedforward[i] = joint.t_ff;
    }
}

void update_shutdown_state(MotorRuntime& runtime, const SharedState& shared,
                           FeedbackStatus feedback, uint64_t now, uint64_t lateness_ns,
                           uint64_t command_age_ns, bool duration_elapsed) {
    const bool expired = runtime.engaged &&
        command_age_ns > uint64_t(runtime.current.ttl_us) * 1000;
    if (runtime.engaged) {
        if (shared.fault.load()) {
            set_hard_fault(runtime, shared_fault_name(shared), now, lateness_ns);
        } else if (!feedback.motors) {
            set_hard_fault(runtime,
                rl::age_ns(now, runtime.feedback_at_ns) > MotorFeedbackTimeoutNs ?
                    "motor_feedback_timeout" : "motor_feedback_invalid_or_disabled",
                now, lateness_ns);
        } else if (!feedback.imu) {
            set_hard_fault(runtime,
                rl::age_ns(now, runtime.imu_at_ns) > ImuFeedbackTimeoutNs ?
                    "imu_feedback_timeout" : "imu_feedback_invalid",
                now, lateness_ns);
        } else if (lateness_ns >= HardLatenessNs) {
            set_hard_fault(runtime, "motor_thread_lateness", now, lateness_ns);
        }
    }
    // A lost NX stream pauses motion in place, without running the lie trajectory.
    // Capture once: do not keep following a sagging measured pose on each tick.
    if (expired && !runtime.command_hold && !runtime.passive &&
        !runtime.shutting_down && !runtime.hard_fault && runtime.current.mode == rl::Active) {
        for (unsigned i = 0; i < MotorCount; ++i) {
            runtime.current.joints[i][0] = runtime.feedback.q[i];
            runtime.current.joints[i][1] = 0;
            runtime.current.joints[i][4] = 0;
        }
        runtime.command_hold = true;
        ++runtime.command_hold_events;
    }
    const bool terminate = requested_stop || duration_elapsed || runtime.hard_fault ||
                           runtime.current.mode == rl::Shutdown;
    if (runtime.engaged && terminate) runtime.exit_after_shutdown = true;
    const bool passive_requested = !runtime.passive && runtime.current.mode == rl::Passive;
    if (runtime.engaged && !runtime.shutting_down && (terminate || passive_requested) &&
        (!runtime.passive || runtime.hard_fault)) {
        begin_shutdown(runtime, now);
    }
}

void build_motor_output(MotorRuntime& runtime, const rl::Profile& profile, float shutdown_age) {
    for (unsigned i = 0; i < MotorCount; ++i) {
        auto& joint = motor_command(runtime.output, i);
        if (runtime.passive && !runtime.hard_fault) {
            joint.p_des = profile.wheel[i] ? 0 : runtime.feedback.q[i];
            joint.v_des = 0;
            joint.kp = 0;
            joint.kd = profile.passive_kd[i];
            joint.t_ff = 0;
        } else if (!runtime.shutting_down) {
            joint.p_des = profile.wheel[i] ? 0 :
                std::clamp(runtime.current.joints[i][0], profile.lower[i], profile.upper[i]);
            joint.v_des = runtime.current.joints[i][1];
            joint.kp = runtime.current.joints[i][2];
            joint.kd = runtime.current.joints[i][3];
            joint.t_ff = runtime.current.joints[i][4];
        } else if (runtime.hard_fault) {
            // Bad feedback cannot support a blind descent trajectory.
            joint.p_des = 0;
            joint.v_des = 0;
            joint.kp = 0;
            joint.kd = profile.passive_kd[i];
            joint.t_ff = 0;
        } else {
            const float ratio = std::clamp(
                (shutdown_age - ShutdownGainSeconds) / ShutdownDescentSeconds, 0.0f, 1.0f);
            const float blend = ratio * ratio * (3 - 2 * ratio);
            const float desired = runtime.shutdown_position[i] +
                blend * (profile.shutdown_q[i] - runtime.shutdown_position[i]);
            joint.p_des = profile.wheel[i] ? 0 :
                std::clamp(desired, profile.lower[i], profile.upper[i]);
            const float gain_ratio = std::clamp(shutdown_age / ShutdownGainSeconds, 0.0f, 1.0f);
            const float gain_blend = gain_ratio * gain_ratio * (3 - 2 * gain_ratio);
            joint.v_des = (1 - gain_blend) * runtime.shutdown_velocity[i];
            joint.kp = runtime.shutdown_kp[i] +
                gain_blend * (profile.hold_kp[i] - runtime.shutdown_kp[i]);
            joint.kd = runtime.shutdown_kd[i] +
                gain_blend * (profile.hold_kd[i] - runtime.shutdown_kd[i]);
            joint.t_ff = (1 - gain_blend) * runtime.shutdown_feedforward[i];
            if (runtime.passive_started_ns) {
                joint.kp = 0;
                joint.kd = profile.passive_kd[i];
            }
        }
        runtime.output.legs[i / 4].flags = 1;
    }
}

void update_descent_completion(MotorRuntime& runtime, const rl::Profile& profile,
                               uint64_t now, float shutdown_age) {
    if (!runtime.shutting_down || runtime.hard_fault ||
        shutdown_age < ShutdownTrajectorySeconds || runtime.passive_started_ns) {
        return;
    }
    bool at_rest = true;
    for (unsigned i = 0; i < MotorCount; ++i) {
        if (!profile.wheel[i] &&
            (std::fabs(runtime.feedback.q[i] - profile.shutdown_q[i]) > RestPositionTolerance ||
             std::fabs(runtime.feedback.dq[i]) > RestVelocityTolerance)) {
            at_rest = false;
        }
    }
    runtime.shutdown_stable_ticks = at_rest ? runtime.shutdown_stable_ticks + 1 : 0;
    if (runtime.shutdown_stable_ticks >= RestStableTicks) {
        runtime.passive_started_ns = now;
        runtime.passive = true;
        runtime.shutting_down = false;
    }
    // Until feedback settles, retain support gains instead of switching on a timer.
}

void publish_motor_output(SharedState& shared, MotorRuntime& runtime,
                           spline_data_p spline, uint64_t now) {
    if (shared.fault.load()) return; // Never compete with a restarted vendor controller.
    const auto gap = rl::age_ns(now, runtime.last_consumed_ns);
    if (__atomic_load_n(&spline->cmd.consumer_flags[CONSUMER_CONTROL], __ATOMIC_ACQUIRE) == 0) {
        if (runtime.publication_busy) {
            // Measure a completed occupied-slot interval once, including its
            // recovery tick. Initial idle time before engagement is excluded.
            runtime.max_publication_gap_ns = std::max(runtime.max_publication_gap_ns, gap);
            ++runtime.publication_recoveries;
            if (runtime.publication_warned) ++runtime.publication_warning_recoveries;
            unsigned bucket = 0;
            while (bucket < PublicationGapBucketsNs.size() && gap > PublicationGapBucketsNs[bucket]) {
                ++bucket;
            }
            ++runtime.publication_recovery_histogram[bucket];
        }
        runtime.publication_busy = runtime.publication_warned = false;
        runtime.last_consumed_ns = now;
        std::memcpy(spline->cmd.legs, runtime.output.legs, sizeof(runtime.output.legs));
        __atomic_store_n(&spline->cmd.consumer_flags[CONSUMER_CONTROL], 1u, __ATOMIC_RELEASE);
    } else {
        shared.publication_skips++;
        if (!runtime.publication_busy) {
            runtime.publication_busy = true;
            ++runtime.publication_busy_periods;
        }
        runtime.max_publication_gap_ns = std::max(runtime.max_publication_gap_ns, gap);
        if (gap > PublicationWarningNs && !runtime.publication_warned) {
            runtime.publication_warned = true;
            ++runtime.publication_warning_events;
            if (rl::age_ns(now, runtime.feedback_at_ns) > PublicationWarningNs) {
                ++runtime.publication_warnings_with_motor_gap;
            }
        }
        if (gap > PublicationTimeoutNs) {
            set_hard_fault(runtime, "spline_command_not_consumed", now);
        }
    }
}

bool shutdown_complete(const MotorRuntime& runtime, uint64_t now, float shutdown_age) {
    return (runtime.hard_fault && runtime.shutting_down && shutdown_age >= EmergencyExitSeconds) ||
           (runtime.exit_after_shutdown && runtime.passive &&
            now - runtime.passive_started_ns >= PassiveExitDelayNs);
}

// Retire the command session, not the process. A changed bridge token invalidates
// every queued command; the next session must start with a fresh Passive handshake.
bool retire_session(MotorRuntime& runtime, SharedState& shared, FeedbackStatus feedback,
                    uint64_t now, float shutdown_age, uint64_t lateness_ns = 0) {
    if (!shutdown_complete(runtime, now, shutdown_age)) return false;
    if (runtime.hard_fault) {
        runtime.recovery_stable_ticks = feedback.motors && feedback.imu && !shared.fault.load() &&
            lateness_ns < HardLatenessNs
            ? runtime.recovery_stable_ticks + 1 : 0;
        if (runtime.recovery_stable_ticks < RestStableTicks) return false;
    }
    std::unique_lock<std::mutex> lock(shared.mutex, std::try_to_lock);
    if (!lock) return false;
    runtime.engaged = runtime.shutting_down = runtime.hard_fault = runtime.passive = false;
    runtime.command_hold = false;
    runtime.exit_after_shutdown = false;
    runtime.received_ns = runtime.accepted_sequence = runtime.claimed_session = 0;
    runtime.shutdown_started_ns = runtime.passive_started_ns = 0;
    runtime.shutdown_stable_ticks = runtime.recovery_stable_ticks = 0;
    runtime.seen_received_ns = shared.command_received_ns;
    runtime.current = rl::CommandPacket{};
    runtime.output = spline_cmd_data_t{};
    runtime.last_consumed_ns = now;
    runtime.publication_busy = runtime.publication_warned = false;
    ++runtime.completed_sessions;
    ++runtime.feedback.h.session;
    if (!runtime.feedback.h.session) ++runtime.feedback.h.session;
    runtime.feedback.accepted_command = 0;
    shared.command_received_ns = 0;
    shared.command = rl::CommandPacket{};
    return true;
}

void publish_state_snapshot(SharedState& shared, MotorRuntime& runtime, const Options& options,
                            FeedbackStatus feedback, bool rt_ready, const LoopStatistics& statistics,
                            uint64_t now, uint64_t lateness_ns, uint64_t command_age_ns) {
    auto& state = runtime.feedback;
    const bool descent_timed_out = runtime.shutting_down &&
        double(now - runtime.shutdown_started_ns) / 1e9 > ShutdownFaultSeconds;
    if (descent_timed_out) record_first_fault(runtime, "descent_tracking_timeout", now);
    state.h.flags =
        (feedback.motors ? uint32_t(rl::MotorsValid) : 0u) |
        (feedback.imu ? uint32_t(rl::ImuValid) : 0u) |
        (options.control ? uint32_t(rl::Control) : 0u) |
        (options.mock ? uint32_t(rl::Mock) : 0u) |
        (runtime.hard_fault || shared.fault.load() || descent_timed_out ? uint32_t(rl::Fault) : 0u) |
        (rt_ready ? uint32_t(rl::RtReady) : 0u) |
        (runtime.passive && !runtime.shutting_down ? uint32_t(rl::PassiveReady) : 0u) |
        (runtime.shutting_down ? uint32_t(rl::Descending) : 0u);
    if (runtime.command_hold) state.h.flags |= rl::Holding;
    state.h.timestamp_ns = now;
    state.accepted_command = runtime.accepted_sequence;
    state.command_age_ms = runtime.received_ns ? float(command_age_ns) / 1e6f : 0;
    state.feedback_age_ms = runtime.feedback_at_ns ? float(now - runtime.feedback_at_ns) / 1e6f : 99999;
    state.lateness_us = float(lateness_ns) / 1000;
    state.missed_ticks = uint32_t(statistics.missed_ticks);
    std::unique_lock<std::mutex> lock(shared.mutex, std::try_to_lock);
    if (lock) shared.state = state;
    else shared.skipped_snapshots++;
}

void write_report(const SharedState& shared, const MotorRuntime& runtime,
                   const LoopStatistics& statistics, const Options& options) {
    std::ostringstream report;
    report.precision(12);
    report << "{\"mock\":" << (options.mock ? "true" : "false")
           << ",\"control\":" << (options.control ? "true" : "false")
           << ",\"ticks\":" << runtime.ticks
           << ",\"completed_sessions\":" << runtime.completed_sessions
           << ",\"command_hold\":" << (runtime.command_hold ? "true" : "false")
           << ",\"command_hold_events\":" << runtime.command_hold_events
           << ",\"missed_ticks\":" << statistics.missed_ticks
           << ",\"max_lateness_us\":" << statistics.max_lateness_ns / 1000
           << ",\"p99_lateness_us\":" << statistics.percentile(.99)
           << ",\"p999_lateness_us\":" << statistics.percentile(.999)
           << ",\"rejected_commands\":" << shared.rejected.load()
           << ",\"publication_skips\":" << shared.publication_skips.load()
           << ",\"max_publication_gap_ms\":" << double(runtime.max_publication_gap_ns) / 1e6
           << ",\"publication_warning_ms\":" << double(PublicationWarningNs) / 1e6
           << ",\"publication_timeout_ms\":" << double(PublicationTimeoutNs) / 1e6
           << ",\"publication_busy_periods\":" << runtime.publication_busy_periods
           << ",\"publication_recoveries\":" << runtime.publication_recoveries
           << ",\"publication_warning_events\":" << runtime.publication_warning_events
           << ",\"publication_warning_recoveries\":" << runtime.publication_warning_recoveries
           << ",\"publication_warnings_with_motor_gap\":" << runtime.publication_warnings_with_motor_gap
           << ",\"publication_recovery_gap_ms\":{\"upper_bounds\":[2,5,10,15,20,null],\"counts\":[";
    for (unsigned i = 0; i < runtime.publication_recovery_histogram.size(); ++i) {
        if (i) report << ',';
        report << runtime.publication_recovery_histogram[i];
    }
    report << "]}"
           << ",\"max_motor_feedback_age_ms\":" << double(runtime.max_motor_age_ns) / 1e6
           << ",\"max_imu_feedback_age_ms\":" << double(runtime.max_imu_age_ns) / 1e6
           << ",\"fault\":" << (runtime.hard_fault || shared.fault.load() ||
                                    runtime.fault_diagnostic.reason ? "true" : "false")
           << ",\"fault_reason\":";
    const auto& diagnostic = runtime.fault_diagnostic;
    if (diagnostic.reason || shared.fault.load()) {
        report << '"' << (diagnostic.reason ? diagnostic.reason : shared_fault_name(shared)) << '"';
    } else {
        report << "null";
    }
    report << ",\"first_fault\":";
    if (diagnostic.reason) {
        report << "{\"tick\":" << diagnostic.tick
               << ",\"uptime_ms\":" << double(rl::age_ns(diagnostic.at_ns, runtime.started_ns)) / 1e6
               << ",\"motor_feedback_age_ms\":" << double(diagnostic.motor_age_ns) / 1e6
               << ",\"imu_feedback_age_ms\":" << double(diagnostic.imu_age_ns) / 1e6
               << ",\"publication_gap_ms\":" << double(diagnostic.publication_gap_ns) / 1e6
               << ",\"lateness_us\":" << double(diagnostic.lateness_ns) / 1e3
               << ",\"motor_flags\":[";
        for (unsigned i = 0; i < MotorCount; ++i) {
            if (i) report << ',';
            report << diagnostic.motor_flags[i];
        }
        report << "]}";
    } else {
        report << "null";
    }
    report << '}';
    if (!options.report.empty()) {
        std::ofstream file(options.report);
        file << report.str() << '\n';
    }
    std::cout << report.str() << std::endl;
}

void motor_loop(SharedState& shared, const Options& options, spline_data_p spline,
                 nav_imu_p imu, const rl::Profile& profile, uint64_t boot) {
    bool rt_ready = false;
    if (!configure_motor_thread(options, rt_ready)) {
        set_shared_fault(shared, SharedFailure::RtSetup);
        shared.failed = true;
        shared.done = true;
        write_report(shared, MotorRuntime{}, LoopStatistics{}, options);
        return;
    }
    MotorRuntime runtime;
    LoopStatistics statistics;
    runtime.feedback.h.kind = rl::StateKind;
    runtime.feedback.h.bytes = sizeof(runtime.feedback);
    runtime.feedback.h.session = boot;
    runtime.feedback.profile_crc = rl::profile_crc(profile);
    runtime.started_ns = runtime.last_consumed_ns = now_ns();
    uint64_t next_ns = runtime.started_ns;

    // Fixed-size work only: no heap allocation, socket I/O, filesystem access or blocking locks.
    while (true) {
        uint64_t lateness_ns = 0;
        const uint64_t now = wait_for_motor_tick(next_ns, statistics, lateness_ns);
        if (options.mock) update_mock_feedback(runtime, profile, now);
        else read_physical_feedback(runtime, spline, imu, options.control, now);
        const auto feedback = validate_feedback(runtime, now);
        if (runtime.engaged) {
            runtime.max_motor_age_ns = std::max(runtime.max_motor_age_ns,
                rl::age_ns(now, runtime.feedback_at_ns));
            runtime.max_imu_age_ns = std::max(runtime.max_imu_age_ns,
                rl::age_ns(now, runtime.imu_at_ns));
        }
        accept_latest_command(shared, runtime, options, profile, feedback, now);

        const bool duration_elapsed = options.duration > 0 &&
            double(now - runtime.started_ns) / 1e9 >= options.duration;
        // A receiver may timestamp a packet after this tick sampled 'now'.
        const uint64_t command_age_ns = rl::age_ns(now, runtime.received_ns);
        // Mock has no FIFO scheduling guarantee: measure host lateness without
        // treating desktop preemption as a physical motor-thread failure.
        update_shutdown_state(runtime, shared, feedback, now,
                              options.mock ? 0 : lateness_ns, command_age_ns, duration_elapsed);

        if (runtime.engaged) {
            const float age = runtime.shutting_down ? float(now - runtime.shutdown_started_ns) / 1e9f : 0;
            build_motor_output(runtime, profile, age);
            update_descent_completion(runtime, profile, now, age);
            // Emergency output is bounded; while blocked afterwards, retain
            // the process/feedback loop without continuing blind writes.
            if (!options.mock && !(runtime.hard_fault && age >= EmergencyExitSeconds)) {
                publish_motor_output(shared, runtime, spline, now);
            }
            if (shutdown_complete(runtime, now, age)) {
                if (requested_stop || duration_elapsed) break;
                retire_session(runtime, shared, feedback, now, age, lateness_ns);
            }
        } else if (requested_stop || duration_elapsed) {
            break;
        }
        publish_state_snapshot(shared, runtime, options, feedback, rt_ready, statistics,
                               now, lateness_ns, command_age_ns);
        ++runtime.ticks;
    }
    shared.failed = runtime.hard_fault || shared.fault.load();
    shared.done = true;
    sched_param normal{};
    pthread_setschedparam(pthread_self(), SCHED_OTHER, &normal);
    write_report(shared, runtime, statistics, options);
}

// Filesystem scans must never stall command reception or state publication.
void ownership_watchdog(SharedState& shared,
                        const std::function<bool()>& check = vendor_controller_running) {
    while (!shared.done.load()) {
        try {
            if (check()) {
                set_shared_fault(shared, SharedFailure::VendorController);
            } else if (shared.fault_reason.load() == SharedFailure::VendorController) {
                shared.fault_reason = SharedFailure::None;
                shared.fault = false;
            }
        } catch (...) {
            // An unreadable process table cannot establish exclusive ownership.
            set_shared_fault(shared, SharedFailure::OwnershipCheck);
            return;
        }
        // Only this worker waits; neither the socket nor the motor thread does.
        for (unsigned i = 0; i < 10 && !shared.done.load(); ++i)
            std::this_thread::sleep_for(std::chrono::nanoseconds(OwnershipCheckPeriodNs / 10));
    }
}

// The non-RT network thread owns only socket I/O and session validation.
struct NetworkSession {
    uint64_t highest_sequence = 0;
    uint64_t session = 0;
    uint64_t boot = 0;
};

void receive_commands(int socket, const sockaddr_in& peer, SharedState& shared,
                       const rl::Profile& profile, uint32_t profile_crc,
                       uint64_t boot, NetworkSession& session) {
    for (unsigned batch = 0; batch < ReceiveBatchLimit; ++batch) {
        rl::CommandPacket command{};
        sockaddr_in sender{};
        socklen_t length = sizeof(sender);
        const auto bytes = recvfrom(socket, &command, sizeof(command), MSG_TRUNC,
                                    reinterpret_cast<sockaddr*>(&sender), &length);
        if (bytes < 0) break;
        if (bytes != sizeof(command) || sender.sin_addr.s_addr != peer.sin_addr.s_addr ||
            !rl::valid(command, profile) || command.h.flags != profile_crc ||
            command.state_session != boot ||
            (!session.session && command.mode != rl::Passive) ||
            (session.session == command.h.session && command.h.sequence <= session.highest_sequence)) {
            shared.rejected++;
            continue;
        }
        std::lock_guard<std::mutex> lock(shared.mutex);
        // The motor thread can retire a session while the network thread runs.
        if (shared.state.h.session != boot) {
            shared.rejected++;
            continue;
        }
        const bool takeover_allowed = command.mode == rl::Passive &&
            shared.state.command_age_ms > 100 &&
            (shared.state.h.flags & (rl::Holding | rl::PassiveReady)) &&
            !(shared.state.h.flags & (rl::Fault | rl::Descending)) &&
            shared.command.mode != rl::Shutdown;
        if (session.session && session.session != command.h.session && !takeover_allowed) {
            shared.rejected++;
            continue;
        }
        if (!rl::valid_state_reference(shared.state.h.sequence, command.state_sequence)) {
            shared.rejected++;
            continue;
        }
        // Match the motor-thread takeover rule BEFORE reserving a client.
        // A late Shutdown from a retired NX must not claim an idle bridge:
        // the motor thread would reject it without engaging its TTL cleanup.
        session.session = command.h.session;
        session.highest_sequence = command.h.sequence;
        shared.command = command;
        shared.command_received_ns = now_ns();
    }
}

void send_state(int socket, const sockaddr_in& peer, SharedState& shared) {
    rl::StatePacket state;
    {
        std::lock_guard<std::mutex> lock(shared.mutex);
        state = shared.state;
    }
    if (state.h.sequence) {
        sendto(socket, &state, sizeof(state), 0,
               reinterpret_cast<const sockaddr*>(&peer), sizeof(peer));
    }
}

void network_loop(int socket, const sockaddr_in& peer, SharedState& shared,
                   const rl::Profile& profile, uint64_t boot) {
    NetworkSession session;
    session.boot = boot;
    const uint32_t profile_crc = rl::profile_crc(profile);
    uint64_t next_state_ns = 0;
    while (!shared.done.load()) {
        pollfd descriptor{socket, POLLIN, 0};
        poll(&descriptor, 1, 1);
        {
            std::lock_guard<std::mutex> lock(shared.mutex);
            if (shared.state.h.session && shared.state.h.session != session.boot) {
                session = NetworkSession{};
                session.boot = shared.state.h.session;
            }
        }
        receive_commands(socket, peer, shared, profile, profile_crc, session.boot, session);
        const uint64_t now = now_ns();
        if (now >= next_state_ns) {
            next_state_ns = now + StatePeriodNs;
            send_state(socket, peer, shared);
        }
    }
}

int run_bridge(const Options& options) {
    const uint16_t endian = 1;
    if (*reinterpret_cast<const uint8_t*>(&endian) != 1) {
        throw std::runtime_error("little-endian host required");
    }
    auto profile = make_mock_profile();
    if (options.control && !options.mock) {
        profile = load_profile(options.profile);
        if (vendor_controller_running()) {
            throw std::runtime_error("mc_ctrl is running: use robot-launch stop motion_control first");
        }
    }
    std::unique_ptr<MappedRegion> motor, imu;
    if (!options.mock) {
        motor = std::make_unique<MappedRegion>(options.shm, sizeof(spline_data_t), options.control);
        imu = std::make_unique<MappedRegion>(options.imu, sizeof(nav_imu_t), false);
    }
    UdpSocket socket(options);
    sockaddr_in peer{};
    peer.sin_family = AF_INET;
    peer.sin_port = htons(options.port + 1);
    if (inet_pton(AF_INET, options.peer.c_str(), &peer.sin_addr) != 1) {
        throw std::runtime_error("invalid peer IPv4");
    }
    if (options.control && !options.mock && mlockall(MCL_CURRENT | MCL_FUTURE)) {
        throw std::runtime_error("mlockall failed; refusing control");
    }
    signal(SIGINT, request_stop);
    signal(SIGTERM, request_stop);
    signal(SIGHUP, request_stop);
    SharedState shared;
    const uint64_t boot = now_ns() ^ uint64_t(getpid());
    std::thread motor_thread(motor_loop, std::ref(shared), std::cref(options),
                             motor ? static_cast<spline_data_p>(motor->data()) : nullptr,
                             imu ? static_cast<nav_imu_p>(imu->data()) : nullptr,
                             std::cref(profile), boot);
    std::thread ownership_thread;
    try {
        if (options.control && !options.mock)
            ownership_thread = std::thread([&shared] { ownership_watchdog(shared); });
        network_loop(socket.fd(), peer, shared, profile, boot);
    } catch (...) {
        set_shared_fault(shared, SharedFailure::Network);
        requested_stop = 1; // No network supervisor remains to service resident recovery.
        motor_thread.join();
        if (ownership_thread.joinable()) ownership_thread.join();
        throw;
    }
    motor_thread.join();
    if (ownership_thread.joinable()) ownership_thread.join();
    return shared.failed.load() ? 1 : 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 3 && (std::string(argv[1]) == "--validate-profile" ||
                          std::string(argv[1]) == "--preflight-control")) {
            const auto profile = load_profile(argv[2]);
            if (std::string(argv[1]) == "--preflight-control") check_motor_feedback();
            std::cout << "profile validated; CRC32=" << rl::profile_crc(profile) << '\n';
            return 0;
        }
        return run_bridge(parse_options(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "bridge failed: " << error.what() << '\n';
        return 1;
    }
}
