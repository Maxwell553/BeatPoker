// macOS compatibility for glibc's reentrant 48-bit LCG interface.
// State is per caller, preserving the POSIX drand48 sequence and seed mapping.
#pragma once
#include <cstdint>
#if defined(__APPLE__)
struct drand48_data { std::uint64_t state = 0x1234abcd330eULL; };
inline int srand48_r(long seed, drand48_data *buffer) {
    buffer->state = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(seed)) << 16) | 0x330e;
    return 0;
}
inline int drand48_r(drand48_data *buffer, double *value) {
    buffer->state = (0x5deece66dULL * buffer->state + 0xbULL) & ((1ULL << 48) - 1);
    *value = static_cast<double>(buffer->state) / static_cast<double>(1ULL << 48);
    return 0;
}
#endif
