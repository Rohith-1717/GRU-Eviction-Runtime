#pragma once

#include <cstdint>
#include <cstddef>

using u64 = uint64_t;
using u32 = uint32_t;
using u16 = uint16_t;
using u8 = uint8_t;

constexpr size_t PAGE_SIZE = 4096;
constexpr size_t NUM_FRAMES = 16;
constexpr u32 SWAP_SLOT_INVALID = static_cast<u32>(-1);