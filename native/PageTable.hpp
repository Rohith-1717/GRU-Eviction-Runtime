#pragma once

#include "Common.hpp"
#include <vector>
#include <cassert>

struct PageMeta{
    bool resident = false;
    bool dirty = false;
    bool reference = false;
    u32 bufferSlot = 0;
    u32 swapSlot = SWAP_SLOT_INVALID;
    u64 lastAccess = 0;
    u64 previousAccess = 0;
    u64 frequency = 0;
    u64 loadTimestamp = 0;
    float predictedReuse = 0.0f;
};

class PageTable{
public:
    PageTable() = default;
    explicit PageTable(size_t num_pages);

    void resize(size_t num_pages);
    size_t size() const;
    PageMeta& getEntry(u64 vpn);
    const PageMeta& getEntry(u64 vpn) const;
    void updateEntry(u64 vpn, const PageMeta& entry);
    void clearReference(u64 vpn);

private:
    std::vector<PageMeta> entries;
};
