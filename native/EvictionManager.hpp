#pragma once

#include "Common.hpp"
#include "PageTable.hpp"

enum class EvictionPolicy{
    LRU,
    CLOCK,
};

class EvictionManager{
public:
    explicit EvictionManager(EvictionPolicy policy = EvictionPolicy::LRU);
    void touch(u64 vpn, PageMeta& entry, u64 globalTime);
    u64 choose_victim(PageTable& table);

private:
    EvictionPolicy policy;
    u64 clockHand;
};