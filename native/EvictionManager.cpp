#include "EvictionManager.hpp"
#include <cassert>

// this is for LRU AND CLOCK
// Learned eviction is in MemoryManager.cpp since it needs to interact with the predictor

EvictionManager::EvictionManager(EvictionPolicy policy_) : policy(policy_), clockHand(0){}

void EvictionManager::touch(u64 vpn, PageMeta& entry, u64 globalTime){
    if (policy == EvictionPolicy::LRU){
        entry.lastAccess = globalTime;
    }
    entry.reference = true;
}

u64 EvictionManager::choose_victim(PageTable& table){
    size_t num_pages = table.size();
    assert(num_pages > 0);
    if (policy == EvictionPolicy::LRU){
        u64 victim = SWAP_SLOT_INVALID;
        u64 best_time = UINT64_MAX;
        for (u64 vpn = 0; vpn < num_pages; ++vpn){
            const auto& entry = table.getEntry(vpn);
            if (!entry.resident){
                continue;
            }
            if (entry.lastAccess < best_time){
                best_time = entry.lastAccess;
                victim = vpn;
            }
        }
        assert(victim != SWAP_SLOT_INVALID);
        return victim;
    }

    for (size_t scanned = 0; scanned < num_pages; ++scanned){
        u64 vpn = (clockHand + scanned) % num_pages;
        const auto& entry = table.getEntry(vpn);
        if (!entry.resident){
            continue;
        }
        if (!entry.reference){
            clockHand = (vpn + 1) % num_pages;
            return vpn;
        }
        // clear second chance
        const_cast<PageTable&>(table).clearReference(vpn);
    }
    for (u64 vpn = 0; vpn < num_pages; ++vpn){
        const auto& entry = table.getEntry(vpn);
        if (entry.resident){
            clockHand = (vpn + 1) % num_pages;
            return vpn;
        }
    }
    assert(false);
    return 0;
}