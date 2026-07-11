#pragma once

#include "Common.hpp"
#include "PageTable.hpp"
#include "SwapManager.hpp"
#include "EvictionManager.hpp"
#include "runtime_gru.hpp"

#include <vector>
#include <chrono>

class MemoryManager{
public:
    explicit MemoryManager(EvictionPolicy policy = EvictionPolicy::LRU, bool learned = true);
    ~MemoryManager();

    void initPageTable(size_t num_pages);
    void* bufferData(u32 bufferIndex);
    PageTable& pageTbl();
    void touchPage(u64 vpn);    
    void loadPage(u64 vpn);
    void evictPage(u64 vpn);
    u64 chooseLearnedVictim();
    float computeLearnedScore(u64 vpn);
    void addFaultLatencyNs(uint64_t ns);
    uint64_t faultLatencyNs() const;
    uint64_t swapWriteLatencyNs() const;
    uint64_t swapReadLatencyNs() const;
    uint64_t learnedEvictionCount() const;
    bool learnedEvictionActive() const;
    bool hasFreeResidentSlot() const;
    u32 residentPageCount() const;

private:
    PageTable pageTbl_;
    std::vector<u8> bufferPool;
    SwapManager swapMgr;
    EvictionManager eviction;
    RuntimeGRU learnedPredictor;
    bool learnedEvictionEnabled;
    float learnedRecencyWeight;
    float learnedFrequencyWeight;
    float learnedPredictionWeight;
    u64 accessCounter;
    u32 residentPages;
    static constexpr u32 MAX_RESIDENT_PAGES = NUM_FRAMES;
    uint64_t learnedEvictions;

    uint64_t faultNs;

    uint64_t swapWriteNs;

    uint64_t swapReadNs;
};
