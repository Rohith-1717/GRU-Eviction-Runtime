#include "MemoryManager.hpp"

#include <iostream>
#include <cassert>
#include <cstring>
#include <limits>

using namespace std;

MemoryManager::MemoryManager(EvictionPolicy policy, bool learned)
    : swapMgr(1024, "swapfile.bin"),
      eviction(policy),
      learnedPredictor(),
      learnedEvictionEnabled(learned),
      learnedRecencyWeight(0.45f),
      learnedFrequencyWeight(0.25f),
      learnedPredictionWeight(0.30f),
      accessCounter(0),
      residentPages(0),
      learnedEvictions(0),
      faultNs(0),
      swapWriteNs(0),
      swapReadNs(0){

    cout << "MemoryManager initialized" << endl;
}

MemoryManager::~MemoryManager(){
    cout << "MemoryManager destroyed" << endl;
}

void MemoryManager::initPageTable(size_t num_pages){
    pageTbl_.resize(num_pages);
    bufferPool.resize(num_pages * PAGE_SIZE, 0);
    for(u32 vpn = 0; vpn < num_pages; ++vpn){
        auto& entry = pageTbl_.getEntry(vpn);
        entry.bufferIndex = vpn;
    }

    cout << "Page table initialized for " << num_pages << " pages" << endl;
}

void* MemoryManager::bufferData(u32 bufferIndex){
    assert(bufferIndex < pageTbl_.size());
    return bufferPool.data() + bufferIndex * PAGE_SIZE;
}

PageTable& MemoryManager::pageTbl(){
    return pageTbl_;
}

bool MemoryManager::hasFreeResidentSlot() const{
    return residentPages < MAX_RESIDENT_PAGES;
}

u32 MemoryManager::residentPageCount() const{
    return residentPages;
}

void MemoryManager::touchPage(u64 vpn){
    assert(vpn < pageTbl_.size());

    auto& entry = pageTbl_.getEntry(vpn);

    entry.previousAccess = entry.lastAccess;
    entry.lastAccess = ++accessCounter;
    entry.frequency += 1;

    float accessFeatures[RuntimeGRU::ACCESS_FEATURE_SIZE] = {
        float(vpn),
        float(entry.lastAccess - entry.previousAccess),
        entry.dirty ? 1.0f : 0.0f,
        float(entry.lastAccess),
        0.0f
    };

    learnedPredictor.step(accessFeatures);
    eviction.touch(vpn, entry, accessCounter);
}

void MemoryManager::loadPage(u64 vpn){
    auto& entry = pageTbl_.getEntry(vpn);

    if(entry.swapSlot != SWAP_SLOT_INVALID){
        auto start = std::chrono::high_resolution_clock::now();

        swapMgr.readPage(
            entry.swapSlot,
            bufferData(entry.bufferIndex)
        );

        auto end = std::chrono::high_resolution_clock::now();

        swapReadNs += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    else{
        std::memset(
            bufferData(entry.bufferIndex),
            0,
            PAGE_SIZE
        );
    }

    entry.resident = true;
    entry.dirty = false;
    entry.loadTimestamp = accessCounter;

    residentPages++;

    touchPage(vpn);
}

void MemoryManager::evictPage(u64 vpn){
    auto& entry = pageTbl_.getEntry(vpn);

    if(entry.dirty){

        if(entry.swapSlot == SWAP_SLOT_INVALID){
            entry.swapSlot = swapMgr.allocSlot();
        }

        auto start = std::chrono::high_resolution_clock::now();

        swapMgr.writePage(
            entry.swapSlot,
            bufferData(entry.bufferIndex)
        );

        auto end = std::chrono::high_resolution_clock::now();

        swapWriteNs += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }

    entry.resident = false;
    entry.reference = false;
    entry.dirty = false;

    residentPages--;
}

void MemoryManager::addFaultLatencyNs(uint64_t ns){
    faultNs += ns;
}

uint64_t MemoryManager::faultLatencyNs() const{
    return faultNs;
}

float MemoryManager::computeLearnedScore(u64 vpn){
    auto& entry = pageTbl_.getEntry(vpn);

    float pageFeatures[RuntimeGRU::PAGE_FEATURE_SIZE] = {
        float(accessCounter - entry.lastAccess),
        float(entry.frequency),
        entry.dirty ? 1.0f : 0.0f,
        float(accessCounter - entry.loadTimestamp)
    };

    float mlScore = learnedPredictor.predictReuseProbability(pageFeatures);

    entry.predictedReuse = mlScore;

    float recency = float(accessCounter - entry.lastAccess);
    float frequencyPenalty = 1.0f / (1.0f + float(entry.frequency));

    float score = learnedRecencyWeight * recency;
    score += learnedFrequencyWeight * frequencyPenalty;
    score += learnedPredictionWeight * (1.0f - mlScore);

    return score;
}

u64 MemoryManager::chooseLearnedVictim(){
    u64 victim = SWAP_SLOT_INVALID;
    float bestScore = -std::numeric_limits<float>::infinity();

    for(u64 vpn = 0; vpn < pageTbl_.size(); ++vpn){

        const auto& entry = pageTbl_.getEntry(vpn);

        if(!entry.resident){
            continue;
        }

        float score = computeLearnedScore(vpn);

        if(score > bestScore){
            bestScore = score;
            victim = vpn;
        }
    }

    if(victim == SWAP_SLOT_INVALID){
        return eviction.choose_victim(pageTbl_);
    }

    learnedEvictions++;

    return victim;
}

uint64_t MemoryManager::swapWriteLatencyNs() const{
    return swapWriteNs;
}

uint64_t MemoryManager::swapReadLatencyNs() const{
    return swapReadNs;
}

uint64_t MemoryManager::learnedEvictionCount() const{
    return learnedEvictions;
}

bool MemoryManager::learnedEvictionActive() const{
    return learnedEvictionEnabled;
}
