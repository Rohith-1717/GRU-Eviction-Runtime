#include "MemoryManager.hpp"

#include <iostream>
#include <cassert>
#include <cstring>
#include <limits>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/userfaultfd.h>

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

void MemoryManager::evictPage(u64 vpn, void* pageAddr){
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

    madvise(pageAddr, PAGE_SIZE, MADV_DONTNEED);

    entry.resident = false;
    entry.reference = false;
    entry.dirty = false;

    residentPages--;
}
