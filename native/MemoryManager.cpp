#include "MemoryManager.hpp"

#include <iostream>
#include <cassert>
#include <cstring>
#include <limits>

using namespace std;


// initialize the memory manager with the given policy
MemoryManager::MemoryManager(EvictionPolicy policy, bool learned)
    : frames(NUM_FRAMES * PAGE_SIZE, 0),
      swapMgr(1024, "swapfile.bin"),
      eviction(policy),
      learnedPredictor(),
      learnedEvictionEnabled(learned),
      learnedRecencyWeight(0.45f),
      learnedFrequencyWeight(0.25f),
      learnedPredictionWeight(0.30f),
      accessCounter(0),
      learnedEvictions(0),
      faultNs(0),
      swapWriteNs(0),
      swapReadNs(0){

    for(u32 i = 0; i < NUM_FRAMES; ++i){
        freeFrames.push(i); // Initially, all frames are free (duh)
    }
    cout << "MemoryManager initialized with " << NUM_FRAMES << " frames" << endl;
}

MemoryManager::~MemoryManager(){
    cout << "MemoryManager destroyed" << endl;
}

void MemoryManager::initPageTable(size_t num_pages){
    pageTbl_.resize(num_pages);
    cout << "Page table initialized for " << num_pages << " pages" << endl;
}

u32 MemoryManager::allocFrame(u64 vpn){
    if(!freeFrames.empty()){ // If we have free frames, just use one, so no faults :)
        u32 idx = freeFrames.front();
        freeFrames.pop();
        cout << "Allocated frame " << idx << " for VPN " << vpn << endl;
        return idx;
    }

    // welp now we have to evict
    u64 victim_vpn = learnedEvictionEnabled ? chooseLearnedVictim() : eviction.choose_victim(pageTbl_);
    auto& victim_entry = pageTbl_.getEntry(victim_vpn); // btw note that this is not a copy, it is a reference
    // so since its a reference, the modifications affect the real page table
    assert(victim_entry.resident); // this checks if the victim page is actually in the RAM
    u32 frameIndex = victim_entry.frameIndex; // this is the frame index of the victim page, which we'll reuse for the new page
    cout << "Evicting VPN " << victim_vpn << " from frame " << frameIndex << endl;
    if(victim_entry.dirty){
        if(victim_entry.swapSlot == SWAP_SLOT_INVALID){
            victim_entry.swapSlot = swapMgr.allocSlot();
        }

        auto start = std::chrono::high_resolution_clock::now();
        swapMgr.writePage(victim_entry.swapSlot, frameData(frameIndex)); // we write the victim page's data to swap before we reuse its frame for the new page
        auto end = std::chrono::high_resolution_clock::now();
        swapWriteNs += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        cout << "Wrote dirty VPN " << victim_vpn << " to swap slot " << victim_entry.swapSlot << endl;
    }

    victim_entry.resident = false;
    victim_entry.frameIndex = SWAP_SLOT_INVALID;
    victim_entry.dirty = false;
    victim_entry.reference = false;

    return frameIndex;
}

void MemoryManager::freeFrame(u32 frameIndex){
    freeFrames.push(frameIndex);
    cout << "Freed frame " << frameIndex << endl;
}

void* MemoryManager::frameData(u32 frameIndex){ // returns the pointer to the data of the given frame index
    assert(frameIndex < NUM_FRAMES);
    return frames.data() + frameIndex * PAGE_SIZE; // useful for lets say swapping, where we might need to write the data of a frame to the swap file
}

PageTable& MemoryManager::pageTbl(){
    return pageTbl_;
}


// now this basically tells the runtime that the page with the given VPN was accessed
// useful for GRU
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

void MemoryManager::loadPage(u64 vpn, u32 frameIndex){
    auto& entry = pageTbl_.getEntry(vpn);
    if(entry.swapSlot != SWAP_SLOT_INVALID){
        auto start = std::chrono::high_resolution_clock::now();
        swapMgr.readPage(entry.swapSlot, frameData(frameIndex));
        auto end = std::chrono::high_resolution_clock::now();
        swapReadNs += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        cout << "Read VPN " << vpn << " from swap slot " << entry.swapSlot << endl;
    }
    else{
        std::memset(frameData(frameIndex), 0, PAGE_SIZE);
    }
    entry.resident = true;
    entry.frameIndex = frameIndex;
    entry.dirty = false;
    entry.loadTimestamp = accessCounter;
    touchPage(vpn);
}

void MemoryManager::addFaultLatencyNs(uint64_t ns){
    faultNs += ns;
}

uint64_t MemoryManager::faultLatencyNs() const{ // this is the nanoseconds spent handling the page faults
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


// Learned evictor is a bit more complex, so it's in memory manager
// LRU and CLOCK are in eviction manager since they only need timestamps and reference bits
u64 MemoryManager::chooseLearnedVictim(){
    u64 victim_vpn = SWAP_SLOT_INVALID; // no victim selected yet
    // so we initialize with a invalid sentinel value
    float best_score = -std::numeric_limits<float>::infinity(); // basically the lowest score possible
    size_t num_pages = pageTbl_.size();
    for(u64 vpn = 0; vpn < num_pages; ++vpn){
        const auto& entry = pageTbl_.getEntry(vpn);
        if(!entry.resident){
            continue;
        }
        float score = computeLearnedScore(vpn);
        if(score > best_score){
            best_score = score;
            victim_vpn = vpn;
        }
    }

    if(victim_vpn == SWAP_SLOT_INVALID){
        return eviction.choose_victim(pageTbl_);
    }
    learnedEvictions += 1;
    return victim_vpn;
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
