#include "MemRegion.hpp"
#include <sys/mman.h>
#include <cassert>
#include <iostream>
using namespace std;

// This is the virtual memory region of SwapCore


MemRegion::MemRegion(size_t size, FaultHandler* handler) : size(size), handler(handler), manager(handler->getManager()){
    // this uses mmap to allocate a memory of the given size 
    // basically this asks Linux to give me a chunk of virtual memory xD
    addr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (addr == MAP_FAILED){
        cerr << "Failed to mmap region" << endl;
        exit(1);
    }
    // regsiter the region to the fault handler
    handler->registerRegion(addr, size);
}


MemRegion::~MemRegion(){
    if (addr){
        munmap(addr, size);
    }
}


// This is for reading a byte
u8 MemRegion::get(size_t offset){
    assert(offset < size);
    u64 vpn = offset/PAGE_SIZE;
    if (manager){
        manager->touchPage(vpn);  // This updates the metdata for the page, useful for GRU to learn the access pattern
    }
    return *(u8*)((char*)addr+offset);
}

// This is for writing a byte
void MemRegion::set(size_t offset, u8 value){
    assert(offset < size);
    *(u8*)((char*)addr + offset) = value;
    u64 vpn = offset/PAGE_SIZE;
    if (manager){
        auto& entry = manager->pageTbl().getEntry(vpn); // 
        entry.dirty = true; // marks it as dirty, so when it is evicted, the data in it has to be written back to swap
    }
}
