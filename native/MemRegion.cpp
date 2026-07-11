#include "MemRegion.hpp"

#include <sys/mman.h>
#include <cassert>
#include <cstring>
#include <iostream>

using namespace std;

MemRegion::MemRegion(size_t size, FaultHandler* handler)
    : size(size),
      handler(handler),
      manager(handler->getManager()){

    addr = mmap(
        nullptr,
        size,
        PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS,
        -1,
        0
    );

    if(addr == MAP_FAILED){
        cerr << "Failed to mmap region" << endl;
        exit(1);
    }

    handler->registerRegion(addr, size);
}

MemRegion::~MemRegion(){
    if(addr){
        munmap(addr, size);
    }
}

u8 MemRegion::get(size_t offset){
    assert(offset < size);
    u64 vpn = offset / PAGE_SIZE;
    u8 value = *(u8*)((char*)addr + offset);
    if(manager){
        manager->touchPage(vpn);
    }
    return value;
}

void MemRegion::set(size_t offset, u8 value){
    assert(offset < size);
    u64 vpn = offset / PAGE_SIZE;
    u64 pageOffset = offset % PAGE_SIZE;

    *(u8*)((char*)addr + offset) = value;

    if(manager){
        auto& entry = manager->pageTbl().getEntry(vpn);
        ((u8*)manager->bufferData(entry.bufferIndex))[pageOffset] = value;
        entry.dirty = true;

        manager->touchPage(vpn);
    }
}
