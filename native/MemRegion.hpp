#pragma once

#include "Common.hpp"
#include "FaultHandler.hpp"

class MemRegion{
public:
    MemRegion(size_t size, FaultHandler* handler);
    ~MemRegion();
    u8 get(size_t offset);
    void set(size_t offset, u8 value);
    size_t getSize() const { return size; }

private:
    void* addr;
    size_t size;
    FaultHandler* handler;
    MemoryManager* manager;
};