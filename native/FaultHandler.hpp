#pragma once

#include "Common.hpp"
#include "MemoryManager.hpp"

#include <thread>
#include <vector>

class FaultHandler{
public:
    explicit FaultHandler(MemoryManager* mgr);
    ~FaultHandler();
    void registerRegion(void* addr, size_t size);
    void start();
    void stop();
    MemoryManager* getManager();

private:
    void handleFaults();
    int uffd;
    bool running;
    void* zeroPage;
    MemoryManager* manager;
    void* regionBase;
    size_t regionSize;
    std::vector<struct uffdio_register> regions;
    std::thread handlerThread;
};
