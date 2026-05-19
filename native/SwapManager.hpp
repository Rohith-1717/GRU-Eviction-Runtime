#pragma once

#include "Common.hpp"
#include <cstdint>
#include <string>
#include <vector>

class SwapManager{
public:
    explicit SwapManager(size_t max_pages = 1024, const std::string& path = "swapfile.bin");
    ~SwapManager();

    u32 allocSlot();
    void releaseSlot(u32 slot);
    void writePage(u32 slot, const void* data);
    void readPage(u32 slot, void* dst);

private:
    int fd;
    u32 max_pages;
    std::vector<bool> slot_used;
    std::string path;
};