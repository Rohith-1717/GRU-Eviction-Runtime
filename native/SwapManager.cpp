#include "SwapManager.hpp"
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <iostream>
#include <cstring>
#include <cassert>

SwapManager::SwapManager(size_t max_pages_, const std::string& path_)
    : fd(-1), max_pages(max_pages_), slot_used(max_pages_, false), path(path_){
    fd = open(path.c_str(), O_CREAT | O_RDWR, 0600);
    if (fd == -1){
        std::cerr << "Failed to open swap file: " << path << std::endl;
        exit(1);
    }
    off_t target_size = static_cast<off_t>(max_pages) * PAGE_SIZE;
    if (ftruncate(fd, target_size) == -1){
        std::cerr << "Failed to size swap file" << std::endl;
        exit(1);
    }
    std::cout << "SwapManager initialized with " << max_pages << " slots" << std::endl;
}

SwapManager::~SwapManager(){
    if (fd != -1){
        close(fd);
    }
}

u32 SwapManager::allocSlot(){
    for (u32 i = 0; i < max_pages; ++i){
        if (!slot_used[i]){
            slot_used[i] = true;
            return i;
        }
    }
    std::cerr << "Swap space exhausted" << std::endl;
    assert(false);
    return SWAP_SLOT_INVALID;
}

void SwapManager::releaseSlot(u32 slot){
    if (slot >= max_pages){
        return;
    }
    slot_used[slot] = false;
}

void SwapManager::writePage(u32 slot, const void* data){
    assert(slot < max_pages);
    off_t offset = static_cast<off_t>(slot) * PAGE_SIZE;
    ssize_t written = pwrite(fd, data, PAGE_SIZE, offset);
    if (written != static_cast<ssize_t>(PAGE_SIZE)){
        std::cerr << "Swap write failed at slot " << slot << std::endl;
        exit(1);
    }
}

void SwapManager::readPage(u32 slot, void* dst){
    assert(slot < max_pages);
    off_t offset = static_cast<off_t>(slot) * PAGE_SIZE;
    ssize_t read_bytes = pread(fd, dst, PAGE_SIZE, offset);
    if (read_bytes != static_cast<ssize_t>(PAGE_SIZE)){
        std::cerr << "Swap read failed at slot " << slot << std::endl;
        exit(1);
    }
}