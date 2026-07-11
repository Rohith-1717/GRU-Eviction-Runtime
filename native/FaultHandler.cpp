#include "FaultHandler.hpp"

#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <poll.h>
#include <linux/userfaultfd.h>
#include <cstring>
#include <cassert>
#include <iostream>
#include <chrono>

using namespace std;

FaultHandler::FaultHandler(MemoryManager* mgr)
    : uffd(-1),
      running(false),
      zeroPage(nullptr),
      manager(mgr),
      regionBase(nullptr),
      regionSize(0){

    uffd = static_cast<int>(syscall(__NR_userfaultfd, 0));

    if(uffd == -1){
        cerr << "Failed to create userfaultfd" << endl;
        exit(1);
    }

    struct uffdio_api api;

    api.api = UFFD_API;
    api.features = 0;

    if(ioctl(uffd, UFFDIO_API, &api) == -1){
        cerr << "UFFDIO_API failed" << endl;
        exit(1);
    }

    zeroPage = mmap(
        nullptr,
        PAGE_SIZE,
        PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS,
        -1,
        0
    );

    if(zeroPage == MAP_FAILED){
        cerr << "Failed to allocate zero page" << endl;
        exit(1);
    }

    memset(zeroPage, 0, PAGE_SIZE);
}

FaultHandler::~FaultHandler(){
    if(running){
        stop();
    }

    if(zeroPage){
        munmap(zeroPage, PAGE_SIZE);
    }

    if(uffd != -1){
        close(uffd);
    }
}

void FaultHandler::registerRegion(void* addr, size_t size){
    regionBase = addr;
    regionSize = size;

    manager->initPageTable(size / PAGE_SIZE);

    struct uffdio_register reg;

    reg.range.start = (unsigned long)addr;
    reg.range.len = size;
    reg.mode = UFFDIO_REGISTER_MODE_MISSING;

    if(ioctl(uffd, UFFDIO_REGISTER, &reg) == -1){
        cerr << "UFFDIO_REGISTER failed" << endl;
        exit(1);
    }

    regions.push_back(reg);
}

void FaultHandler::start(){
    running = true;
    handlerThread = thread(&FaultHandler::handleFaults, this);
}

void FaultHandler::stop(){
    running = false;

    if(handlerThread.joinable()){
        handlerThread.join();
    }
}

MemoryManager* FaultHandler::getManager(){
    return manager;
}

void FaultHandler::handleFaults(){
    while(running){

        struct pollfd pfd = {uffd, POLLIN, 0};

        int ret = poll(&pfd, 1, 1000);

        if(ret == -1){
            cerr << "Poll failed" << endl;
            break;
        }

        if(ret == 0){
            continue;
        }

        if(!(pfd.revents & POLLIN)){
            continue;
        }

        struct uffd_msg msg;

        ssize_t nread = read(uffd, &msg, sizeof(msg));

        if(nread == 0){
            cerr << "EOF on userfaultfd" << endl;
            break;
        }

        if(nread == -1){
            cerr << "Read failed" << endl;
            break;
        }

        if(msg.event != UFFD_EVENT_PAGEFAULT){
            cerr << "Unexpected event" << endl;
            continue;
        }

        void* faultAddr = (void*)msg.arg.pagefault.address;

        void* pageAddr =
            (void*)((unsigned long)faultAddr & ~(PAGE_SIZE - 1));

        u64 vpn =
            ((u64)pageAddr - (u64)regionBase) / PAGE_SIZE;

        assert(vpn < regionSize / PAGE_SIZE);

        auto faultStart =
            std::chrono::high_resolution_clock::now();

        auto& entry = manager->pageTbl().getEntry(vpn);

        if(!entry.resident){

            if(!manager->hasFreeResidentSlot()){

                u64 victim = manager->chooseLearnedVictim();

                void* victimAddr =
                    (char*)regionBase + victim * PAGE_SIZE;

                manager->evictPage(victim);

                madvise(
                    victimAddr,
                    PAGE_SIZE,
                    MADV_DONTNEED
                );
            }

            manager->loadPage(vpn);
            struct uffdio_copy copy;

            copy.src = (unsigned long)manager->bufferData(entry.bufferIndex);
            copy.dst = (unsigned long)pageAddr;
            copy.len = PAGE_SIZE;
            copy.mode = 0;
            copy.copy = 0;

            if(ioctl(uffd, UFFDIO_COPY, &copy) == -1){
                cerr << "UFFDIO_COPY failed" << endl;
                break;
            }

            cout << "Loaded VPN " << vpn << endl;
        }
        else{

            manager->touchPage(vpn);

            struct uffdio_copy copy;

            copy.src = (unsigned long)manager->bufferData(entry.bufferIndex);
            copy.dst = (unsigned long)pageAddr;
            copy.len = PAGE_SIZE;
            copy.mode = 0;
            copy.copy = 0;

            if(ioctl(uffd, UFFDIO_COPY, &copy) == -1){
                cerr << "UFFDIO_COPY failed" << endl;
                break;
            }

            cout << "Re-fault for resident VPN " << vpn << endl;
        }

        auto faultEnd = std::chrono::high_resolution_clock::now();

        manager->addFaultLatencyNs(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                faultEnd - faultStart
            ).count()
        );
    }
}
