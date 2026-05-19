import sys
import os
import random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'native', 'build'))

import swapcore_native as scn


def test_basic_fault():
    manager = scn.MemoryManager()
    handler = scn.FaultHandler(manager)
    region = scn.MemRegion(8192, handler)  # Two pages
    handler.start()
    region.set(0, 42)
    assert region.get(0) == 42
    region.set(4096, 24)
    assert region.get(4096) == 24
    handler.stop()
    print("basic fault test passed")


def test_sequential_access():
    manager = scn.MemoryManager()
    handler = scn.FaultHandler(manager)
    region = scn.MemRegion(65536, handler)  # 16 pages
    handler.start()
    for i in range(0, region.getSize(), 4096):
        region.set(i, i // 4096)
    for i in range(0, region.getSize(), 4096):
        assert region.get(i) == i // 4096
    handler.stop()
    print("sequential access test passed")


def test_random_access():
    manager = scn.MemoryManager()
    handler = scn.FaultHandler(manager)
    region = scn.MemRegion(65536, handler)
    handler.start()
    rng = random.Random(12345)
    entries = []
    for _ in range(32):
        offset = rng.randrange(0, region.getSize(), 4096)
        value = rng.randrange(0, 256)
        region.set(offset, value)
        entries.append((offset, value))
    for offset, value in entries:
        assert region.get(offset) == value
    handler.stop()
    print("random access test passed")


def test_looping_access():
    manager = scn.MemoryManager()
    handler = scn.FaultHandler(manager)
    region = scn.MemRegion(32768, handler)
    handler.start()
    for _ in range(10):
        for i in range(0, region.getSize(), 4096):
            region.set(i, (i // 4096) + 1)
            assert region.get(i) == (i // 4096) + 1
    handler.stop()
    print("looping access test passed")


def test_eviction_swapback():
    manager = scn.MemoryManager()
    handler = scn.FaultHandler(manager)
    region = scn.MemRegion(4096 * 20, handler)  # 20 pages, more than 16 frames
    handler.start()

    for page in range(20):
        addr = page * 4096
        region.set(addr, page + 1)

    # Access the oldest pages again to trigger swap reloads
    for page in range(4):
        addr = page * 4096
        assert region.get(addr) == page + 1

    handler.stop()
    assert manager.swapWriteLatencyNs() > 0
    assert manager.swapReadLatencyNs() > 0
    print("eviction and swapback test passed")


if __name__ == "__main__":
    test_basic_fault()
    test_sequential_access()
    test_random_access()
    test_looping_access()
    test_eviction_swapback()
