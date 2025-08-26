#include <iostream>
#include <cstdio>
#include "allocator.h"

void debugSlabArena() {
    std::cout << "\n=== Debug Slab Arena ===" << std::endl;
    
    TestSlabArena arena;
    
    std::cout << "Arena created successfully" << std::endl;
    std::cout << "SLAB_COUNT: " << TestSlabArena::SLAB_COUNT << std::endl;
    std::cout << "SLAB_SIZE: " << TestSlabArena::slabType::SIZE << std::endl;
    
    // Test basic slab allocation
    size_t objectSize = 64;
    std::cout << "\nTesting SlabArena::alloc(" << objectSize << ")..." << std::endl;
    
    auto slabAddr = arena.alloc(objectSize);
    std::cout << "SlabArena::alloc returned: " << slabAddr << std::endl;
    std::cout << "NULL_ADDR is: " << TestSlabArena::NULL_ADDR << std::endl;
    
    if (slabAddr == TestSlabArena::NULL_ADDR) {
        std::cout << ":( SlabArena::alloc failed!" << std::endl;
        return;
    }
    
    std::cout << ":) Got slab " << slabAddr << std::endl;
    
    // Now test proxy operations
    auto& proxy = arena.proxyAt(slabAddr).data;
    auto& slab = arena.slabAt(slabAddr);
    
    std::cout << "\nTesting proxy.claim..." << std::endl;
    std::cout << "Proxy reservation state: " << intr::atomic::load_relaxed(&proxy.reservationState) << std::endl;
    std::cout << "Proxy alloc state: " << proxy.allocState << std::endl;
    
    bool claimResult = proxy.claim(&slab, objectSize);
    std::cout << "proxy.claim(" << objectSize << ") returned: " << claimResult << std::endl;
    
    if (!claimResult) {
        std::cout << ":( proxy.claim failed!" << std::endl;
        std::cout << "Final reservation state: " << intr::atomic::load_relaxed(&proxy.reservationState) << std::endl;
        std::cout << "Final alloc state: " << proxy.allocState << std::endl;
        return;
    }
    
    std::cout << "\nTesting proxy.confirmReservation..." << std::endl;
    std::cout << "Before confirmReservation - allocState: " << proxy.allocState << std::endl;
    std::cout << "Before confirmReservation - reservationState: " << intr::atomic::load_relaxed(&proxy.reservationState) << std::endl;

    bool confirmResult = proxy.confirmReservation(objectSize);
    std::cout << "proxy.confirmReservation(" << objectSize << ") returned: " << confirmResult << std::endl;
    
    std::cout << "After confirmReservation - allocState: " << proxy.allocState << std::endl;
    std::cout << "After confirmReservation - reservationState: " << intr::atomic::load_relaxed(&proxy.reservationState) << std::endl;
    
    if (!confirmResult) {
        std::cout << ":( proxy.confirmReservation failed!" << std::endl;
        return;
    }
    
    std::cout << "\nTesting proxy.alloc..." << std::endl;
    bool slabFilled = false;
    void* ptr = proxy.alloc(&slab, slabFilled);
    std::cout << "proxy.alloc returned: " << ptr << std::endl;
    std::cout << "slabFilled: " << slabFilled << std::endl;
    
    if (!ptr) {
        std::cout << ":( proxy.alloc failed!" << std::endl;
        return;
    }
    
    std::cout << ":) Got pointer " << ptr << std::endl;
}

void debugSimpleAllocator() {
    std::cout << "\n=== Debug Simple Allocator ===" << std::endl;
    
    TestSlabArena arena;
    TestAllocator allocator(arena, 64);  // 64-byte objects
    
    std::cout << "Testing TestAllocator::alloc()..." << std::endl;
    void* ptr = allocator.alloc();
    std::cout << "TestAllocator::alloc() returned: " << ptr << std::endl;
    
    if (ptr) {
        std::cout << ":) Got valid pointer" << std::endl;
        
        // magic to check its real
        uint32_t* intPtr = static_cast<uint32_t*>(ptr);
        *intPtr = 0xDEADBEEF;
        
        if (*intPtr == 0xDEADBEEF) {
            std::cout << ":) Memory is writable and readable" << std::endl;
        } else {
            std::cout << ":( Memory corruption detected" << std::endl;
        }
        
        std::cout << "Testing free..." << std::endl;
        bool freed = allocator.free(ptr);
        std::cout << "Free result: " << (freed ? "SUCCESS" : "FAILED") << std::endl;
    } else {
        std::cout << ":( Allocation returned null" << std::endl;
    }
}

void debugSlabObjCount() {
    std::cout << "\n=== Debug Slab Object Count ===" << std::endl;
    
    TestSlabArena::slabProxyType proxy;
    
    size_t sizes[] = {8, 16, 32, 64, 128, 256, 512, 1024};
    
    for (size_t size : sizes) {
        size_t count = proxy.slabObjCount(size);
        std::cout << "Object size " << size << " -> max objects: " << count << std::endl;
        
        if (count == 0) {
            std::cout << ":( Zero objects for size " << size << "!" << std::endl;
        }
    }
}

int main() {
    std::cout << "FreeList Allocator Debug" << std::endl;
    std::cout << "========================" << std::endl;
    
    try {
        debugSlabObjCount();
        debugSlabArena();
        debugSimpleAllocator();
        
        std::cout << "\nDebug complete!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown exception!" << std::endl;
        return 1;
    }
    
    return 0;
}