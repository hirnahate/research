#include <iostream>
#include <vector>
#include <chrono>

#include "allocator.h"

// Test function
void testBasicAllocation() {
    std::cout << "=== Basic Allocation Test ===" << std::endl;
    
    // Create arena and allocator
    TestSlabArena arena;
    TestAllocator allocator(arena, 64); // 64-byte objects
    
    std::cout << "Arena created, slab count: " << TestSlabArena::SLAB_COUNT << std::endl;
    std::cout << "Slab size: " << TestSlabArena::slabType::SIZE << " bytes" << std::endl;
    
    // Test allocation
    void* ptr1 = allocator.alloc();
    if (ptr1) {
        std::cout << " First allocation successful: " << ptr1 << std::endl;
    } else {
        std::cout << " First allocation failed!" << std::endl;
        return;
    }
    
    void* ptr2 = allocator.alloc();
    if (ptr2) {
        std::cout << " Second allocation successful: " << ptr2 << std::endl;
    } else {
        std::cout << " Second allocation failed!" << std::endl;
    }
    
    // Test that pointers are different
    if (ptr1 != ptr2) {
        std::cout << " Allocated different addresses" << std::endl;
    } else {
        std::cout << " Same address allocated twice!" << std::endl;
    }
    
    // Test writing to allocated memory
    if (ptr1) {
        char* charPtr = static_cast<char*>(ptr1);
        for (int i = 0; i < 64; i++) {
            charPtr[i] = (char)(i % 256);
        }
        std::cout << " Successfully wrote to allocated memory" << std::endl;
        
        // Verify the data
        bool dataOk = true;
        for (int i = 0; i < 64; i++) {
            if (charPtr[i] != (char)(i % 256)) {
                dataOk = false;
                break;
            }
        }
        if (dataOk) {
            std::cout << " Data integrity verified" << std::endl;
        } else {
            std::cout << " Data corruption detected!" << std::endl;
        }
    }
    
    // Test freeing
    if (allocator.free(ptr1)) {
        std::cout << " First free successful" << std::endl;
    } else {
        std::cout << " First free failed!" << std::endl;
    }
    
    if (allocator.free(ptr2)) {
        std::cout << " Second free successful" << std::endl;
    } else {
        std::cout << " Second free failed!" << std::endl;
    }
    
    std::cout << std::endl;
}

void testMultipleAllocations() {
    std::cout << "=== Multiple Allocations Test ===" << std::endl;
    
    TestSlabArena arena;
    TestAllocator allocator(arena, 32); // 32-byte objects
    
    const int numAllocs = 100;
    std::vector<void*> ptrs;
    
    // Allocate multiple objects
    for (int i = 0; i < numAllocs; i++) {
        void* ptr = allocator.alloc();
        if (ptr) {
            ptrs.push_back(ptr);
        } else {
            std::cout << "Allocation failed at iteration " << i << std::endl;
            break;
        }
    }
    
    std::cout << "Successfully allocated " << ptrs.size() << " objects" << std::endl;
    
    // Verify all pointers are unique
    bool allUnique = true;
    for (size_t i = 0; i < ptrs.size() && allUnique; i++) {
        for (size_t j = i + 1; j < ptrs.size(); j++) {
            if (ptrs[i] == ptrs[j]) {
                allUnique = false;
                std::cout << " Duplicate pointer found at indices " << i << " and " << j << std::endl;
                break;
            }
        }
    }
    
    if (allUnique) {
        std::cout << " All allocated pointers are unique" << std::endl;
    }
    
    // Free all objects
    int freedCount = 0;
    for (void* ptr : ptrs) {
        if (allocator.free(ptr)) {
            freedCount++;
        }
    }
    
    std::cout << "Successfully freed " << freedCount << "/" << ptrs.size() << " objects" << std::endl;
    std::cout << std::endl;
}

void testDifferentSizes() {
    std::cout << "=== Different Size Test ===" << std::endl;
    
    TestSlabArena arena;
    
    std::vector<size_t> sizes = {8, 16, 32, 64, 128, 256, 512, 1024};
    
    for (size_t size : sizes) {
        TestAllocator allocator(arena, size);
        void* ptr = allocator.alloc();
        
        if (ptr) {
            std::cout << " Size " << size << " bytes: allocation successful" << std::endl;
            
            // Test writing
            char* charPtr = static_cast<char*>(ptr);
            for (size_t i = 0; i < size; i++) {
                charPtr[i] = (char)(i % 256);
            }
            
            allocator.free(ptr);
        } else {
            std::cout << " Size " << size << " bytes: allocation failed" << std::endl;
        }
    }
    
    std::cout << std::endl;
}

void performanceTest() {
    std::cout << "=== Performance Test ===" << std::endl;
    
    TestSlabArena arena;
    TestAllocator allocator(arena, 64);
    
    const int iterations = 10000;
    std::vector<void*> ptrs;
    ptrs.reserve(iterations);
    
    // Time allocations
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        void* ptr = allocator.alloc();
        if (ptr) {
            ptrs.push_back(ptr);
        } else {
            break;
        }
    }
    
    auto mid = std::chrono::high_resolution_clock::now();
    
    // Time deallocations
    for (void* ptr : ptrs) {
        allocator.free(ptr);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    
    auto allocTime = std::chrono::duration_cast<std::chrono::microseconds>(mid - start);
    auto freeTime = std::chrono::duration_cast<std::chrono::microseconds>(end - mid);
    
    std::cout << "Allocated " << ptrs.size() << " objects in " << allocTime.count() << " μs" << std::endl;
    std::cout << "Freed " << ptrs.size() << " objects in " << freeTime.count() << " μs" << std::endl;
    std::cout << "Average allocation time: " << (double)allocTime.count() / ptrs.size() << " μs" << std::endl;
    std::cout << "Average free time: " << (double)freeTime.count() / ptrs.size() << " μs" << std::endl;
    
    std::cout << std::endl;
}

int main() {
    std::cout << "Slab Allocator Test" << std::endl;
    std::cout << "===================" << std::endl << std::endl;
    
    try {
        testBasicAllocation();
        testMultipleAllocations();
        testDifferentSizes();
        performanceTest();
        
        std::cout << "All tests completed!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown exception caught!" << std::endl;
        return 1;
    }
    
    return 0;
}
