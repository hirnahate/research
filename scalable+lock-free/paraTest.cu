#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <random>
#include <chrono>
#include <cassert>
#include <unordered_set>
#include <mutex>

#include "allocator.h"

// lets make better error codes

enum class AllocationError {
    SUCCESS = 0,
    NULL_POINTER,
    INVALID_SLAB_INDEX,
    SLAB_NOT_BOUND,
    OFFSET_TOO_SMALL,
    MISALIGNED_OBJECT,
    OBJECT_INDEX_OUT_OF_RANGE,
    TRACKING_INDEX_OUT_OF_RANGE,
    CAS_FAILURE_ALREADY_ALLOCATED,
    CAS_FAILURE_DOUBLE_FREE,
    ALLOCATOR_FAILED,
    TRACKER_RECORDING_FAILED
};

const char* errorToString(AllocationError error) {
    switch (error) {
        case AllocationError::SUCCESS: return "SUCCESS";
        case AllocationError::NULL_POINTER: return "NULL_POINTER";
        case AllocationError::INVALID_SLAB_INDEX: return "INVALID_SLAB_INDEX";
        case AllocationError::SLAB_NOT_BOUND: return "SLAB_NOT_BOUND";
        case AllocationError::OFFSET_TOO_SMALL: return "OFFSET_TOO_SMALL";
        case AllocationError::MISALIGNED_OBJECT: return "MISALIGNED_OBJECT";
        case AllocationError::OBJECT_INDEX_OUT_OF_RANGE: return "OBJECT_INDEX_OUT_OF_RANGE";
        case AllocationError::TRACKING_INDEX_OUT_OF_RANGE: return "TRACKING_INDEX_OUT_OF_RANGE";
        case AllocationError::CAS_FAILURE_ALREADY_ALLOCATED: return "CAS_FAILURE_ALREADY_ALLOCATED";
        case AllocationError::CAS_FAILURE_DOUBLE_FREE: return "CAS_FAILURE_DOUBLE_FREE";
        case AllocationError::ALLOCATOR_FAILED: return "ALLOCATOR_FAILED";
        case AllocationError::TRACKER_RECORDING_FAILED: return "TRACKER_RECORDING_FAILED";
        default: return "UNKNOWN_ERROR";
    }
}

// global tracker for allocations/frees using an atomic bitmask
class ParallelTracker {
public:
    static constexpr size_t PER_SLAB_STRIDE = 1024;
    static constexpr size_t MAX_TRACKED_OBJECTS = 256 * PER_SLAB_STRIDE;
private:
    // each entry: [31:16] = size, [15:0] = thread_id, 0 = free
    // std::atomic<uint32_t> trackingArena[MAX_TRACKED_OBJECTS];
    std::unique_ptr<std::atomic<uint32_t>[]> trackingArena;
    std::atomic<size_t> totalAllocations{0};
    std::atomic<size_t> totalFrees{0};
    std::atomic<size_t> totalFailures{0};

    mutable std::mutex debugMutex;
    
public:

    ParallelTracker():trackingArena(new std::atomic<uint32_t>[MAX_TRACKED_OBJECTS]) {
        for (size_t i = 0; i < MAX_TRACKED_OBJECTS; ++i) {
            trackingArena[i].store(0, std::memory_order_relaxed);
        }
    }

    // figure out arena index for a pointer
    std::pair<size_t, AllocationError> getIndexForPointer(void* ptr, TestSlabArena& arena) {
        if (!ptr) {
            return {MAX_TRACKED_OBJECTS, AllocationError::NULL_POINTER};
        }

        // which slab?
        auto slabIndex = arena.slabIndexFor(ptr);
        if (slabIndex >= TestSlabArena::SLAB_COUNT) {
            std::lock_guard<std::mutex> lock(debugMutex);
            std::cout << "    DEBUG: ptr=" << ptr << " -> slabIndex=" << slabIndex 
                        << " (>= SLAB_COUNT=" << TestSlabArena::SLAB_COUNT << ")" << std::endl;
            
            // LSF analysis - show pointer details
            std::cout << "    LSF: ptr=0x" << std::hex << reinterpret_cast<uintptr_t>(ptr) 
                        << std::dec << " (LSF=" << (reinterpret_cast<uintptr_t>(ptr) & 0xF) << ")" << std::endl;
            std::cout << "    Checking all slabs for potential matches:" << std::endl;
            for (size_t i = 0; i < TestSlabArena::SLAB_COUNT; i++) {
                auto& slab = arena.slabAt(i);
                char* slabStart = reinterpret_cast<char*>(&slab);
                char* slabEnd = slabStart + TestSlabArena::slabType::SIZE;
                char* ptrChar = static_cast<char*>(ptr);
                
                if (ptrChar >= slabStart && ptrChar < slabEnd) {
                    std::cout << "      Found ptr in slab " << i << " (range: " 
                                << static_cast<void*>(slabStart) << " - " 
                                << static_cast<void*>(slabEnd) << ")" << std::endl;
                }
            }
            
            return {MAX_TRACKED_OBJECTS, AllocationError::INVALID_SLAB_INDEX};
        }

        // read proxy + slab
        auto& proxy = arena.proxyAt(slabIndex).data;
        if (proxy.getSize() == 0) {
            std::lock_guard<std::mutex> lock(debugMutex);
            std::cout << "    DEBUG: slabIndex=" << slabIndex << " not bound (size=0)" << std::endl;
            return {MAX_TRACKED_OBJECTS, AllocationError::SLAB_NOT_BOUND};
        }

        auto& slab = arena.slabAt(slabIndex);

        // re-compute object index like proxy.free()
        char* p   = static_cast<char*>(ptr);
        char* sb  = reinterpret_cast<char*>(&slab);

        size_t objectSize   = proxy.getSize();
        size_t maxObjCount  = proxy.slabObjCount(objectSize);
        if (maxObjCount == 0) {
            std::lock_guard<std::mutex> lock(debugMutex);
            std::cout << "    DEBUG: slabIndex=" << slabIndex << " objectSize=" << objectSize 
                        << " -> maxObjCount=0" << std::endl;
            return {MAX_TRACKED_OBJECTS, AllocationError::OBJECT_INDEX_OUT_OF_RANGE};
        }

        // number of mask elements and their byte size
        using allocMaskElem = typename TestSlabArena::slabProxyType::allocMaskElem;
        constexpr size_t ELEM_BITS = sizeof(allocMaskElem) * 8;

        size_t maskCount = (maxObjCount + ELEM_BITS - 1) / ELEM_BITS;
        size_t maskBytes = maskCount * sizeof(allocMaskElem);

        size_t firstObjOffset = maskBytes;

        size_t byteOffset = static_cast<size_t>(p - sb);
        if (byteOffset < firstObjOffset) {
            std::lock_guard<std::mutex> lock(debugMutex);
            std::cout << "    DEBUG: byteOffset=" << byteOffset << " < firstObjOffset=" 
                        << firstObjOffset << std::endl;
            return {MAX_TRACKED_OBJECTS, AllocationError::OFFSET_TOO_SMALL};
        }

        size_t objOffset = byteOffset - firstObjOffset;
        if (objOffset % objectSize != 0) {
            std::lock_guard<std::mutex> lock(debugMutex);
            std::cout << "    DEBUG: objOffset=" << objOffset << " % objectSize=" 
                        << objectSize << " = " << (objOffset % objectSize) << " (not aligned)" << std::endl;
            return {MAX_TRACKED_OBJECTS, AllocationError::MISALIGNED_OBJECT};
        }

        size_t objIndex = objOffset / objectSize;
        if (objIndex >= maxObjCount) {
            std::lock_guard<std::mutex> lock(debugMutex);
            std::cout << "    DEBUG: objIndex=" << objIndex << " >= maxObjCount=" 
                        << maxObjCount << std::endl;
            return {MAX_TRACKED_OBJECTS, AllocationError::OBJECT_INDEX_OUT_OF_RANGE};
        }

        // flat index with fixed stride
        size_t idx = static_cast<size_t>(slabIndex) * PER_SLAB_STRIDE + objIndex;
        if (idx >= MAX_TRACKED_OBJECTS) {
            std::lock_guard<std::mutex> lock(debugMutex);
            std::cout << "    DEBUG: computed index=" << idx << " >= MAX_TRACKED_OBJECTS=" 
                        << MAX_TRACKED_OBJECTS << std::endl;
            return {MAX_TRACKED_OBJECTS, AllocationError::TRACKING_INDEX_OUT_OF_RANGE};
        }

        return {idx, AllocationError::SUCCESS};
    } // end of getInd
    
    // wrapper
    size_t getIndexForPtr(void* ptr, TestSlabArena& arena) {
        auto result = getIndexForPointer(ptr, arena);
        return result.first;
    }
    
    // log an allocation with CAS
    bool recordAllocation(void* ptr, size_t size, uint16_t threadId, TestSlabArena& arena) {
        auto result = getIndexForPointer(ptr, arena);
        size_t index = result.first;
        AllocationError error = result.second;
        
        if (error != AllocationError::SUCCESS) {
            std::lock_guard<std::mutex> lock(debugMutex);
            std::cout << "ALLOC FAIL: Thread=" << threadId << " ptr=" << ptr 
                      << " sz=" << size << " ind=" << index 
                      << " error=" << errorToString(error) << std::endl;
            totalFailures.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        
        uint32_t newValue = (static_cast<uint32_t>(size & 0xFFFF) << 16) | (threadId & 0xFFFF);
        uint32_t expected = 0;
        
        if (trackingArena[index].compare_exchange_strong(expected, newValue, std::memory_order_acq_rel)) {
            totalAllocations.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
        
        // CAS failed - slot was already occupied
        {
            std::lock_guard<std::mutex> lock(debugMutex);
            std::cout << "ALLOC CAS FAIL: Thread=" << threadId << " ptr=" << ptr 
                      << " sz=" << size << " ind=" << index 
                      << " expected=0 actual=" << expected 
                      << " (thread=" << (expected & 0xFFFF) 
                      << " size=" << ((expected >> 16) & 0xFFFF) << ")" << std::endl;
        }
        
        totalFailures.fetch_add(1, std::memory_order_relaxed);
        return false;
    } // uped
    
    // log a free with CAS
    bool recordFree(void* ptr, uint16_t threadId, TestSlabArena& arena) {
        auto result = getIndexForPointer(ptr, arena);
        size_t index = result.first;
        AllocationError error = result.second;
        
        if (error != AllocationError::SUCCESS) {
            std::lock_guard<std::mutex> lock(debugMutex);
            std::cout << "FREE FAIL: Thread=" << threadId << " ptr=" << ptr 
                      << " ind=" << index << " error=" << errorToString(error) << std::endl;
            return false; // Don't count as total failure since this is just tracking
        }

        uint32_t curr = trackingArena[index].load(std::memory_order_acquire);
        while (curr != 0) {
            // clear to 0; CAS updates `cur` on failure
            if (trackingArena[index].compare_exchange_weak(curr, 0, std::memory_order_acq_rel)) {
                totalFrees.fetch_add(1, std::memory_order_relaxed);
                return true;
            }
        }
        
        // Was already 0 - possible double free or race
        if (curr == 0) {
            std::lock_guard<std::mutex> lock(debugMutex);
            std::cout << "FREE CAS WARN: Thread=" << threadId << " ptr=" << ptr 
                      << " ind=" << index << " already freed (double-free or race)" << std::endl;
        }
        
        // Treat as success since the slot is clear
        return true;
    }
 
    
    // dump stats
    void getStats(size_t& allocs, size_t& frees, size_t& failures, size_t& leaks) {
        allocs = totalAllocations.load(std::memory_order_relaxed);
        frees = totalFrees.load(std::memory_order_relaxed);
        failures = totalFailures.load(std::memory_order_relaxed);
        
        leaks = 0;
        for (size_t i = 0; i < MAX_TRACKED_OBJECTS; i++) {
            if (trackingArena[i].load(std::memory_order_relaxed) != 0) {
                leaks++;
            }
        }
    }
    
    void reset() {
        totalAllocations.store(0);
        totalFrees.store(0);
        totalFailures.store(0);
        for(size_t i = 0; i < MAX_TRACKED_OBJECTS; i++) {
            trackingArena[i].store(0, std::memory_order_relaxed);
        }
    }
};
constexpr size_t ParallelTracker::MAX_TRACKED_OBJECTS;


// what each worker thread does
void workerThread(TestSlabArena& arena, ParallelTracker& tracker, 
                 uint16_t threadId, size_t iterations, 
                 std::atomic<bool>& shouldStop) {
    
    std::random_device rd;
    std::mt19937 gen(rd() ^ threadId);
    std::uniform_int_distribution<> sizeDist(8, 512);
    std::uniform_int_distribution<> actionDist(0, 100);
    std::uniform_int_distribution<> holdDist(0, 50);
    
    std::vector<std::pair<void*, size_t>> localAllocations;
    localAllocations.reserve(100);
    
    size_t localAllocs = 0, localFrees = 0;
    size_t localAllocatorFails = 0, localTrackerFails = 0;
    size_t localDataCorruption = 0, localFreeFails = 0;    

    for (size_t i = 0; i < iterations && !shouldStop.load(); i++) {
        int action = actionDist(gen);
        
        // 70% chance to alloc, 30% chance to free
        if (action < 70 || localAllocations.empty()) {
            // try alloc
            size_t objSize = sizeDist(gen);
            TestAllocator allocator(arena, objSize);
            
            void* ptr = allocator.alloc();
            if (ptr) {
                if (tracker.recordAllocation(ptr, objSize, threadId, arena)) {
                    localAllocations.push_back({ptr, objSize});
                    localAllocs++;
                    
                    // scribble some data so we can later check corruption
                    if (objSize >= 4) {
                        uint32_t* intPtr = static_cast<uint32_t*>(ptr);
                        *intPtr = (threadId << 16) | (i & 0xFFFF);
                    }
                } else {
                    // tracker failed, free it back right away
                    TestAllocator freeAllocator(arena, objSize);
                    freeAllocator.free(ptr);
                    localTrackerFails++;
                }
            } else {
                localAllocatorFails++;
            }
        } else {
            // pick a random alloc and free it
            if (!localAllocations.empty()) {
                std::uniform_int_distribution<> indexDist(0, localAllocations.size() - 1);
                size_t index = indexDist(gen);
                
                void* ptr = localAllocations[index].first;
                size_t objSize = localAllocations[index].second;
                
                // sanity check data
                if (objSize >= 4) {
                    uint32_t* intPtr = static_cast<uint32_t*>(ptr);
                    if ((*intPtr >> 16) != threadId) {
                        localDataCorruption++;
                        std::cout << "Thread " << threadId << ": data corruption detected!" << std::endl;
                    }
                }
                
                TestAllocator allocator(arena, objSize);
                if (allocator.free(ptr)) {
                    if (tracker.recordFree(ptr, threadId, arena)) {
                        localFrees++;
                    } else {
                        localTrackerFails++;
                    }
                } else {
                    localFreeFails++;
                }
                
                // pop it from local vector
                localAllocations[index] = localAllocations.back();
                localAllocations.pop_back();
            }
        }
        
        // sometimes just yield to mess with timing
        if (holdDist(gen) == 0) {
            std::this_thread::yield();
        }
    }
    
    // free whatever’s left
    for (const auto& alloc : localAllocations) {
        TestAllocator allocator(arena, alloc.second);
        if (allocator.free(alloc.first)) {
            tracker.recordFree(alloc.first, threadId, arena);
        }
    }
    
    std::cout << "Thread " << threadId << " done: " 
              << localAllocs << " allocs, " << localFrees << " frees, " 
              << localAllocatorFails << " alloc_fails, "
              << localTrackerFails << " tracker_fails, "
              << localDataCorruption << " corruptions, "
              << localFreeFails << " free_fails" << std::endl;
}

void testBasicParallel() {
    std::cout << "=== Basic Parallel Test ===" << std::endl;
    
    TestSlabArena arena;
    ParallelTracker tracker;
    
    const size_t numThreads = 4;
    const size_t iterationsPerThread = 1000;
    
    std::atomic<bool> shouldStop{false};
    std::vector<std::thread> threads;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // spin up threads
    for (size_t i = 0; i < numThreads; i++) {
        threads.emplace_back(workerThread, std::ref(arena), std::ref(tracker),
                            static_cast<uint16_t>(i + 1), iterationsPerThread, 
                            std::ref(shouldStop));
    }
    
    // wait for them
    for (auto& thread : threads) {
        thread.join();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // print stats
    size_t totalAllocs, totalFrees, totalFailures, totalLeaks;
    tracker.getStats(totalAllocs, totalFrees, totalFailures, totalLeaks);
    
    std::cout << "\nResults:" << std::endl;
    std::cout << "Duration: " << duration.count() << "ms" << std::endl;
    std::cout << "Allocs: " << totalAllocs << std::endl;
    std::cout << "Frees: " << totalFrees << std::endl;
    std::cout << "Failures: " << totalFailures << std::endl;
    std::cout << "Leaks: " << totalLeaks << std::endl;
    std::cout << "Success rate: " << (100.0 * (totalAllocs + totalFrees)) / 
                                      (totalAllocs + totalFrees + totalFailures) << "%" << std::endl;
    
    if (totalLeaks == 0) {
        std::cout << ":) no leaks" << std::endl;
    } else {
        std::cout << ":( leaks detected!" << std::endl;
    }
    
    std::cout << std::endl;
}

void testHighContentionParallel() {
    std::cout << "=== High contention parallel test ===" << std::endl;
    
    TestSlabArena arena;
    ParallelTracker tracker;
    
    const size_t numThreads = std::thread::hardware_concurrency() * 2;
    const size_t iterationsPerThread = 2000;
    
    std::atomic<bool> shouldStop{false};
    std::vector<std::thread> threads;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // spin up more threads than cores
    for (size_t i = 0; i < numThreads; i++) {
        threads.emplace_back(workerThread, std::ref(arena), std::ref(tracker),
                            static_cast<uint16_t>(i + 1), iterationsPerThread, 
                            std::ref(shouldStop));
    }
    
    // let it run a bit then stop
    std::this_thread::sleep_for(std::chrono::seconds(5));
    shouldStop.store(true);
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    size_t totalAllocs, totalFrees, totalFailures, totalLeaks;
    tracker.getStats(totalAllocs, totalFrees, totalFailures, totalLeaks);
    
    std::cout << "\nresults:" << std::endl;
    std::cout << "threads: " << numThreads << std::endl;
    std::cout << "duration: " << duration.count() << "ms" << std::endl;
    std::cout << "allocs: " << totalAllocs << std::endl;
    std::cout << "frees: " << totalFrees << std::endl;
    std::cout << "failures: " << totalFailures << std::endl;
    std::cout << "leaks: " << totalLeaks << std::endl;
    std::cout << "throughput: " << (totalAllocs + totalFrees) * 1000 / duration.count() 
              << " ops/sec" << std::endl;
    
    if (totalLeaks == 0) {
        std::cout << ":) No leaks under contention" << std::endl;
    } else {
        std::cout << ":( Leaks under contention!" << std::endl;
    }
    
    std::cout << std::endl;
}

void testStressTest() {
    std::cout << "=== Stress Test ===" << std::endl;
    
    TestSlabArena arena;
    ParallelTracker tracker;
    
    const size_t numThreads = 8;
    const size_t iterationsPerThread = 5000;
    
    std::atomic<bool> shouldStop{false};
    std::vector<std::thread> threads;
    
    // run a few rounds
    for (int round = 0; round < 3; round++) {
        std::cout << "Round " << (round + 1) << std::endl;
        tracker.reset();
        threads.clear();
        
        auto start = std::chrono::high_resolution_clock::now();
        
        for (size_t i = 0; i < numThreads; i++) {
            threads.emplace_back(workerThread, std::ref(arena), std::ref(tracker),
                                static_cast<uint16_t>(i + 1), iterationsPerThread, 
                                std::ref(shouldStop));
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        size_t totalAllocs, totalFrees, totalFailures, totalLeaks;
        tracker.getStats(totalAllocs, totalFrees, totalFailures, totalLeaks);
        
        std::cout << "  Round " << (round + 1) << ": " << duration.count() << "ms, "
                  << totalAllocs << " allocs, " << totalFrees << " frees, "
                  << totalLeaks << " leaks" << std::endl;
    }
    
    std::cout << ":) Stress test done!" << std::endl << std::endl;
}

int main() {
    std::cout << "Parallel Test" << std::endl;
    std::cout << "==============" << std::endl << std::endl;
    
    std::cout << "Hardware concurrency: " << std::thread::hardware_concurrency() << " threads" << std::endl;
    std::cout << "Test Slab Arena Size: " << TestSlabArena::SLAB_COUNT << " slabs" << std::endl;
    std::cout << "Slab Size: " << TestSlabArena::slabType::SIZE << " bytes" << std::endl << std::endl;
    
    try {
        testBasicParallel();
        testHighContentionParallel();
        testStressTest();
        
        std::cout << "All tests done!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown exception!" << std::endl;
        return 1;
    }
    
    return 0;
}
