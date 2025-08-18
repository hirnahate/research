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

// global tracker for allocations/frees using an atomic bitmask
class ParallelTracker {
public:
    static const size_t MAX_TRACKED_OBJECTS = 100000;
    
private:
    // each entry: [31:16] = size, [15:0] = thread_id, 0 = free
    std::atomic<uint32_t> trackingArena[MAX_TRACKED_OBJECTS];
    std::atomic<size_t> totalAllocations{0};
    std::atomic<size_t> totalFrees{0};
    std::atomic<size_t> totalFailures{0};
    
public:
    ParallelTracker() {
        for(size_t i = 0; i < MAX_TRACKED_OBJECTS; i++) {
            trackingArena[i].store(0, std::memory_order_relaxed);
        }
    }
    
    // figure out arena index for a pointer
    size_t getIndexForPtr(void* ptr, TestSlabArena& arena) {
        if (!ptr) return MAX_TRACKED_OBJECTS;
        
        char* ptrChar = static_cast<char*>(ptr);
        char* arenaBase = static_cast<char*>(static_cast<void*>(&arena));
        
        size_t offset = ptrChar - arenaBase;
        size_t index = offset / 64; // assuming 64-byte chunks
        
        return (index < MAX_TRACKED_OBJECTS) ? index : MAX_TRACKED_OBJECTS;
    }
    
    // log an allocation with CAS
    bool recordAllocation(void* ptr, size_t size, uint16_t threadId, TestSlabArena& arena) {
        size_t index = getIndexForPtr(ptr, arena);
        if (index >= MAX_TRACKED_OBJECTS) return false;
        
        uint32_t newValue = (static_cast<uint32_t>(size & 0xFFFF) << 16) | (threadId & 0xFFFF);
        uint32_t expected = 0;
        
        if (trackingArena[index].compare_exchange_strong(expected, newValue, std::memory_order_acq_rel)) {
            totalAllocations.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
        
        totalFailures.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    
    // log a free with CAS
    bool recordFree(void* ptr, uint16_t threadId, TestSlabArena& arena) {
        size_t index = getIndexForPtr(ptr, arena);
        if (index >= MAX_TRACKED_OBJECTS) return false;
        
        uint32_t current = trackingArena[index].load(std::memory_order_acquire);
        
        // check if it was really allocated by this thread
        if ((current & 0xFFFF) != threadId || current == 0) {
            totalFailures.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        
        if (trackingArena[index].compare_exchange_strong(current, 0, std::memory_order_acq_rel)) {
            totalFrees.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
        
        totalFailures.fetch_add(1, std::memory_order_relaxed);
        return false;
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
    
    size_t localAllocs = 0, localFrees = 0, localErrors = 0;
    
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
                    localErrors++;
                }
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
                        localErrors++;
                        std::cout << "Thread " << threadId << ": data corruption detected!" << std::endl;
                    }
                }
                
                TestAllocator allocator(arena, objSize);
                if (allocator.free(ptr)) {
                    if (tracker.recordFree(ptr, threadId, arena)) {
                        localFrees++;
                    } else {
                        localErrors++;
                    }
                } else {
                    localErrors++;
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
              << localErrors << " errors" << std::endl;
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
    std::cout << "=== high contention parallel test ===" << std::endl;
    
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
