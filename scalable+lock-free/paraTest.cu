#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <random>
#include <chrono>
#include <cassert>
#include <unordered_set>
#include <mutex>

#include "log.h"
#include "allocator.h"



// global tracker for allocations/frees using an atomic bitmask
template <typename SIZE_TYPE>
class ParallelTracker {
public:
    static const size_t MAX_TRACKED_OBJECTS = SIZE_TYPE::VALUE;
    
private:
    // each entry: [31:16] = size, [15:0] = thread_id, 0 = free
    std::atomic<uint32_t> trackingArena[MAX_TRACKED_OBJECTS];
    std::atomic<size_t> currentByteTotal{0};
    std::atomic<size_t> minimumRefusalTotal{MAX_TRACKED_OBJECTS*8};

    std::atomic<size_t> totalAllocations{0};
    std::atomic<size_t> totalFrees{0};
    std::atomic<size_t> totalFailures{0};

    mutable std::mutex debugMutex;
    
public:

    ParallelTracker() {
        for(size_t i = 0; i < MAX_TRACKED_OBJECTS; i++) {

            trackingArena[i].store(0, std::memory_order_relaxed);
        }
    }


    size_t getCurrentByteTotal() {
        return currentByteTotal.load();
    }

    void logRefusalAtTotal(size_t total) {
        size_t expected = minimumRefusalTotal.load();
        while ( (expected > total) && (minimumRefusalTotal.compare_exchange_strong(expected,total,std::memory_order_acq_rel)) ) {}
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

        if (!ptr) return MAX_TRACKED_OBJECTS;
        
        char* ptrChar = static_cast<char*>(ptr);
        char* arenaBase = static_cast<char*>(static_cast<void*>(&arena));
        
        size_t offset = ptrChar - arenaBase;
        size_t index = offset / 8; // assuming 64-byte chunks
        
        return (index < MAX_TRACKED_OBJECTS) ? index : MAX_TRACKED_OBJECTS;

    }
    
    // log an allocation with CAS
    bool recordAllocation(void* ptr, size_t size, uint16_t threadId, TestSlabArena& arena) {
        size_t index = getIndexForPtr(ptr, arena);
        if (index >= MAX_TRACKED_OBJECTS) {
            Log() << "Failed: Allocation index for size "<< size << " at " << ptr <<" exceeds arena bounds!" << std::endl;

            return false;
        }
        
        uint32_t newValue = (static_cast<uint32_t>(size & 0xFFFF) << 16) | (threadId & 0xFFFF);
        uint32_t expected = 0;
        
        if (trackingArena[index].compare_exchange_strong(expected, newValue, std::memory_order_acq_rel)) {
            totalAllocations.fetch_add(1, std::memory_order_relaxed);
            currentByteTotal.fetch_add(size, std::memory_order_relaxed);
            return true;
        }
        uint32_t bad_size = (expected>>16) & 0xFFFF;
        uint32_t bad_id   = expected & 0xFFFF;
        Log() << "Failed to allocate size " << size << ". Expected 0, but found ("<<bad_size<<","<<bad_id<<")" << std::endl;

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
        
        if (trackingArena[index].compare_exchange_strong(current, 0, std::memory_order_acq_rel)) {
            totalFrees.fetch_add(1, std::memory_order_relaxed);
            uint32_t size = (current>>16) & 0xFFFF;
            currentByteTotal.fetch_sub(size, std::memory_order_relaxed);
            return true;
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
template<typename SIZE_TYPE>
void workerThread(TestSlabArena& arena, ParallelTracker<SIZE_TYPE>& tracker, 
                 uint16_t threadId, size_t iterations, 
                 std::atomic<bool>& shouldStop) {
    
    std::random_device rd;
    std::mt19937 gen(rd() ^ threadId);
    std::uniform_int_distribution<> sizeDist(3,9);
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
            size_t objSize = 1<<sizeDist(gen);
            TestAllocator allocator(arena, objSize);
            
            void* ptr = allocator.alloc();
            //Log() << "Allocated size " << objSize << " at " << ptr << std::endl;
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
                    Log() << "Failed to record allocation of size " << objSize << " at " << ptr << std::endl;
                    // tracker failed, free it back right away
                    TestAllocator freeAllocator(arena, objSize);
                    freeAllocator.free(ptr);
                    localTrackerFails++;
                }
            } else {
                size_t total = tracker.getCurrentByteTotal();
                tracker.logRefusalAtTotal(total);
                float proportion = ((float)total) / (ParallelTracker<SIZE_TYPE>::MAX_TRACKED_OBJECTS*8.0f);
                Log() << "Failed to  allocate object of size " << objSize << " with "
                      << (100.0*(1.0-proportion)) << "% capacity left"<<  std::endl;

            }
        } else {
            // pick a random alloc and free it
            if (!localAllocations.empty()) {
                std::uniform_int_distribution<> indexDist(0, localAllocations.size() - 1);
                size_t index = indexDist(gen);
                
                void* ptr = localAllocations[index].first;
                //Log() << "Deallocating at " << ptr << std::endl;
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
                        localErrors++;
                        Log() << "Failed to record deallocation of size " << objSize << " at " << ptr << std::endl;
                    }
                } else {
                    localErrors++;
                    Log() << "Failed to deallocate object of size " << objSize << " at " << ptr << std::endl;
                        //localTrackerFails++;
             
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
    
    TestSlabArena *arena_ptr = new TestSlabArena;
    if (!arena_ptr) {
        return;
    }
    TestSlabArena &arena = *arena_ptr;
    constexpr size_t OBJECT_COUNT = TestSlabArena::SLAB_COUNT * TestSlabArena::slabType::SIZE / 8;
    typedef Size<OBJECT_COUNT> SizeType;
    ParallelTracker<SizeType> *tracker_ptr = new ParallelTracker<SizeType>;
    ParallelTracker<SizeType> &tracker = *tracker_ptr;
    
    const size_t numThreads = 2;
    const size_t iterationsPerThread = 1024;
    
    std::atomic<bool> shouldStop{false};
    std::vector<std::thread> threads;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // spin up threads
    for (size_t i = 0; i < numThreads; i++) {
        threads.emplace_back(workerThread<SizeType>, std::ref(arena), std::ref(tracker),
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
    delete arena_ptr;
}

void testHighContentionParallel() {
    std::cout << "=== High contention parallel test ===" << std::endl;
    
    TestSlabArena arena;
    constexpr size_t OBJECT_COUNT = TestSlabArena::SLAB_COUNT * TestSlabArena::slabType::SIZE / 8;
    typedef Size<OBJECT_COUNT> SizeType;
    ParallelTracker<SizeType> tracker;
    
    const size_t numThreads = 1; std::thread::hardware_concurrency() * 2;
    const size_t iterationsPerThread = 1;
    
    std::atomic<bool> shouldStop{false};
    std::vector<std::thread> threads;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // spin up more threads than cores
    for (size_t i = 0; i < numThreads; i++) {
        threads.emplace_back(workerThread<SizeType>, std::ref(arena), std::ref(tracker),
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
    constexpr size_t OBJECT_COUNT = TestSlabArena::SLAB_COUNT * TestSlabArena::slabType::SIZE / 8;
    typedef Size<OBJECT_COUNT> SizeType;
    ParallelTracker<SizeType> tracker;
    
    const size_t numThreads = 1;
    const size_t iterationsPerThread = 1;
    
    std::atomic<bool> shouldStop{false};
    std::vector<std::thread> threads;
    
    // run a few rounds
    for (int round = 0; round < 1; round++) {
        std::cout << "Round " << (round + 1) << std::endl;
        tracker.reset();
        threads.clear();
        
        auto start = std::chrono::high_resolution_clock::now();
        
        for (size_t i = 0; i < numThreads; i++) {
            threads.emplace_back(workerThread<SizeType>, std::ref(arena), std::ref(tracker),
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
        //testHighContentionParallel();
        //testStressTest();
        
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
