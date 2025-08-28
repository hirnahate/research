#include <vector>
#include <thread> 
#include <atomic> 
#include <random> 
#include <chrono> 
#include <cassert> 
#include <unordered_set> 

#include "log.h"
#include "allocator.h"

// lets make better error codes

// global tracker for allocations/frees using an atomic bitmask
template <typename SIZE_TYPE>
class ParallelTracker {
public:
    //static constexpr size_t PER_SLAB_STRIDE = 512;
    static constexpr size_t MAX_TRACKED_OBJECTS = TestSlabArena::SLAB_COUNT * 1024;
private:
    // each entry: [31:16] = size, [15:0] = thread_id, 0 = free
    // std::atomic<uint32_t> trackingArena[MAX_TRACKED_OBJECTS];
    std::unique_ptr<uint32_t[]> trackingArena;
    size_t currentByteTotal   = 0;
    size_t minimumRefusalTotal = MAX_TRACKED_OBJECTS * 8;
    size_t totalAllocations   = 0;
    size_t totalFrees         = 0;
    size_t totalFailures      = 0;
    size_t reservationFailures= 0;

    
public:
    ParallelTracker(): trackingArena(new uint32_t[MAX_TRACKED_OBJECTS]) {
        for (size_t i = 0; i < MAX_TRACKED_OBJECTS; ++i) 
            intr::atomic::store_relaxed(&trackingArena[i], 0u);
        intr::atomic::store_relaxed(&totalAllocations, static_cast<size_t>(0));
        intr::atomic::store_relaxed(&totalFrees, static_cast<size_t>(0));
        intr::atomic::store_relaxed(&totalFailures, static_cast<size_t>(0));
        intr::atomic::store_relaxed(&reservationFailures, static_cast<size_t>(0));
        intr::atomic::store_relaxed(&currentByteTotal, static_cast<size_t>(0));
    }

    size_t getCurrentByteTotal() {
        return intr::atomic::load_relaxed(&currentByteTotal);
    }

    void logReservationFailure(){
        intr::atomic::add_system(&reservationFailures, size_t{1});
    }

    void logRefusalAtTotal(size_t total) {
        size_t expected = intr::atomic::load_relaxed(&minimumRefusalTotal);
        while (expected > total){
            size_t old = intr::atomic::CAS_system(&minimumRefusalTotal, expected, total);
            if(old == expected)
                break; // yay!s
            expected = old; // try again!
        }     
    }

    // figure out arena index for a pointer
    size_t getIndexForPtr(void* ptr, TestSlabArena& arena) {
        if (!ptr) 
            return MAX_TRACKED_OBJECTS;
        
        // find slab index holding ptr
        auto slabInd = arena.slabIndexFor(ptr);
        if(slabInd >= TestSlabArena::SLAB_COUNT)
            return MAX_TRACKED_OBJECTS;

        // find beginning - skip mask
        auto& slab = arena.slabAt(slabInd);
        char* slabStart = static_cast<char*>(static_cast<void*>(&slab));
        char* ptrChar = static_cast<char*>(ptr);

        size_t offset = ptrChar - slabStart;

        // find size and mask layout
        auto& proxy = arena.proxyAt(slabInd).data;
        size_t objectSz = proxy.getSize();

        if(objectSz == 0)
            return MAX_TRACKED_OBJECTS;
        
        // find mask overhead
        size_t maxObj = proxy.slabObjCount(objectSz);
        size_t maskElemCount = (maxObj + proxy.SLAB_ELEM_BIT_SIZE - 1) / proxy.SLAB_ELEM_BIT_SIZE;
        size_t maskOverhead = maskElemCount * sizeof(typename TestSlabArena::slabProxyType::allocMaskElem);

        if(offset < maskOverhead)
            return MAX_TRACKED_OBJECTS;
        
        // alignment check
        size_t objectOffset = offset - maskOverhead;
        if(objectOffset % objectSz != 0)
            return MAX_TRACKED_OBJECTS;

        size_t objectInd = objectOffset / objectSz;
        if(objectInd >= maxObj)
            return MAX_TRACKED_OBJECTS;

        size_t objectsPerSlab = proxy.slabObjCount(objectSz);
        size_t globalInd = slabInd * objectsPerSlab + objectInd;
        
        return globalInd;
    }
    
    // log an allocation with CAS
    bool recordAllocation(void* ptr, size_t size, uint16_t threadId, TestSlabArena& arena) {
        size_t index = getIndexForPtr(ptr, arena);
        printf("TRACKING  :  ptr=%p -> index=%zu (max=%zu)\n", ptr, index, MAX_TRACKED_OBJECTS);
        if (index >= MAX_TRACKED_OBJECTS) {
            Log() << "Failed: Allocation index for size "<< size 
                  << " at " << ptr <<" exceeds arena bounds!" << std::endl;
            return false;
        }

        uint32_t newValue = (static_cast<uint32_t>(size & 0xFFFF) << 16) | (threadId & 0xFFFF);
        uint32_t expected = 0;
        uint32_t old = intr::atomic::CAS_system(&trackingArena[index], expected, newValue);

        if (old == expected) {
            intr::atomic::add_system(&totalAllocations, size_t{1});
            intr::atomic::add_system(&currentByteTotal, size);
            return true;
        }

        uint32_t bad_size = (expected>>16) & 0xFFFF;
        uint32_t bad_id   = expected & 0xFFFF;
        Log() << "Failed to allocate size " << size << ". Expected 0, but found ("<<bad_size<<","<<bad_id<<")" << std::endl;
        intr::atomic::add_system(&totalFailures, size_t{1});
        return false;
    }

    
    // log a free with CAS
    bool recordFree(void* ptr, uint16_t threadId, TestSlabArena& arena) {
        size_t index = getIndexForPtr(ptr, arena);
        if (index >= MAX_TRACKED_OBJECTS) return false;
        
        uint32_t current = intr::atomic::load_acquire(&trackingArena[index]);
        
        // check if it was really allocated by this thread
        if ((current & 0xFFFF) != threadId || current == 0) {
            intr::atomic::add_system(&totalFailures, size_t{1});
            return false;
        }

        uint32_t old = intr::atomic::CAS_system(&trackingArena[index], current, 0u);
        if(old == current) {
            intr::atomic::add_system(&totalFrees, size_t{1});
            uint32_t size = (current >> 16) & 0xFFFF;
            intr::atomic::sub_system(&currentByteTotal, static_cast<size_t>(size));
            return true;
        }
        
        // CAS failed (raced)...tracker failure
        intr::atomic::add_system(&totalFailures, size_t{1});
        return false;
    } // end of recordFree
 
    
    // dump stats
    void getStats(size_t& allocs, size_t& frees, size_t& failures, size_t& leaks, size_t& reservationFails) {
        allocs = intr::atomic::load_relaxed(&totalAllocations);
        frees  = intr::atomic::load_relaxed(&totalFrees);
        failures = intr::atomic::load_relaxed(&totalFailures);
        reservationFails = intr::atomic::load_relaxed(&reservationFailures);

        leaks = 0;
        for (size_t i = 0; i < MAX_TRACKED_OBJECTS; i++) {
            if (intr::atomic::load_relaxed(&trackingArena[i]) != 0u) {
                leaks++;
            }
        }
    }
    
    void reset() {
        intr::atomic::store_relaxed(&totalAllocations, size_t(0));
        intr::atomic::store_relaxed(&totalFrees, size_t(0));
        intr::atomic::store_relaxed(&totalFailures, size_t(0));
        intr::atomic::store_relaxed(&reservationFailures, size_t(0));
        intr::atomic::store_relaxed(&currentByteTotal, size_t(0));
        intr::atomic::store_relaxed(&minimumRefusalTotal, MAX_TRACKED_OBJECTS * 8);

        for (size_t i = 0; i < MAX_TRACKED_OBJECTS; ++i) 
            intr::atomic::store_relaxed(&trackingArena[i], uint32_t(0));
    } // end of resets
};


// what each worker thread does
template<typename SIZE_TYPE>
void workerThread(TestSlabArena& arena, ParallelTracker<SIZE_TYPE>& tracker, 
                 uint16_t threadId, size_t iterations, 
                 std::atomic<bool>& shouldStop) {
    
    std::random_device rd;
    std::mt19937 gen(rd() ^ threadId);
    std::uniform_int_distribution<> sizeDist(3, 9);
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
            size_t objSize = 1 << sizeDist(gen);
            TestAllocator allocator(arena, objSize);
            
            void* ptr = allocator.alloc();
            Log() << "Allocated size " << objSize << " at " << ptr << std::endl;
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
                    Log() << "Failed to record allocation of size " << objSize << " at " << ptr << std::endl;
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
                Log() << "Deallocating at " << ptr << std::endl;
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
                        Log() << "Failed to record deallocation of size " << objSize << " at " << ptr << std::endl;
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
    
    TestSlabArena *arena_ptr = new TestSlabArena;
    if (!arena_ptr) {
        return;
    }
    TestSlabArena &arena = *arena_ptr;
    constexpr size_t OBJECT_COUNT = TestSlabArena::SLAB_COUNT * TestSlabArena::slabType::SIZE / 8;
    typedef Size<OBJECT_COUNT> SizeType;
    ParallelTracker<SizeType> *tracker_ptr = new ParallelTracker<SizeType>;
    ParallelTracker<SizeType> &tracker = *tracker_ptr;
    
    const size_t numThreads = 4;
    const size_t iterationsPerThread = 10000;
    
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
    size_t totalAllocs, totalFrees, totalFailures, totalLeaks, reservationFailures;
    tracker.getStats(totalAllocs, totalFrees, totalFailures, totalLeaks, reservationFailures);
    
    std::cout << "\nResults:" << std::endl;
    std::cout << "Duration: " << duration.count() << "ms" << std::endl;
    std::cout << "Allocs: " << totalAllocs << std::endl;
    std::cout << "Frees: " << totalFrees << std::endl;
    std::cout << "Tracking Failures: " << totalFailures << std::endl;
    std::cout << "Reservation failures: " << reservationFailures << std::endl;
    std::cout << "Leaks: " << totalLeaks << std::endl;
    std::cout << "Success rate: " << (100.0 * (totalAllocs + totalFrees)) / 
                                      (totalAllocs + totalFrees + totalFailures + reservationFailures) << "%" << std::endl;
    
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
    
    const size_t numThreads = 4; 
    std::thread::hardware_concurrency() * 2;
    const size_t iterationsPerThread = 500;
    
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
    
    size_t totalAllocs, totalFrees, totalFailures, totalLeaks, reservationFailures;
    tracker.getStats(totalAllocs, totalFrees, totalFailures, totalLeaks, reservationFailures);
    
    std::cout << "\nResults:" << std::endl;
    std::cout << "Threads: " << numThreads << std::endl;
    std::cout << "Duration: " << duration.count() << "ms" << std::endl;
    std::cout << "Allocs: " << totalAllocs << std::endl;
    std::cout << "Frees: " << totalFrees << std::endl;
    std::cout << "Tracking failures: " << totalFailures << std::endl;
    std::cout << "Reservation failures: " << reservationFailures << std::endl;
    std::cout << "Leaks: " << totalLeaks << std::endl;
    std::cout << "Throughput: " << (totalAllocs + totalFrees) * 1000 / duration.count() 
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
    
    const size_t numThreads = 4;
    const size_t iterationsPerThread = 500;
    
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
        
        size_t totalAllocs, totalFrees, totalFailures, totalLeaks, reservationFailures;
        tracker.getStats(totalAllocs, totalFrees, totalFailures, totalLeaks, reservationFailures);
        
        std::cout << "  Round " << (round + 1) << ": " << duration.count() << "ms, "
                  << totalAllocs << " allocs, " << totalFrees << " frees, "
                  << totalFailures << " tracking fails, " << reservationFailures << " res fails, "
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
