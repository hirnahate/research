#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <chrono>
//#include <thread>

#include "intr/mod.h"
#include "allocator.h"

__device__ 
uint32_t simpleRand(uint32_t* seed){
    *seed = *seed * 1664525u + 1013904223u;
    return *seed;
}

class GPUTracker {
public:
    static const size_t MAX_TRACKED_OBJECTS = 262144;
    uint32_t* trackingArena;
    uint32_t* stats;

    __device__ bool 
    recordAllocation(void* ptr, size_t size, uint32_t threadId, TestSlabArena* arena) {
        if (!ptr) 
            return false;

        // simple index calc for GPU
        size_t index = getIndexForPtr(ptr, arena);
        if (index >= MAX_TRACKED_OBJECTS) 
            return false;
        
        uint32_t newValue = (static_cast<uint32_t>(size & 0xFFFF) << 16) | (threadId & 0xFFFF);
        uint32_t expected = 0;
        uint32_t old = intr::atomic::CAS_system(&trackingArena[index], expected, newValue);
        
        if (old == expected) {
            intr::atomic::add_system(&stats[0], 1u); // allocs
            return true;
        }
        
        intr::atomic::add_system(&stats[2], 1u); // failures
        return false;
    } // end of recordAllo

    __device__ bool 
    recordFree(void* ptr, uint32_t threadId, TestSlabArena* arena) {
        if (!ptr) 
            return false;
        
        size_t index = getIndexForPtr(ptr, arena);
        if (index >= MAX_TRACKED_OBJECTS) 
            return false;
        
        uint32_t current = intr::atomic::load_relaxed(&trackingArena[index]);
        if ((current & 0xFFFF) != threadId || current == 0) {
            intr::atomic::add_system(&stats[2], 1u); // failures
            return false;
        }
        
        uint32_t old = intr::atomic::CAS_system(&trackingArena[index], current, 0u);
        if (old == current) {
            intr::atomic::add_system(&stats[1], 1u); // frees
            return true;
        }
        
        intr::atomic::add_system(&stats[2], 1u); // failures
        return false;
    } // end of recordfree

private:
    __device__ 
    size_t getIndexForPtr(void* ptr, TestSlabArena* arena) {
        if (!ptr) 
            return MAX_TRACKED_OBJECTS;
        
        auto slabInd = arena->slabIndexFor(ptr);
        if(slabInd >= TestSlabArena::SLAB_COUNT) 
            return MAX_TRACKED_OBJECTS;

        auto& proxy = arena->proxyAt(slabInd).data;
        size_t objectSz = proxy.getSize();
        if(objectSz == 0) 
            return MAX_TRACKED_OBJECTS;
        
        // calc object index within slab
        auto& slab = arena->slabAt(slabInd);
        char* slabStart = static_cast<char*>(static_cast<void*>(&slab));
        char* ptrChar = static_cast<char*>(ptr);
        size_t offset = ptrChar - slabStart;

        size_t maxObj = proxy.slabObjCount(objectSz);
        size_t maskElemCount = (maxObj + proxy.SLAB_ELEM_BIT_SIZE - 1) / proxy.SLAB_ELEM_BIT_SIZE;
        size_t maskOverhead = maskElemCount * sizeof(typename TestSlabArena::slabProxyType::allocMaskElem);

        if(offset < maskOverhead) 
            return MAX_TRACKED_OBJECTS;
        
        size_t objectOffset = offset - maskOverhead;
        if(objectOffset % objectSz != 0) 
            return MAX_TRACKED_OBJECTS;

        size_t objectInd = objectOffset / objectSz;
        if(objectInd >= maxObj) 
            return MAX_TRACKED_OBJECTS;

        // total index calculation
        size_t globalInd = 0;
        for(size_t i = 0; i < slabInd; i++) {
            auto& prevProxy = arena->proxyAt(i).data;
            size_t prevSize = prevProxy.getSize();
            if(prevSize > 0) {
                globalInd += prevProxy.slabObjCount(prevSize);
            } else {
                // safe estimate
                globalInd += 256; 
            }
            if(globalInd >= MAX_TRACKED_OBJECTS) 
                return MAX_TRACKED_OBJECTS;
        }
        globalInd += objectInd;
        
        return (globalInd < MAX_TRACKED_OBJECTS) ? globalInd : MAX_TRACKED_OBJECTS;
    }

}; // end of class

// GPU kernel for testing
__global__ 
void allocatorTestKernel(TestSlabArena* arena, GPUTracker* tracker, int iterations, uint32_t* shouldStop) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t seed = tid + 12345u; // simple seed
    
    // local allocation tracking
    void* localPtrs[64];        // Reduced for GPU stack limits
    size_t localSizes[64];
    int localCount = 0;
    
    for(int i = 0; i < iterations && !(*shouldStop); i++) {
        uint32_t action = simpleRand(&seed) % 100;
        
        if(action < 70 || localCount == 0) {
            // alloc
            uint32_t sizeChoice = simpleRand(&seed) % 7; // 0-6
            size_t objSize = 1 << (3 + sizeChoice);      // 8, 16, 32, 64, 128, 256, 512
            
            TestAllocator allocator(*arena, objSize);
            void* ptr = allocator.alloc();
            
            if(ptr && localCount < 64) {
                if(tracker->recordAllocation(ptr, objSize, tid, arena)) {
                    localPtrs[localCount] = ptr;
                    localSizes[localCount] = objSize;
                    localCount++;
                    
                    // write test pattern
                    if(objSize >= 4) {
                        uint32_t* intPtr = static_cast<uint32_t*>(ptr);
                        *intPtr = (tid << 16) | (i & 0xFFFF);
                    }
                } else {
                    // tracking failed, free immediately
                    TestAllocator freeAllocator(*arena, objSize);
                    freeAllocator.free(ptr);
                }
            }
        } else {
            // free random allocation
            if(localCount > 0) {
                uint32_t idx = simpleRand(&seed) % localCount;
                void* ptr = localPtrs[idx];
                size_t objSize = localSizes[idx];
                
                // check test pattern
                if(objSize >= 4) {
                    uint32_t* intPtr = static_cast<uint32_t*>(ptr);
                    uint32_t expected = (tid << 16) | ((i - (localCount - idx)) & 0xFFFF);
                    if(*intPtr != expected) {
                        printf(":( - GPU Thread %u: Data corruption detected!\n", tid);
                    }
                }
                
                TestAllocator allocator(*arena, objSize);
                if(allocator.free(ptr)) {
                    tracker->recordFree(ptr, tid, arena);
                }
                
                // remove from local array (swap with last)
                localPtrs[idx] = localPtrs[localCount-1];
                localSizes[idx] = localSizes[localCount-1];
                localCount--;
            }
        }
        
        // yield for better interleaving
        if((simpleRand(&seed) % 100) == 0) {
            __syncthreads();
        }
    }
    
    // cleanup
    for(int j = 0; j < localCount; j++) {
        TestAllocator allocator(*arena, localSizes[j]);
        if(allocator.free(localPtrs[j])) {
            tracker->recordFree(localPtrs[j], tid, arena);
        }
    }
} // end of alloc

// run GPU test
void runGPUAllocatorTest() {
    std::cout << "=== GPU Allocator Test ===" << std::endl;
    
    // Device memory allocation
    TestSlabArena* d_arena;
    GPUTracker* d_tracker;
    uint32_t* d_trackingArena;
    uint32_t* d_stats;
    uint32_t* d_shouldStop;
    
    cudaMalloc(&d_arena, sizeof(TestSlabArena));
    cudaMalloc(&d_tracker, sizeof(GPUTracker));
    cudaMalloc(&d_trackingArena, GPUTracker::MAX_TRACKED_OBJECTS * sizeof(uint32_t));
    cudaMalloc(&d_stats, 4 * sizeof(uint32_t));
    cudaMalloc(&d_shouldStop, sizeof(uint32_t));
    
    // Initialize device memory
    cudaMemset(d_trackingArena, 0, GPUTracker::MAX_TRACKED_OBJECTS * sizeof(uint32_t));
    cudaMemset(d_stats, 0, 4 * sizeof(uint32_t));
    
    uint32_t stopFlag = 0;
    cudaMemcpy(d_shouldStop, &stopFlag, sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    // Initialize arena on device
    TestSlabArena h_arena;
    cudaMemcpy(d_arena, &h_arena, sizeof(TestSlabArena), cudaMemcpyHostToDevice);
    
    // Initialize tracker structure on device
    GPUTracker h_tracker;
    h_tracker.trackingArena = d_trackingArena;
    h_tracker.stats = d_stats;
    cudaMemcpy(d_tracker, &h_tracker, sizeof(GPUTracker), cudaMemcpyHostToDevice);
    
    // Launch parameters
    const int threadsPerBlock = 256;
    const int numBlocks = 16;        // 4096 total threads
    const int iterations = 1000;
    
    std::cout << "Launching " << (numBlocks * threadsPerBlock) << " GPU threads..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Launch kernel
    allocatorTestKernel<<<numBlocks, threadsPerBlock>>>(d_arena, d_tracker, iterations, d_shouldStop);
    
    // Wait for completion
    cudaError_t result = cudaDeviceSynchronize();
    if(result != cudaSuccess) {
        std::cerr << ":( - CUDA error: " << cudaGetErrorString(result) << std::endl;
        return;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Get results back to host
    uint32_t h_stats[4];
    cudaMemcpy(h_stats, d_stats, 4 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    // Count leaks by scanning tracking arena
    uint32_t* h_trackingArena = new uint32_t[GPUTracker::MAX_TRACKED_OBJECTS];
    cudaMemcpy(h_trackingArena, d_trackingArena, 
               GPUTracker::MAX_TRACKED_OBJECTS * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    uint32_t leaks = 0;
    for(size_t i = 0; i < GPUTracker::MAX_TRACKED_OBJECTS; i++) {
        if(h_trackingArena[i] != 0) leaks++;
    }
    
    // Print results
    std::cout << "\nGPU Results:" << std::endl;
    std::cout << "Duration: " << duration.count() << "ms" << std::endl;
    std::cout << "Threads: " << (numBlocks * threadsPerBlock) << std::endl;
    std::cout << "Allocs: " << h_stats[0] << std::endl;
    std::cout << "Frees: " << h_stats[1] << std::endl;
    std::cout << "Failures: " << h_stats[2] << std::endl;
    std::cout << "Leaks: " << leaks << std::endl;
    std::cout << "Throughput: " << ((h_stats[0] + h_stats[1]) * 1000) / duration.count() << " ops/sec" << std::endl;
    
    uint32_t totalOps = h_stats[0] + h_stats[1] + h_stats[2];
    if(totalOps > 0) {
        std::cout << "Success rate: " << (100.0 * (h_stats[0] + h_stats[1])) / totalOps << "%" << std::endl;
    }
    
    if(leaks == 0) {
        std::cout << ":) No leaks detected" << std::endl;
    } else {
        std::cout << ":( " << leaks << " leaks detected!" << std::endl;
    }
    
    // Cleanup
    delete[] h_trackingArena;
    cudaFree(d_arena);
    cudaFree(d_tracker);
    cudaFree(d_trackingArena);
    cudaFree(d_stats);
    cudaFree(d_shouldStop);
    
    std::cout << std::endl;
} // end of run


__global__ 
void stressTestKernel(TestSlabArena* arena, GPUTracker* tracker, int maxIterations, uint32_t* shouldStop) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t seed = tid + 54321u;
    
    void* localPtrs[32];  // smaller for stress test
    size_t localSizes[32];
    int localCount = 0;
    
    for(int i = 0; i < maxIterations; i++) {
        // check stop condition every 100 iterations
        if(i % 100 == 0 && *shouldStop) 
            break;
        
        uint32_t action = simpleRand(&seed) % 100;
        
        if(action < 80 || localCount == 0) {
            // higher allocation rate for stress
            uint32_t sizeChoice = simpleRand(&seed) % 6; // 0-5
            size_t objSize = 1 << (3 + sizeChoice);      // 8 to 256 bytes
            
            TestAllocator allocator(*arena, objSize);
            void* ptr = allocator.alloc();
            
            if(ptr && localCount < 32) {
                if(tracker->recordAllocation(ptr, objSize, tid, arena)) {
                    localPtrs[localCount] = ptr;
                    localSizes[localCount] = objSize;
                    localCount++;
                    
                    // write test pattern
                    if(objSize >= 4) {
                        uint32_t* intPtr = static_cast<uint32_t*>(ptr);
                        *intPtr = (tid << 16) | (i & 0xFFFF);
                    }
                }
            }
        } else {
            // free
            if(localCount > 0) {
                uint32_t idx = simpleRand(&seed) % localCount;
                void* ptr = localPtrs[idx];
                size_t objSize = localSizes[idx];
                
                TestAllocator allocator(*arena, objSize);
                if(allocator.free(ptr)) {
                    tracker->recordFree(ptr, tid, arena);
                }
                
                // remove from array
                localPtrs[idx] = localPtrs[localCount-1];
                localSizes[idx] = localSizes[localCount-1];
                localCount--;
            }
        }
        
        // sync for better race condition exposure
        if((simpleRand(&seed) % 1000) == 0) {
            __syncthreads();
        }
    }
    
    // cleanup
    for(int j = 0; j < localCount; j++) {
        TestAllocator allocator(*arena, localSizes[j]);
        if(allocator.free(localPtrs[j])) {
            tracker->recordFree(localPtrs[j], tid, arena);
        }
    }
} // end of stressK

void runGPUStressTest() {
    std::cout << "=== GPU Stress Test ===" << std::endl;
    
    // allocate device memory
    TestSlabArena* d_arena;
    GPUTracker* d_tracker;
    uint32_t* d_trackingArena;
    uint32_t* d_stats;
    uint32_t* d_shouldStop;
    
    cudaMalloc(&d_arena, sizeof(TestSlabArena));
    cudaMalloc(&d_tracker, sizeof(GPUTracker));
    cudaMalloc(&d_trackingArena, GPUTracker::MAX_TRACKED_OBJECTS * sizeof(uint32_t));
    cudaMalloc(&d_stats, 4 * sizeof(uint32_t));
    cudaMalloc(&d_shouldStop, sizeof(uint32_t));
    
    // Initialize
    cudaMemset(d_trackingArena, 0, GPUTracker::MAX_TRACKED_OBJECTS * sizeof(uint32_t));
    cudaMemset(d_stats, 0, 4 * sizeof(uint32_t));
    
    uint32_t stopFlag = 0;
    cudaMemcpy(d_shouldStop, &stopFlag, sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    TestSlabArena h_arena;
    cudaMemcpy(d_arena, &h_arena, sizeof(TestSlabArena), cudaMemcpyHostToDevice);
    
    GPUTracker h_tracker;
    h_tracker.trackingArena = d_trackingArena;
    h_tracker.stats = d_stats;
    cudaMemcpy(d_tracker, &h_tracker, sizeof(GPUTracker), cudaMemcpyHostToDevice);
    
    const int threadsPerBlock = 512;
    const int numBlocks = 32;        // 16384 threads - high contention
    const int maxIterations = 2000;
    
    std::cout << "Launching " << (numBlocks * threadsPerBlock) << " threads for stress test..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // launch 
    stressTestKernel<<<numBlocks, threadsPerBlock>>>(d_arena, d_tracker, maxIterations, d_shouldStop);
    
    // let it run for 3 seconds then stop
    auto sleepStart = std::chrono::high_resolution_clock::now();
    while(true) {
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - sleepStart);
        if(elapsed.count() >= 3) break;
    }    
    stopFlag = 1;
    cudaMemcpy(d_shouldStop, &stopFlag, sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    cudaDeviceSynchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Get results
    uint32_t h_stats[4];
    cudaMemcpy(h_stats, d_stats, 4 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    uint32_t* h_trackingArena = new uint32_t[GPUTracker::MAX_TRACKED_OBJECTS];
    cudaMemcpy(h_trackingArena, d_trackingArena, 
               GPUTracker::MAX_TRACKED_OBJECTS * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    uint32_t leaks = 0;
    for(size_t i = 0; i < GPUTracker::MAX_TRACKED_OBJECTS; i++) {
        if(h_trackingArena[i] != 0) leaks++;
    }
    
    std::cout << "\nGPU Stress Results:" << std::endl;
    std::cout << "Duration: " << duration.count() << "ms" << std::endl;
    std::cout << "Total Threads: " << (numBlocks * threadsPerBlock) << std::endl;
    std::cout << "Allocs: " << h_stats[0] << std::endl;
    std::cout << "Frees: " << h_stats[1] << std::endl;
    std::cout << "Failures: " << h_stats[2] << std::endl;
    std::cout << "Leaks: " << leaks << std::endl;
    std::cout << "Throughput: " << ((h_stats[0] + h_stats[1]) * 1000) / duration.count() << " ops/sec" << std::endl;
    
    if(leaks == 0) {
        std::cout << ":) No leaks under extreme GPU contention" << std::endl;
    } else {
        std::cout << ":( Leaks detected under GPU stress!" << std::endl;
    }
    
    delete[] h_trackingArena;
    cudaFree(d_arena);
    cudaFree(d_tracker);
    cudaFree(d_trackingArena);
    cudaFree(d_stats);
    cudaFree(d_shouldStop);
    
    std::cout << std::endl;
} // end of stressGPU

// comparison test: CPU vs GPU
void runComparisonTest() {
    std::cout << "=== CPU vs GPU Comparison ===" << std::endl;
    
    // CPU first (simplified)
    TestSlabArena cpuArena;
    const size_t cpuAllocs = 10000;
    void* cpuPtrs[1000];
    size_t cpuSizes[1000];
    size_t cpuCount = 0;


    auto cpu_start = std::chrono::high_resolution_clock::now();
    
    
    for(size_t i = 0; i < cpuAllocs && cpuCount < 1000; i++) {
        size_t objSize = 1 << (3 + (i % 7));  // 8, 16, 32, ..., 512
        TestAllocator allocator(cpuArena, objSize);
        void* ptr = allocator.alloc();
        if(ptr) {
            cpuPtrs[cpuCount] = ptr;
            cpuSizes[cpuCount] = objSize;
            cpuCount++;
        }
    }
    
    // free all
    for(size_t i = 0; i < cpuCount; i++) {
        TestAllocator allocator(cpuArena, cpuSizes[i]);
        allocator.free(cpuPtrs[i]);
    }
    
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start);
    
    std::cout << "CPU Results:" << std::endl;
    std::cout << "Duration: " << cpu_duration.count() << "ms" << std::endl;
    std::cout << "Successful Allocs: " << cpuCount << std::endl;
    std::cout << "CPU Throughput: " << (cpuCount * 2 * 1000) / cpu_duration.count() << " ops/sec" << std::endl;
    std::cout << std::endl;

    // Now run GPU version
    runGPUAllocatorTest();
} // end of compare

int main() {

    #ifndef GPU_ONLY
        runComparisonTest();  // Skip CPU comparison
    #endif

    std::cout << "GPU Allocator Testing" << std::endl;
    std::cout << "=====================" << std::endl << std::endl;
    
    // Check GPU capabilities
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if(deviceCount == 0) {
        std::cout << "No CUDA devices found. Exiting." << std::endl;
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Using GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Multiprocessors: " << prop.multiProcessorCount << std::endl << std::endl;
    
    try {
        runGPUAllocatorTest();
        runGPUStressTest();
        runComparisonTest();
        
        std::cout << "All GPU tests completed!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} // end of main