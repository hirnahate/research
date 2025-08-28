#pragma once
#include "intr/mod.h"
#include <cstddef>
#include <cstdint>
//#include <device_launch_parameters.h>

#ifdef __CUDACC__
    #include <cuda_runtime.h>
    #include <device_launch_parameters.h>
#else
    #define __host__
    #define __device__
#endif

typedef unsigned long long int defaultAllocMaskElem;
typedef unsigned long long int defaultSlabAddr;

const size_t DEFAULT_SLAB_SIZE = 1 << 16;       // 64KB

template <size_t n>
struct Size {
    static const size_t VALUE = n;
};

template <typename type>
struct AdrInfo {
    __host__ __device__
    static type null(){
        return static_cast<type>(-1);
    }
};


// basic slab structure
template <size_t SLAB_SIZE = DEFAULT_SLAB_SIZE, typename ELEM_TYPE = defaultAllocMaskElem>
struct Slab {
    typedef ELEM_TYPE allocMaskElem;

    static const size_t ELEM_SIZE = sizeof(ELEM_TYPE);
    static const size_t SIZE = SLAB_SIZE;
    static const size_t ELEM_COUNT = (SLAB_SIZE + ELEM_SIZE - 1) / ELEM_SIZE;

    ELEM_TYPE data[ELEM_COUNT];
}; 

// slab proxy  
template <typename SLAB_TYPE>
struct defaultSlabProxy {
    typedef SLAB_TYPE slabType;
    typedef typename SLAB_TYPE::allocMaskElem allocMaskElem;
    typedef unsigned long long int AllocState;

    static const AllocState OFFSET = sizeof(AllocState) * 8 / 2;
    static const AllocState COUNT_MASK = (((AllocState)1) << OFFSET) - (AllocState)1;
    static const AllocState SIZE_MASK = ~COUNT_MASK;
    
    static const size_t SLAB_SIZE = SLAB_TYPE::SIZE;
    static const size_t SLAB_ELEM_COUNT = SLAB_TYPE::ELEM_COUNT;
    static const size_t SLAB_ELEM_BIT_SIZE = sizeof(allocMaskElem) * 8;         // check sizeof

    enum SlabState : uint32_t   {
        FREE = 0,
        PARTIAL = 1,
        FULL = 2
    };

    allocMaskElem allocMask;
    AllocState allocState;
    uint32_t reservationState;

    __host__ __device__
    size_t getSize()
    { return (intr::atomic::load_relaxed(&allocState) & SIZE_MASK) >> OFFSET;}

    __host__ __device__
    size_t getCount()
    { return intr::atomic::load_relaxed(&allocState) & COUNT_MASK;}

    __host__ __device__
    size_t isEmpty()
    { return getCount() == 0; }

    __host__ __device__
    bool isFull(){
        size_t objSize = getSize();
        if(objSize == 0)
            return false;

        size_t currentCount = getCount();
        size_t maxCount = slabObjCount(objSize);
                
        return (currentCount >= maxCount);
    } // end of isFull
    
    __host__ __device__
    bool isAvailable(){
        uint32_t state = intr::atomic::load_acquire(&reservationState);
        return (state == PARTIAL && !isFull());
    }

    __host__ __device__
    bool tryClaim(size_t objectSz){
        AllocState expected = 0;
        AllocState newState = ((AllocState)objectSz) << OFFSET;

        AllocState old = intr::atomic::CAS_system(&allocState, expected, newState);
        if(old == expected){
            intr::atomic::store_relaxed(&reservationState, static_cast<uint32_t>(PARTIAL));
            allocMask = 0;
            return true;
        }

        AllocState current = intr::atomic::load_relaxed(&allocState);
        size_t currentSz = (old & SIZE_MASK) >> OFFSET;
        uint32_t state = intr::atomic::load_relaxed(&reservationState);

        return (currentSz == objectSz) && (state == PARTIAL);
    } // end of tryClaim

    __host__ __device__
    bool clearAllocState(){
        AllocState current = intr::atomic::load_relaxed(&allocState);

        while(true){
            size_t count = current & COUNT_MASK;
            if(count != 0)
                return false;
            size_t size = (current & SIZE_MASK) >> OFFSET;
            if(size == 0)
                return false;

            AllocState old = intr::atomic::CAS_system(&allocState, current, 0ull);
            if(old == current){
                intr::atomic::store_relaxed(&reservationState, static_cast<uint32_t>(FREE));
                return true;
            }
            current = old;
        }// end while
    } // end 

    __host__ __device__
    size_t slabObjCount(size_t objectSize){
        if(objectSize <= 0)
            return 0;           // lets say yes...

        size_t maxObjects = SLAB_SIZE / objectSize;
        if(maxObjects <= SLAB_ELEM_BIT_SIZE)
            return maxObjects;
        
        size_t maskElem = (maxObjects + SLAB_ELEM_BIT_SIZE - 1) / SLAB_ELEM_BIT_SIZE;
        size_t maskOverhead = maskElem * sizeof(allocMaskElem);

        if(maskOverhead >= SLAB_SIZE)
            return 0;

        size_t openSlabs = SLAB_SIZE - maskOverhead;
        size_t realObjs = openSlabs / objectSize;

        return realObjs;
    } // end of slabObjCount

    __host__ __device__
    bool tryAllocateBit(allocMaskElem& elem, size_t& bitIndex){
        for(size_t attempts = 0; attempts < 64; attempts++){
            allocMaskElem currentMask = intr::atomic::load_acquire(&elem);

            allocMaskElem freeMask = ~currentMask;
            if(freeMask == 0)
                return false;

            #if defined(__CUDA_ARCH__)
                int ff = __ffsll(static_cast<unsigned long long>(freeMask));
                if(ff == 0)
                    return false;
                size_t target = static_cast<size_t>(ff - 1);
            #else
                size_t target = static_cast<size_t>(__builtin_ctzll(static_cast<unsigned long long>(freeMask)));
            #endif

            if(target >= SLAB_ELEM_BIT_SIZE)
                return false;

            allocMaskElem targetBit = ((allocMaskElem)1) << target;
            allocMaskElem newMask = currentMask | targetBit;

            allocMaskElem old = intr::atomic::CAS_system(&elem, currentMask, newMask);
            if(old == currentMask){
                bitIndex = target;
                return true;
            }
        }
        return false;
    } // end of tryAllocBit


    __host__ __device__
    void clearSlab(SLAB_TYPE* slab, size_t maskElemCount){
        for(size_t i = 0; i < maskElemCount; i++)
            slab->data[i] = 0;
    }

    __host__ __device__
    bool attemptMaskAlloc(allocMaskElem& elem, size_t& result){
        for(size_t attempts = 0; attempts < SLAB_ELEM_BIT_SIZE; attempts++){
            allocMaskElem currentMask = elem;
            
            // find first free bit
            allocMaskElem freeMask = ~currentMask;
            if(freeMask == 0) 
                return false;
            
            #if defined(__CUDA_ARCH__)
                // if CUDA device ... use __ffsll
                int ff = __ffsll(static_cast<unsigned long long>(freeMask));
                if(ff == 0) return false;
                    size_t target = static_cast<size_t>(ff - 1);
            #else
                // if host ... use compiler builtin
                size_t target = static_cast<size_t>(__builtin_ctzll(static_cast<unsigned long long>(freeMask)));
            #endif

            if(target >= SLAB_ELEM_BIT_SIZE) 
                return false;
            
            allocMaskElem targetBit = ((allocMaskElem)1) << target;
            allocMaskElem expected = currentMask;
            
            allocMaskElem old = intr::atomic::CAS_system(&elem, expected, currentMask | targetBit);
            if(old == expected){
                result = target;
                return true;
            }
            printf(":( - CAS failed(proxy) - Expected:0x%llx Got:0x%llx (retry %zu)\n", 
               (unsigned long long)expected, (unsigned long long)old, attempts);

        }
        return false;
    } // end of attempt

    __host__ __device__
    void* indexToPtr(SLAB_TYPE* slab, size_t objectSize, size_t maskElemCount, size_t ind){
        char* bytePtr = static_cast<char*>(static_cast<void*>(slab));
        bytePtr += sizeof(allocMaskElem) * maskElemCount;
        bytePtr += objectSize * ind;

        return static_cast<void*>(bytePtr);
    } // end

    __host__ __device__
    void* alloc(SLAB_TYPE* slab, bool& slabFilled){
        // just in case...
        uint32_t currentReservState = intr::atomic::load_relaxed(&reservationState);
        if(currentReservState == FULL)
            return nullptr;
        
        AllocState currentState = intr::atomic::load_relaxed(&allocState);
        size_t objectSz = (currentState & SIZE_MASK) >> OFFSET;
        size_t maxObjCount = slabObjCount(objectSz);

        if(maxObjCount == 0)
            return nullptr;

        size_t maskElemCount = (maxObjCount + SLAB_ELEM_BIT_SIZE - 1) / SLAB_ELEM_BIT_SIZE;

        // try to allocate a bit..
        size_t bitIndex;
        bool bitAllocated = false;
        size_t globalIndex = 0;

        if(maxObjCount <= SLAB_ELEM_BIT_SIZE){
            // single mask elem
            if(tryAllocateBit(allocMask, bitIndex)){
                bitAllocated = true;
                globalIndex = bitIndex;
            }
        } else {
            // multi mask elements  
            for(size_t i = 0; i < maskElemCount; i++){
                if(tryAllocateBit(slab->data[i], bitIndex)){
                    bitAllocated = true;
                    globalIndex = SLAB_ELEM_BIT_SIZE * i + bitIndex;
                    break;
                }
            }
        }
        
        if(!bitAllocated) {
            printf(":( - No bits available - maxObj:%zu maskElems:%zu (alloc)\n", maxObjCount, maskElemCount);
            // Check each mask element
            for(size_t i = 0; i < maskElemCount; i++) {
                allocMaskElem mask = (maxObjCount <= SLAB_ELEM_BIT_SIZE) ? allocMask : slab->data[i];
                printf(":( - Mask[%zu] = 0x%llx (popcount: %d)\n", 
                    i, (unsigned long long)mask, __builtin_popcountll(mask));
            }
            return nullptr;
        }

        if(!bitAllocated){
            // no bits - slab is full
            intr::atomic::store_relaxed(&reservationState, static_cast<uint32_t>(FULL));
            return nullptr;
        }

        // increment count only after bit is secured!!
        AllocState prev = intr::atomic::add_system(&allocState, (AllocState)1);
        AllocState newCount = (prev & COUNT_MASK) + 1;

        // sanity check
        if(newCount > maxObjCount){
            printf(":( - Count exceeded max after bit allocation! Count:%llu Max:%zu\n", 
                   (unsigned long long)newCount, maxObjCount);
            // decrement count and clear bit
            intr::atomic::add_system(&allocState, ((AllocState)0) - ((AllocState)1));

            allocMaskElem clearBit = ~(((allocMaskElem)1) << (globalIndex % SLAB_ELEM_BIT_SIZE));
            if(maxObjCount <= SLAB_ELEM_BIT_SIZE){
                intr::atomic::and_system(&allocMask, clearBit);
            } else {
                size_t maskInd = globalIndex / SLAB_ELEM_BIT_SIZE;
                intr::atomic::and_system(&slab->data[maskInd], clearBit);
            }
            return nullptr;
        }

        // update slab state
        slabFilled = (newCount == maxObjCount);
        if(slabFilled)
            intr::atomic::store_relaxed(&reservationState, static_cast<uint32_t>(FULL));

        void* result = indexToPtr(slab, objectSz, maskElemCount, globalIndex);

        return result;
    } // end of alloc

    __host__ __device__
    bool free(SLAB_TYPE* slab, void* objPtr, bool& slabEmptied){
        if(!objPtr)
            return false;
        
        char* objBytePtr = static_cast<char*>(objPtr);
        char* slabBytePtr = static_cast<char*>(static_cast<void*>(slab));
        size_t byteOffset = objBytePtr - slabBytePtr;

        AllocState currentState = intr::atomic::load_relaxed(&allocState);
        size_t objectSz = (currentState & SIZE_MASK) >> OFFSET;
        size_t currentCount = currentState & COUNT_MASK;
        size_t maxObjCount = slabObjCount(objectSz);

        // nothing allocated
        if(currentCount == 0)
            return false;

        // calc object index
        size_t maskCount = (maxObjCount + SLAB_ELEM_BIT_SIZE - 1) / SLAB_ELEM_BIT_SIZE;
        size_t maskSize = maskCount * sizeof(allocMaskElem);
        size_t firstObjOffset = maskSize;

        // invalid ptr
        if(byteOffset < firstObjOffset)
            return false;

        // misaligned
        size_t objOffset = byteOffset - firstObjOffset;
        if(objOffset % objectSz != 0)
            return false;

        // out of range
        size_t objIndex = objOffset / objectSz;
        if(objIndex >= maxObjCount)
            return false;

        // clear the bit first
        size_t maskIndex = objIndex / SLAB_ELEM_BIT_SIZE;
        size_t targetBit = objIndex % SLAB_ELEM_BIT_SIZE;
        allocMaskElem targetMask = ((allocMaskElem)1) << targetBit;
        allocMaskElem prevMask;

        if(maskCount <= 1) {
            prevMask = intr::atomic::and_system(&allocMask, ~targetMask);
        } else {
            prevMask = intr::atomic::and_system(&(slab->data[maskIndex]), ~targetMask);
        }

        // check if bit was actually set
        bool wasAllocated = ((prevMask & targetMask) != 0);
        if(!wasAllocated) 
            return false; 

        // decrement count after clearing bit
        AllocState prev = intr::atomic::add_system(&allocState, ((AllocState)0) - ((AllocState)1));
        AllocState newCount = (prev & COUNT_MASK) - 1;
        
        slabEmptied = (newCount == 0);
        if(!slabEmptied) {
            uint32_t state = intr::atomic::load_relaxed(&reservationState);
            if(state == FULL) 
                intr::atomic::store_relaxed(&reservationState, static_cast<uint32_t>(PARTIAL));
        }
        
        return true;
    } // end of free

}; // end of slabProxy



// node class
template <typename Type, typename IndexType, typename SizeType>
class Node{
    public:
        Type data;
        IndexType next;
        IndexType prev;

        __host__ __device__
        Node() :
        next(AdrInfo<IndexType>::null()), prev(AdrInfo<IndexType>::null()) {}
}; // end of Node
 

// direct arena class
template <typename Type, typename IndexType, typename SizeType>
class DirectArena {
    public:
        Type arena[SizeType::VALUE];

        __host__ __device__
        Type& at(IndexType index)
        { return arena[index];}

        __host__ __device__
        const Type& at(IndexType index) const
        { return arena[index];}

        static constexpr size_t size()  
        { return SizeType::VALUE; }
}; // end of DirectArena

// simple slab arena
template <typename ARENA_SIZE, 
          template<typename> typename SLAB_PROXY_TYPE = defaultSlabProxy,
          typename SLAB_TYPE = Slab<DEFAULT_SLAB_SIZE>,
          typename SLAB_ADDR_TYPE = defaultSlabAddr>
class SlabArena {
    public: 
        typedef SLAB_ADDR_TYPE slabAddrType;
        typedef SLAB_TYPE slabType;
        typedef SLAB_PROXY_TYPE<SLAB_TYPE> slabProxyType;
        typedef Node<slabProxyType, slabAddrType, Size<2>> proxyNodeType;

        static const size_t SLAB_COUNT = (ARENA_SIZE::VALUE + SLAB_TYPE::SIZE - 1) / SLAB_TYPE::SIZE;
        static const slabAddrType NULL_ADDR = static_cast<slabAddrType>(-1);

        typedef DirectArena<slabType, slabAddrType, Size<SLAB_COUNT>> BackingArenaType;
        typedef DirectArena<proxyNodeType, slabAddrType, Size<SLAB_COUNT>> ProxyArenaType;

    private:
        BackingArenaType slabs;
        ProxyArenaType proxies;
        
        slabAddrType freeListHead;
        size_t nextSearch;

    public:
        __host__ __device__
        SlabArena() {
            intr::atomic::store_relaxed(&freeListHead, NULL_ADDR);
            intr::atomic::store_relaxed(&nextSearch, static_cast<size_t>(0));
            
            for(size_t i = 0; i < SLAB_COUNT; i++){
                proxies.arena[i].data.allocState = 0;
                proxies.arena[i].data.allocMask = 0;
                intr::atomic::store_relaxed(&proxies.arena[i].data.reservationState, static_cast<uint32_t>(slabProxyType::FREE));

                // build list
                proxies.arena[i].next = (i == SLAB_COUNT - 1) ? NULL_ADDR : static_cast<slabAddrType>(i + 1);
                proxies.arena[i].prev = (i == 0) ? NULL_ADDR : static_cast<slabAddrType>(i - 1);
            }
        } 


        __host__ __device__
        slabAddrType popFreeList(){
            slabAddrType current = intr::atomic::load_acquire(&freeListHead);

            while(current != NULL_ADDR){
                slabAddrType next = proxies.arena[current].next;
                slabAddrType old = intr::atomic::CAS_system(&freeListHead, current, next);
                if(old == current){
                    proxies.arena[current].next = NULL_ADDR;
                    proxies.arena[current].prev = NULL_ADDR;
                    return current;
                }
                current = old;
            }

            return NULL_ADDR;
        } // end of pop

        __host__ __device__
        void pushFreeList(slabAddrType slabInd){
            if(slabInd >= SLAB_COUNT)
                return;

            slabAddrType oldHead = intr::atomic::load_acquire(&freeListHead);

            do {
                proxies.arena[slabInd].next = oldHead;
                proxies.arena[slabInd].prev = NULL_ADDR;

                // if oldHead, update prev
                if(oldHead != NULL_ADDR)
                    proxies.arena[oldHead].prev = slabInd;

                slabAddrType old = intr::atomic::CAS_system(&freeListHead, oldHead, slabInd);
                if(old == oldHead) break;
                    oldHead = old;
            } while(true);
        } // end of push

        __host__ __device__
        void removeFromFreeList(slabAddrType slabInd){
            if(slabInd >= SLAB_COUNT)
                return;

            slabAddrType prev = proxies.arena[slabInd].prev;
            slabAddrType next = proxies.arena[slabInd].next;

            // updates prevs next
            if(prev != NULL_ADDR){
                proxies.arena[prev].next = next;
            } else {
                // it was the head!
                slabAddrType expected = slabInd;
                intr::atomic::CAS_system(&freeListHead, expected, next);
            }
            // update nexts prev
            if(next != NULL_ADDR)
                proxies.arena[next].prev = prev;
            
            
            proxies.arena[slabInd].next = NULL_ADDR;
            proxies.arena[slabInd].prev = NULL_ADDR;
        } // end

        __host__ __device__
        bool isValidPtr(void* ptr){
            if(!ptr)
                return false;
            
            slabAddrType slabIndex = slabIndexFor(ptr);
            if(slabIndex >= SLAB_COUNT)
                return false;

            if(proxies.arena[slabIndex].data.getSize() == 0)
                return false;

            return true;
        } // end of isValidPtr

        __host__ __device__
        slabAddrType slabIndexFor(void* ptr){
            char* bytePtr = static_cast<char*>(ptr);
            char* baseBytePtr = static_cast<char*>(static_cast<void*>(slabs.arena));
            
            size_t ptrOffset = bytePtr - baseBytePtr;
            size_t slabIndex = ptrOffset / SLAB_TYPE::SIZE;

            return static_cast<slabAddrType>(slabIndex);
        } // end of slabIndexFor

        __host__ __device__
        slabType& slabAt(slabAddrType slabIndex)
        { return slabs.arena[slabIndex];}

        __host__ __device__
        proxyNodeType& proxyAt(slabAddrType slabIndex)
        { return proxies.arena[slabIndex];}

        __host__ __device__
        slabType& slabFor(void* ptr){
            slabAddrType slabIndex = slabIndexFor(ptr);
            return slabs.arena[slabIndex];
        }

        __host__ __device__
        proxyNodeType& proxyFor(void* ptr){
            slabAddrType slabIndex = slabIndexFor(ptr);
            return proxies.arena[slabIndex];
        }

        // round-robin :) + res
        __host__ __device__
        slabAddrType alloc(size_t objectSize = 0){
            if(objectSize == 0)
                objectSize = 1;
            
            // look for slabs of size sz
            for(size_t i = 0; i < SLAB_COUNT; i++){
                slabProxyType& proxy = proxies.arena[i].data;
                if(proxy.getSize() == objectSize && proxy.isAvailable() && !proxy.isFull()){
                    uint32_t state = intr::atomic::load_relaxed(&proxy.reservationState);
                    if(state == slabProxyType::PARTIAL && !proxy.isFull()) 
                        return static_cast<slabAddrType>(i);
                }
            }

            // try free slab from the free list
            slabAddrType freeSlab = popFreeList();
            if (freeSlab != NULL_ADDR) {
                slabProxyType& proxy = proxies.arena[freeSlab].data;
                if (proxy.tryClaim(objectSize)) {
                    return freeSlab;
                } else {
                    // fail? back to list
                    pushFreeList(freeSlab);
                }
            }

            // fallback - round-robin search 
            size_t start = intr::atomic::add_system(&nextSearch, static_cast<size_t>(1)) % SLAB_COUNT;
            for(size_t offset = 0; offset < SLAB_COUNT; offset++){
                size_t i = (start + offset) % SLAB_COUNT;
                slabProxyType& proxy = proxies.arena[i].data;

                if(intr::atomic::load_acquire(&proxy.reservationState) == slabProxyType::FREE
                && proxy.getSize() == 0){
                    if(proxy.tryClaim(objectSize)) {
                        // remove from free list - reserved
                        removeFromFreeList(static_cast<slabAddrType>(i));
                        return static_cast<slabAddrType>(i);
                    }
                }
            }
            
            return NULL_ADDR;
        } // end of alloc

        __host__ __device__
        bool free(slabAddrType slabAddr){
            if(slabAddr >= SLAB_COUNT)
                return false;
            bool success = proxies.arena[slabAddr].data.clearAllocState();
            if (success)
                pushFreeList(slabAddr);
            
        return success;
        } // end of free

        __host__ __device__
        void getStats(size_t& totalSlabs, size_t& usedSlabs, size_t& emptyReusables, size_t& freeSlabs){
            totalSlabs = SLAB_COUNT;
            usedSlabs = 0;
            emptyReusables = 0;
            freeSlabs = 0;

            slabAddrType current = intr::atomic::load_acquire(&freeListHead);
            while (current != NULL_ADDR) {
                freeSlabs++;
                current = proxies.arena[current].next;
            }

            for(size_t i = 0; i < SLAB_COUNT; i++){
                slabProxyType& proxy = proxies.arena[i].data;
                if(proxy.getSize() > 0){
                    if(proxy.isEmpty()){
                        emptyReusables++;
                    } else {
                        usedSlabs++;
                    }
                }
            }
        } // end of getStats

        __host__ __device__
        bool validateFreeList() {
            size_t count = 0;
            slabAddrType current = intr::atomic::load_acquire(&freeListHead);
            slabAddrType prev = NULL_ADDR;
            
            while (current != NULL_ADDR && count < SLAB_COUNT) {
                // check that prev
                if (proxies.arena[current].prev != prev) {
                    return false;
                }
                
                // check slab is free
                if (proxies.arena[current].data.getSize() != 0) {
                    return false;
                }
                
                prev = current;
                current = proxies.arena[current].next;
                count++;
            }
            
            return count < SLAB_COUNT; 
        }
}; // end of SlabArena

// test allocator
template <typename SLAB_ALLOCATOR_TYPE>
class SimpleAllocator {
    public:
        typedef SLAB_ALLOCATOR_TYPE SlabAllocatorType;
        typedef typename SlabAllocatorType::slabType SlabType;
        typedef typename SlabAllocatorType::slabAddrType SlabAddrType;
        typedef typename SlabAllocatorType::slabProxyType SlabProxyType;

    private:
        SlabAllocatorType& slabAllocator;
        size_t objectSize;

    public:
        __host__ __device__
        SimpleAllocator(SlabAllocatorType& slabAlloc, size_t objSize) :
            slabAllocator(slabAlloc), objectSize(objSize == 0 ? 1 : objSize) {}


        __host__ __device__
        bool isValidPtr(void* ptr) {
            return slabAllocator.isValidPtr(ptr);
        }

        __host__ __device__
        size_t getAllocatedCount() {
            size_t totalCount = 0;
            
            // Iterate through all slabs in the allocator
            for(size_t i = 0; i < SlabAllocatorType::SLAB_COUNT; i++) {
                SlabProxyType& slabProxy = slabAllocator.proxyAt(static_cast<SlabAddrType>(i)).data;
                
                // Only count slabs that are allocated for this object size
                if(slabProxy.getSize() == objectSize) {
                    totalCount += (slabProxy.allocState & SlabProxyType::COUNT_MASK);
                }
            }
            
            return totalCount;
        } // end of getCount

        __host__ __device__
        void getReuseStats(size_t& activeSlabs, size_t& reusableSlabs, size_t& totalObjects){
            activeSlabs = 0;
            reusableSlabs = 0;
            totalObjects = 0;

            for(size_t i = 0; i < SlabAllocatorType::SLAB_COUNT; i++){
                SlabProxyType& slabProxy = slabAllocator.proxyAt(static_cast<SlabAddrType>(i)).data;

                if(slabProxy.getSize() == objectSize){
                    size_t count = slabProxy.getCount();
                    totalObjects += count;

                    if(count > 0){
                        activeSlabs++;
                    } else {
                        reusableSlabs++;
                    } // else
                } // outer if
            } // for
        } // end of getReuse
        
        __host__ __device__
        void* alloc(){
            SlabAddrType slabAddr = slabAllocator.alloc(objectSize);
            if(slabAddr == SlabAllocatorType::NULL_ADDR)
                return nullptr;

            SlabType& slab = slabAllocator.slabAt(slabAddr);
            SlabProxyType& slabProxy = slabAllocator.proxyAt(slabAddr).data;

            bool slabFilled = false;
            void* result = slabProxy.alloc(&slab, slabFilled);

            return result;
        } // end of alloc

        __host__ __device__
        bool free(void* ptr){
            if(!ptr)
                return false;

            SlabAddrType slabAddr = slabAllocator.slabIndexFor(ptr);
            SlabType& slab = slabAllocator.slabAt(slabAddr);
            SlabProxyType& slabProxy = slabAllocator.proxyAt(slabAddr).data;

            bool slabEmptied = false;
            if(!slabProxy.free(&slab, ptr, slabEmptied))
                return false;

            return true;
        } // end of free
}; // end of simpleAlloc

// smaller slabs!!
typedef SlabArena<Size<1024*1024>, defaultSlabProxy, Slab<4096>> TestSlabArena; // 4KB slabs
typedef SimpleAllocator<TestSlabArena> TestAllocator;
