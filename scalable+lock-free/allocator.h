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

    allocMaskElem allocMask;
    AllocState allocState;

    __host__ __device__
    size_t getSize()
    { return (allocState & SIZE_MASK) >> OFFSET;}

    __host__ __device__
    size_t getCount()
    { return (allocState & COUNT_MASK);}

    __host__ __device__
    size_t isEmpty()
    { return getCount() == 0; }

    __host__ __device__
    bool isFull(){
        size_t objSize = getSize();
        if(objSize == 0)
            return false;

        return (getCount() >= slabObjCount(objSize));
    } // end of isFull

    // IS THIS REALLY NECESSARY
    __host__ __device__
    bool bindSize(unsigned int newSize) {
        allocMaskElem longSize = newSize;            
        allocMaskElem prev = intr::atomic::CAS_system(&allocState, 0ull, longSize << OFFSET);
        return (prev == 0);
    }

    __host__ __device__
    bool clearAllocState(){
        if(allocState == 0)
            return false;
        
        allocMaskElem prev = intr::atomic::exch_system(&allocState, 0ull);
        return ((prev & SIZE_MASK) != 0) && ((prev & COUNT_MASK) == 0);
    }

    __host__ __device__
    size_t slabObjCountNoMask(size_t objectSize)
    { return SLAB_SIZE / objectSize;}

    __host__ __device__
    size_t slabObjCountWithMask(size_t objectSize){
        size_t objectBitSize = objectSize * 8;
        size_t bitCostPerObj = (objectBitSize + 1);
        size_t totalBits = SLAB_ELEM_COUNT + SLAB_ELEM_BIT_SIZE;
        size_t idealTOBC = (totalBits * objectBitSize) / bitCostPerObj;     // total object bit count
        size_t maxExcluTOBC = (idealTOBC / SLAB_ELEM_BIT_SIZE) * SLAB_ELEM_BIT_SIZE;

        size_t objectCount = maxExcluTOBC / objectBitSize;
        return objectCount;
    }

    __host__ __device__
    size_t slabObjCount(size_t objectSize){
        if(objectSize <= 0)
            return (size_t)nullptr;           // min allocation size? NULLPTR

        size_t result = slabObjCountNoMask(objectSize);
        if(result > 64)
            result = slabObjCountWithMask(objectSize);

        return result;
    }

    __host__ __device__
    void clearSlab(SLAB_TYPE* slab, size_t maskElemCount){
        for(size_t i = 0; i < maskElemCount; i++)
            slab->data[i] = 0;
    }

    __host__ __device__
    bool claim(SLAB_TYPE* slab, size_t objectSize){
        if(objectSize <= 0)
            return false;           // nullptr?

        AllocState startingState = ((AllocState)objectSize) << OFFSET;
        AllocState currentState = allocState;

        // new slab
        if(currentState == 0) {
            if(intr::atomic::CAS_system(&allocState, (AllocState)0, startingState) != 0)
                return false;

            size_t objectCount = slabObjCount(objectSize);
            if(objectCount == 0)
                return false;

            allocMask = 0;          
            if(objectCount > SLAB_ELEM_BIT_SIZE){
                size_t maskElemCount = (objectCount + SLAB_ELEM_BIT_SIZE - 1) / SLAB_ELEM_BIT_SIZE;
                clearSlab(slab, maskElemCount);
            }

            return true;
        }

        // empty slab with same objSz
        size_t currentSize = (currentState & SIZE_MASK) >> OFFSET;
        size_t currentCount = currentState & COUNT_MASK;

        // check slab is ready for objSz+free
        if(currentSize == objectSize && currentCount == 0)
            return true;        
        
        return false;
    } // end of claim

    __host__ __device__
    bool attemptMaskAlloc(allocMaskElem& elem, size_t& result){
        allocMaskElem maskCopy = allocMask;

        while(intr::bitwise::population_count(maskCopy) < SLAB_ELEM_BIT_SIZE){
            size_t target = intr::bitwise::first_set(~maskCopy);
            allocMaskElem targetMask = 1 << target;
            maskCopy = intr::atomic::or_system(&allocMask, targetMask);

            if((maskCopy & targetMask) == 0){
                result = target;
                return true;
            }
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
        allocMaskElem prev = intr::atomic::add_system(&allocState, (allocMaskElem)1);

        allocMaskElem prevCount = prev & COUNT_MASK;
        allocMaskElem objectSize = (prev & SIZE_MASK) >> OFFSET;

        allocMaskElem maxObjCount = slabObjCount(objectSize);
        if(prevCount >= maxObjCount){
            intr::atomic::add_system(&allocState, ((AllocState)0) - ((AllocState)1));
            return nullptr;
        }        

        slabFilled = (prevCount == (maxObjCount - 1));

        size_t maskElemCount = (maxObjCount + SLAB_ELEM_BIT_SIZE - 1) / SLAB_ELEM_BIT_SIZE;
        if(maxObjCount <= SLAB_ELEM_BIT_SIZE){
            size_t index;
            if(attemptMaskAlloc(allocMask, index)){
                return indexToPtr(slab, objectSize, maskElemCount, index);
            } else {
                return nullptr;
            }
        } else {
            for(size_t i = 0; i < 2048; i++){
                for(size_t j = 0; j < SLAB_ELEM_COUNT; j++){
                    size_t index;
                    if(attemptMaskAlloc(slab->data[j], index)){
                        size_t fullIndex = SLAB_ELEM_BIT_SIZE * j + index;
                        return indexToPtr(slab, objectSize, maskElemCount, fullIndex);
                    }
                }
            }
            return nullptr;
        }
    } // end of alloc

    __host__ __device__
    bool free(SLAB_TYPE* slab, void* objPtr, bool& slabEmptied){
        char* objBytePtr = static_cast<char*>(objPtr);
        char* slabBytePtr = static_cast<char*>(static_cast<void*>(slab));
        size_t byteOffset = objBytePtr - slabBytePtr;

        allocMaskElem prev = intr::atomic::add_system(&allocState, ((AllocState)1) - ((AllocState)1));
        allocMaskElem prevCount = prev & COUNT_MASK;

        if(prevCount == 0){
            intr::atomic::add_system(&allocState, 1ull);
            return false;
        }

        slabEmptied = (prevCount == 1);

        size_t objectSize = (prev & SIZE_MASK) >> OFFSET;
        size_t maxObjCount = slabObjCount(objectSize);

        size_t maskCount = (maxObjCount + SLAB_ELEM_BIT_SIZE - 1) / SLAB_ELEM_BIT_SIZE;
        size_t maskSize = maskCount * sizeof(allocMaskElem);
        size_t firstObjOffset = maskSize;

        size_t objOffset = byteOffset - firstObjOffset;
        if(objOffset % objectSize != 0)
            return false;
        
        size_t objIndex = objOffset / objectSize;
        if(objIndex >= maxObjCount)
            return false;

        size_t maskIndex = objIndex / SLAB_ELEM_BIT_SIZE;
        size_t targetBit = objIndex % SLAB_ELEM_BIT_SIZE;
        allocMaskElem targetMask = ((allocMaskElem)1) << ((allocMaskElem)targetBit);
        allocMaskElem prevMask = 0;

        if(maskCount <= 1){
            prevMask = intr::atomic::or_system(&allocMask, ~targetMask);
        } else {
            prevMask = intr::atomic::or_system(&(slab->data[maskIndex]), ~targetMask);
        }

        return ((prevMask & targetMask) != 0);
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

        typedef DirectArena<slabType, slabAddrType, Size<SLAB_COUNT>> BackingArenaType;
        typedef DirectArena<proxyNodeType, slabAddrType, Size<SLAB_COUNT>> ProxyArenaType;

    private:
        BackingArenaType slabs;
        ProxyArenaType proxies;

    public:
        __host__ __device__
        SlabArena() {
            for(size_t i = 0; i < SLAB_COUNT; i++){
                proxies.arena[i].data.allocState = 0;
                proxies.arena[i].data.allocMask = 0;
            }
        } 

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

        // first fit :) + reuse
        __host__ __device__
        slabAddrType alloc(size_t objectSize = 0){
            slabAddrType bestFit = AdrInfo<slabAddrType>::null();

            // check for empty slabs of same size
            if(objectSize > 0){
                for(size_t i = 0; i < SLAB_COUNT; i++){
                    slabProxyType& proxy = proxies.arena[i].data;
                    if(proxy.getSize() == objectSize && proxy.isEmpty())
                        return static_cast<slabAddrType>(i);
                }
            }

            // check for free slabs
            for(size_t i = 0; i < SLAB_COUNT; i++){
                if(proxies.arena[i].data.getSize() == 0){
                    if(bestFit == AdrInfo<slabAddrType>::null())
                        bestFit = static_cast<slabAddrType>(i);
                }
            }
            
            return bestFit;
        } // end of alloc

        // IS THIS NEEDED
        __host__ __device__
        slabAddrType allocForSize(size_t objSize)
        { return alloc(objSize);}

        __host__ __device__
        bool free(slabAddrType slabAddr){
            if(slabAddr >= SLAB_COUNT)
                return false;
            return proxies.arena[slabAddr].data.clearAllocState();
        } // end of free

        __host__ __device__
        void getStats(size_t& totalSlabs, size_t& usedSlabs, size_t& emptyResusables){
            totalSlabs = SLAB_COUNT;
            usedSlabs = 0;
            emptyResusables = 0;

            for(size_t i = 0; i < SLAB_COUNT; i++){
                slabProxyType& proxy = proxies.arena[i].data;
                if(proxy.getSize() > 0){
                    if(proxy.isEmpty()){
                        emptyResusables++;
                    } else {
                        usedSlabs++;
                    }
                }
            }
        } // end of getStats
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
            slabAllocator(slabAlloc), objectSize(objSize) {}

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
                    }
                }
                    
            }
        } // end of getReuse
        
        __host__ __device__
        void* alloc(){
            SlabAddrType slabAddr = slabAllocator.allocForSize(objectSize);
            if(slabAddr == AdrInfo<SlabAddrType>::null())
                return nullptr;

            SlabType& slab = slabAllocator.slabAt(slabAddr);
            SlabProxyType& slabProxy = slabAllocator.proxyAt(slabAddr).data;

            if(!slabProxy.claim(&slab, objectSize))
                return nullptr;

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

// type definitions for easy use
typedef SlabArena<Size<1024*1024>> TestSlabArena; // 1MB arena
typedef SimpleAllocator<TestSlabArena> TestAllocator;
