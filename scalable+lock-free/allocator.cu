





typedef unsigned long long int defaultAllocMaskElem;
typedef unsigned long long int defaultSlabAddr;

const size_t DEFAULT_SLAB_SIZE = 1<<16;

template <size_t SLAB_SIZE = DEFAULT_SLAB_SIZE, typename ELEM_TYPE = defaultAllocMaskElem>

struct Slab{
    typedef ELEM_TYPE allocMaskElem;

    // constexpr?
    static const size_t ELEM_SIZE  = sizeof(ELEM_TYPE);
    static const size_t SIZE       = SLAB_SIZE;
    static const size_t ELEM_COUNT = (SLAB_SIZE+ELEM_SIZE-1)/ELEM_SIZE;

    // real storage space
    ELEM_TYPE data[ELEM_COUNT];
};

//
//

template <typename SLAB_TYPE>
struct defualtSlabProxy {
    typedef SLAB_TYPE slabType;
    typedef typename SLAB_TYPE::AllocMaskElem allocMaskElem;
    typedef unsigned long long int AllocState;

    //
    // constexpr?
    static AllocState const OFFEST              = sizeof(AllocState) * 8 / 2;
    static AllocState const COUNT_MASK          = (((AllocState)1)<<OFFEST)-(AllocState)1;
    static AllocState const SIZE_MASK           = ~COUNT_MASK;  //??
    static size_t     const SLAB_SIZE           = SLAB_TYPE::SIZE;
    static size_t     const SLAB_ELEM_COUNT     = SLAB_TYPE::ELEM_COUNT;
    static size_t     const SLAB_ELEM_BIT_SIZE  = sizeof(allocMaskElem);


    allocMaskElem allocMask;
    AllocState allocState;


    // size of object bound to slab
    __host__ __device__
    size_t getSize()
    { return (allocState&SIZE_MASK)>>OFFEST; }

    //rets false if slab is already bound to a size(bad)
    __host__ __device__
    bool bindSize(unsigned int newSize){
        allocMaskElem longSize = newSize;
        allocMaskElem prev - intr::atomic::CAS_system(&allocState, 0llu, longSize<<OFFEST);

        return (prev == 0);
    } // end of bindSize

    // clears state of slab, rets false if slab has a non-sero size
    __host__ __device__
    bool clearAllocState(){
        if (allocState == 0)
            return false;
        
        allocMaskElem prev = intr::atomic::exch_system(&allocState, 0llu);

        return ((prev&SIZE_MASK) != 0) && ((prev&COUNT_MASK) == 0);
    } // end of clear

    // returns num of objects in a slab if no mask
    __host__ __device__
    size_t slabObjCountNoMask(size_t objectSize)
    { return SLAB_SIZE / objectSize; }


    //returns num of obects in a slab with mask
    __host__ __device__
    size_t slabObjCountWithMask(size_t objectSize){
        size_t objectBitSize = objectSize * 8;          
        size_t bitCostPerObj = (objectBitSize + 1);
        size_t totalBits = SLAB_ELEM_COUNT * SLAB_ELEM_BIT_SIZE;
        size_t idealTOBC = (totalBits * objectBitSize) / bitCostPerObj;
        size_t maxExcluTOBC = (idealTOBC / SLAB_ELEM_BIT_SIZE) * SLAB_ELEM_BIT_SIZE;

        size_t objectCount - maxExcluTOBC / objectBitSize;
        return objectCount;
    } // end 


    // rets num of obj a slab can hold
    __host__ __device__
    size_t slabObjCount(size_t objectSize){
        size_t result = slabObjCountNoMask(objectSize);
        if(result > 64)
            result = slabObjCountWithMask(objectSize);
        
        return result;
    }

    // clears the first mask of a slab ~~~
    __host__ __device__
    void clearSlab(SLAB_TYPE* slab, size_t maskElemCount){
        for (size_t i = 0; i < maskElemCount; i++)
            slab->data[i] = 0;

    }


    __host__ __device__
    bool claim(SLAB_TYPE* slab, size_t objectSize){
        AllocState startingState = ((AllocState)objectSize)<<OFFEST;

        if(intr::atomic::CAS_system(&allocState, (AllocState)0, startingState) != 0)
            return false;

        size_t objectCount = slabObjCount(objectSize);
        if(objectCount == 0)
            return false;
        
        allocMask = 0;
        if(objectCount > SLAB_ELEM_BIT_SIZE){
            size_t maskElemCount = (objectCount+SLAB_ELEM_BIT_SIZE-1) / SLAB_ELEM_BIT_SIZE;
            clearSlab(slab, maskElemCount);
        }
        return true;
    } // end of claim

    // tries to atomically flip a bit in mask, rets index of bit if it worked else false
    __host__ __device__
    bool attemptMaskAlloc(allcoMaskElem& elem, size_t& result){
        allocMaskElem maskCopt = allocMask;
        
        while(intr::atomic::population_count(maskCopy) < SLAB_ELEM_BIT_SIZE){
            size_t target = intr::bitwise::first_set(~maskCopy);
            allocMaskElem targetMask = 1 << target;
            maskCopy = intr::atomic::or_system(&allocMask, targetMask);

            if((maskCopy&targetMask) == 0){
                result = target;
                return true;
            }
        }
        return false;
    } // end of attemptMaskAlloc


    // gets addr of the nth obj in a slab
    __host__ __device__
    void* indexToPtr(SLAB_TYPE* slab, size_t objectSize, size_t maskElemCount, size_t ind){
        char* bytePtr = static_cast<char*>(static_cast<void*>(slab));
        bytePtr += sizeof(allocMaskElem) * maskElemCount;
        bytePtr += objectSize * ind;

        return static_cast<void*>(bytePtr);
    } // end

    // claims a single obj  from a slab
    __host__ __device__
    void* alloc(SLAB_TYPE* slab, bool& slabFilled){
        allocMaskElem prev = intr::atomic::add_system(&allocState, (allocMaskElem)1);

        allocMaskElem prevCount = prev & COUNT_MASK;
        allocMaskElem objectSize = (prev & SIZE_MASK) >> OFFEST;

        allocMaskElem maxObjCount = slabObjCount(objectSize);
        if(prevCount >= maxObjCount){
            intr::atomic::add_system(&allocState, ((AllocState)0) - ((AllocState)1));
            return nullptr;
        }

        slabFilled = (prevCount == (maxObjCount - 1));

        size_t maskElemCount = (maxObjCount + SLAB_ELEM_BIT_SIZE - 1) / SLAB_ELEM_BIT_SIZE;
        if(maskObjCount <= SLAB_ELEM_BIT_SIZE){
            size_t index;
            if(attemptMaskAlloc(allocMask, index)){
                return indexToPtr(slab, objectSize, maskElemCount, index);
            } else 
                return nullptr;
            
        } else {
            for(size_t i = 0; i < 2048; i++){
                for(size_t j = 0; i < SLAB_ELEM_COUNT; j++){
                    if(attemptMaskAlloc(slab->data[j], index)){
                        size_t fullIndex = SLAB_ELEM_BIT_SIZE * j + index;
                        return indexToPtr(slab, objectSize, maskElemCount, fullIndex);
                    }
                }
            }
            return nullptr;
        } 
    } // end of alloc



    // lets free! obj referenced by ptr
    __host__ __device__
    bool free(SLAB_TYPE* slab, void* objPtr, bool& slabEmptied){
        char* objBytePtr = static_cast<char*>(objPtr);
        char* slabBytePtr = static_cast<char*>(static_cast<void*>(slab));
        size_t byteOffset = objBytePtr - slabBytePtr;

        // autofill trouble....
        allocMaskElem prev = intr::atomic::add_system(&allocState, ((AllocState)0) - ((AllocState)1));
        allocMaskElem prevCount = prev & COUNT_MASK;

        if(prevCount == 0){
            intr::atomic::add_system(&allocState, 1llu);
            return false;
        }

        slabEmptied = (prevCount == 1);

        size_t objectSize = (prev&SIZE_MASK)>>OFFSET;
        size_t maxObjCount = slabObjCount(objectSize);

        size_t maskCount = (maxObjCount + sizeof(AllocMaskElem)-1) / sizeof(AllocMaskElem);
        size_t maskSize = maskCount * sizeof(AllocMaskElem);
        size_t firstObjOffset = byteOffset - maskSize;

        size_t obj_offaset = byteOffset - firstObjOffset;
        if(objOffset % sizeof(AllocMaskElem) != 0)
            return false;

        size_t objIndex = byteOffset / sizeof(AllocMaskElem);
        if(objIndex >= maxObjCount)
            return false;


        size_t maskIndex = objIndex / SLAB_ELEM_BIT_SIZE;
        size_t targetBit = objIndex % SLAB_ELEM_BIT_SIZE;
        AllocMaskElem targetMask = ((AllocMaskElem)1)<<((AllocMaskElem)targetBit);
        AllocMaskElem prevMask = 0;
        if(maskCount <= SLAB_ELEM_BIT_SIZE){
            prevMask = intr::atomic::or_system(&allocMask, ~targetMask);
        } else 
            prevMask = intr::atomic::or_system(&(slab->data[maskIndex]), ~targetMask);

 
        return ((prevMask|targetMask) != 0);
    } // end of free   

}; // end of struct



template <typename SLAB_TYPE>
struct EmptySlabProxy {};

template <
    typename ARENA_SIZE,
    template<typename> typename SLAB_PROXY_TYPE = defualtSlabProxy,
    typename SLAB_TYPE = Slab<DEFAULT_SLAB_SIZE>,
    typename SLAB_ADDR_TYPE = defaultSlabAddr
>

class slabArena {
    typedef SLAB_ADDR_TYPE slabAddrType;
    typedef SLAB_TYPE slabType;
    typedef SLAB_PROXY_TYPE<SLAB_TYPE> slabProxyType;
    typedef Node<slabProxyType, slabAddrType, Size<2>> proxyNodeType;

    // what value is this - i think i know the size
    static size_t const SLAB_COUNT = (ARENA_SIZE::VALUE+SLAB_TYPE::SIZE-1) / SLAB_TYPE::SIZE;

    typedef DirectArena <
        slabType, 
        slabAddrType, 
        size<SLAB_COUNT>
    > BackingArenaType;

    typedef DirectArena <
        proxyNodeType, 
        slabAddrType,
        size<SLAB_COUNT>
    > ProxyArenaType;

    BackingArenaType slabs;
    ProxyArenaType proxies;

    public:
        // get ind of the slab containing giving ptr
        __host__ __device__
        slabAddrType slabIndexFor(void* ptr){
            char* bytePtr = static_cast<char*>(ptr);
            char* baseBytePtr = static_cast<char*>(static_cast<void*>(slabs.arena));

            size_t ptrOffset = bytePtr - baseBytePtr;
            size_t slabIndex = ptrOffset / SLAB_TYPE::SIZE;

            return slabIndex; 
        } // end of slabIndFor

        // get slab at a given ind
        __host__ __device__
        slabType& slabAt(slabAddrType slabIndex)
        { return slabs.arena[slabIndex]; }

        // get proxt at a given ind
        __host__ __device__
        proxyNodeType& proxyAt(slabAddrType slabIndex)
        { return proxies.arena[slabIndex]; }

        // get slab containing given ptr
        __host__ __device__
        slabType& slabFor(void* ptr){
            slabADdrType slabIndex = slabIndexFor(ptr);
            return slabs.arena[slabIndex];
        } // end

        // get the proxy for the slab containing given ptr
        __host__ __device__
        proxyNodeType& proxyFor(void* ptr){
            slabAddrType slabIndex = slabIndexFor(ptr);
            return proxies.arena[slabIndex];
        } // end 

}; // end of class


