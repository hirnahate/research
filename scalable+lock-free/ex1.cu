// lets try simple slab allocation



// superblock descriptor structure
typedef header : 
    unsigned avail: 10 , count: 10, state: 2, tag: 42;
    // state codes active = 0 full = 1 partial = 2 empty = 3

// processor heap structure
typedef active : unsigned ptr: 58, credits: 6;
typedef procheap : 
    active Active;          // init=NULL
    descriptor* Partial;    // init=NULL
    sizeclass* sc;          // ptr to parent sizeclass

typedef descriptor :
    header Header;
    descriptor* Next;
    procheap* heap;     // ptr to owner procheap
    void* sb;           // ptr to superblock

    unsigned sz;        // block size
    unsigned maxCount;  // superblock size


// size class structure
typedef sizeclass :
    descList Partial;   // init empty
    unsigned sz;        // block size
    unsigned sbSize;    // superblock size

// check and swap
 bool CAS(addr, oldVal, newVal){ // atomically do 
    if(*addr == oldVal){
        *addr = newVal;
        return true;
    }
    return false;
} // end of CAS


void* malloc(sz){
    // use sz and thread id to find heap
    heap = find.heap(sz);
    if(!heap)   // too large
        // allocate block from OS and return addr
    while(true){
        addr = mallocFromActive(heap);
        if (addr) return addr;
        addr = mallocFromPartial(heap);
        if(addr) return addr;
        addr = mallocFromNewSB(heap);
        if(addr) return addr;
    }
} // end of malloc

void* mallocFromActive(heap){
    do { // 1. reverse block
        newA = oldA = heap->Active;
        if(!oldA) return NULL;
        if (oldA.credits == 0)
            newA = NULL;
        else
            newA.credits--;
    } while( CAS(&heap, oldA, newA);)   // until
    // 2. pop block
    desc = mask.credits(oldA);
    do {
        // state may be A/P/F
        newAnchor = oldAnchor = desc->Anchor;
        addr = desc->sb+oldAnchor.avail*desc->sz;
        next = *(unsigned*)addr;
        newAnchor.avail = next;
        newAnchor.tag++;
        if(oldA.credits == 0){
            // state must be active
            if(oldA.count == 0)
                newA.state = FULL;
            else{
                moreCredits = 
                    min(oldAnchor.count, MAXCREDITS);
                newAnchor.count -= moreCredits;
            }
        }
    } while(CAS(&desc->Anchor, oldAnchor, newAnchor);)  // until

    if(oldA.credits == 0 && oldAnchor.count > 0)
        updateActive(heap, desc, moreCredits);
    *addr = desc;

    return addr+EIGHTBYTES;
} // end of active malloc

// helper (?)
updateActive(heap, desc, moreCredits){
    newA = desc;
    newA.credits = moreCredits - 1;
    if CAS(&heap->Actice, NULL, newA) return;
    // someone installed another active SB
    // return credits to sb and make is partial
    do {
        newAnchor = oldAnchor = desc->Anchor;
        newAnchor.count += moreCredits;
        newAnchor.state = PARTIAL;
    } while (CAS(&desc->Anchor, oldAnchor, newAnchor);) // until
    heapPutPartial(desc);
} // end of updateActive

