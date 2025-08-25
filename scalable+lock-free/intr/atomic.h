#ifndef ATOMIC
#define ATOMIC

namespace atomic {
/*
    // Add these specializations after your existing template functions
template<>
__host__ __device__ inline size_t add_system<size_t>(size_t* adr, size_t val) {
    #ifdef __CUDA_ARCH__
        return atomicAdd((unsigned long long*)adr, (unsigned long long)val);
    #else
        return __sync_fetch_and_add(adr, val);
    #endif
}

template<>
__host__ __device__ inline size_t sub_system<size_t>(size_t* adr, size_t val) {
    #ifdef __CUDA_ARCH__
        return atomicSub((unsigned long long*)adr, (unsigned long long)val);
    #else
        return __sync_fetch_and_sub(adr, val);
    #endif
}

template<>
__host__ __device__ inline size_t CAS_system<size_t>(size_t* adr, size_t comp, size_t val) {
    #ifdef __CUDA_ARCH__
        return atomicCAS((unsigned long long*)adr, (unsigned long long)comp, (unsigned long long)val);
    #else
        size_t expected = comp;
        __atomic_compare_exchange(adr, &expected, &val, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
        return expected;
    #endif
}

    template<typename T>
    __host__ __device__ T and_system(T* adr,T val) {
        #ifdef __CUDA_ARCH__
            return atomicAnd(adr,val);
        #else
            return __sync_fetch_and_and(adr,val);
        #endif
    }

    template<typename T>
    __host__ __device__ T or_system(T* adr,T val) {
        #ifdef __CUDA_ARCH__
            return atomicOr(adr,val);
        #else
            return __sync_fetch_and_or(adr,val);
        #endif
    }

    template<typename T>
    __host__ __device__ T xor_system(T* adr,T val) {
        #ifdef __CUDA_ARCH__
            return atomicXor(adr,val);
        #else
            return __sync_fetch_and_xor(adr,val);
        #endif
    }

    template<typename T>
    __host__ __device__ T min_system(T* adr,T val) {
        #ifdef __CUDA_ARCH__
            return atomicMin(adr,val);
        #else
            return __sync_fetch_and_min(adr,val);
        #endif
    }

    template<typename T>
    __host__ __device__ T exch_system(T* adr,T val) {
        #ifdef __CUDA_ARCH__
            return atomicExch(adr,val);
        #else
            T result;
            __atomic_exchange(adr,&val,&result,__ATOMIC_ACQ_REL);
            return result;
        #endif
    }
*/
    template<typename T>
    __host__ __device__ T add_system(T* adr,T val) {
        #ifdef __CUDA_ARCH__
            return atomicAdd(adr,val);
        #else
            return __sync_fetch_and_add(adr,val);
        #endif
    }

    template<typename T>
    __host__ __device__ T sub_system(T* adr,T val) {
        #ifdef __CUDA_ARCH__
            return atomicSub(adr,val);
        #else
            return __sync_fetch_and_sub(adr,val);
        #endif
    }

    template<typename T>
    __host__ __device__ T and_system(T* adr,T val) {
        #ifdef __CUDA_ARCH__
            return atomicAnd(adr,val);
        #else
            return __sync_fetch_and_and(adr,val);
        #endif
    }

    template<typename T>
    __host__ __device__ T or_system(T* adr,T val) {
        #ifdef __CUDA_ARCH__
            return atomicOr(adr,val);
        #else
            return __sync_fetch_and_or(adr,val);
        #endif
    }

    template<typename T>
    __host__ __device__ T xor_system(T* adr,T val) {
        #ifdef __CUDA_ARCH__
            return atomicXor_system(adr,val);
        #else
            return __sync_fetch_and_xor(adr,val);
        #endif
    }

    template<typename T>
    __host__ __device__ T min_system(T* adr,T val) {
        #ifdef __CUDA_ARCH__
            return atomicMin_system(adr,val);
        #else
            return __sync_fetch_and_min(adr,val);
        #endif
    }

    template<typename T>
    __host__ __device__ T exch_system(T* adr,T val) {
        #ifdef __CUDA_ARCH__
            return atomicExch_system(adr,val);
        #else
            T result;
            __atomic_exchange(adr,&val,&result,__ATOMIC_ACQ_REL);
            return result;
        #endif
    }

    template<typename T>
    __host__ __device__ T CAS_system(T* adr,T comp,T val) {
        #ifdef __CUDA_ARCH__
            return atomicCAS_system(adr,comp,val);
        #else
            bool success = false;
            T expected = comp;
            while ((!success) && (comp == expected)) {
                success = __atomic_compare_exchange(adr,&expected,&val,false,__ATOMIC_ACQ_REL,__ATOMIC_ACQUIRE);
            }
            return expected;
        #endif
    }

    template<typename T>
    __host__ __device__ inline void store_relaxed(T* adr, T val) {
        #ifdef __CUDA_ARCH__
            *adr = val;
        #else
            __atomic_store_n(adr, val, __ATOMIC_RELAXED);
        #endif
    }

    template<typename T>
    __host__ __device__ inline T load_relaxed(const T* adr) {
       #ifdef __CUDA_ARCH__
            return *adr;
        #else
            return __atomic_load_n(adr, __ATOMIC_RELAXED);
        #endif
    }

    template<typename T>
    __host__ __device__ inline T load_acquire(const T* adr) {
        #ifdef __CUDA_ARCH__
            return *adr;
        #else
            return __atomic_load_n(adr, __ATOMIC_ACQUIRE);
        #endif
    }


};
# endif