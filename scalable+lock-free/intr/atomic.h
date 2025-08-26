#ifndef ATOMIC
#define ATOMIC

namespace atomic {

    template<typename T>
    __host__ __device__ T add_system(T* adr,T val) {
       #ifdef __CUDA_ARCH__
            if (sizeof(T) == 4) {
                return atomicAdd(reinterpret_cast<unsigned int*>(adr), static_cast<unsigned int>(val));
            } else if (sizeof(T) == 8) {
                return atomicAdd(reinterpret_cast<unsigned long long*>(adr), static_cast<unsigned long long>(val));
            }
        #else
            return __sync_fetch_and_add(adr, val);
        #endif
        return T{};
    }

    template<typename T>
    __host__ __device__ T sub_system(T* adr,T val) {
        #ifdef __CUDA_ARCH__
            if (sizeof(T) == 4) {
                return atomicSub(reinterpret_cast<unsigned int*>(adr), static_cast<unsigned int>(val));
            } else if (sizeof(T) == 8) {
                return atomicAdd(reinterpret_cast<unsigned long long*>(adr), 
                               static_cast<unsigned long long>(-static_cast<long long>(val)));
            }
        #else
            return __sync_fetch_and_sub(adr, val);
        #endif
        return T{};
    }

    template<typename T>
    __host__ __device__ T and_system(T* adr,T val) {
         #ifdef __CUDA_ARCH__
            if (sizeof(T) == 4) {
                return atomicAnd(reinterpret_cast<unsigned int*>(adr), static_cast<unsigned int>(val));
            } else if (sizeof(T) == 8) {
                return atomicAnd(reinterpret_cast<unsigned long long*>(adr), static_cast<unsigned long long>(val));
            }
        #else
            return __sync_fetch_and_and(adr, val);
        #endif
        return T{};
    }

    template<typename T>
    __host__ __device__ T or_system(T* adr,T val) {
        #ifdef __CUDA_ARCH__
            return atomicOr_system(adr,val);
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
            if (sizeof(T) == 4) {
                return atomicExch(reinterpret_cast<unsigned int*>(adr), static_cast<unsigned int>(val));
            } else if (sizeof(T) == 8) {
                return atomicExch(reinterpret_cast<unsigned long long*>(adr), static_cast<unsigned long long>(val));
            }
        #else
            T result;
            __atomic_exchange(adr, &val, &result, __ATOMIC_ACQ_REL);
            return result;
        #endif
        return T{};
    }

    template<typename T>
    __host__ __device__ T CAS_system(T* adr,T comp,T val) {
        #ifdef __CUDA_ARCH__
            if (sizeof(T) == 4) {
                return atomicCAS(reinterpret_cast<unsigned int*>(adr), 
                               static_cast<unsigned int>(comp), 
                               static_cast<unsigned int>(val));
            } else if (sizeof(T) == 8) {
                return atomicCAS(reinterpret_cast<unsigned long long*>(adr), 
                               static_cast<unsigned long long>(comp), 
                               static_cast<unsigned long long>(val));
            }
        #else
            T expected = comp;
            __atomic_compare_exchange_n(adr, &expected, val, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
            return expected;
        #endif
        return T{};
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