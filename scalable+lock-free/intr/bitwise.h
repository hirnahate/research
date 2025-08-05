#ifndef BITWISE
#define BITWISE


namespace bitwise {

__host__ __device__
size_t population_count(unsigned int val) {
    #ifdef __CUDA_ARCH__
        return __popc(val);
    #else
        return __builtin_popcount(val);
    #endif
}

__host__ __device__
size_t population_count(unsigned long long int val) {
    #ifdef __CUDA_ARCH__
        return __popcll(val);
    #else
        return __builtin_popcountll(val);
    #endif
}

__host__ __device__
size_t leading_zeros(unsigned int val) {
    #ifdef __CUDA_ARCH__
        return __clz(val);
    #else
        return __builtin_clz(val);
    #endif
}

__host__ __device__
size_t leading_zeros(unsigned long long int val) {
    #ifdef __CUDA_ARCH__
        return __clzll(val);
    #else
        return __builtin_clzll(val);
    #endif
}

__host__ __device__
size_t first_set(unsigned int val) {
    #ifdef __CUDA_ARCH__
        return __ffs(val);
    #else
        return __builtin_ffs(val);
    #endif
}

__host__ __device__
size_t first_set(unsigned long long int val) {
    #ifdef __CUDA_ARCH__
        return __ffsll(val);
    #else
        return __builtin_ffsll(val);
    #endif
}


}


#endif
