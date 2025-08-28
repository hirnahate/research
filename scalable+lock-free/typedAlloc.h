#pragma once
#include "allocator.h"
#include <new>
#include <type_traits>

// quick trait to check if a type has a trivial destructor
template<typename T>
struct is_trivially_destructible {
    static const bool value = std::is_trivially_destructible<T>::value;
};

// a type-safe allocator wrapper around our slab allocator
template <typename T, typename SLAB_ALLOCATOR_TYPE = TestSlabArena>
class TypeAllocator {
public:
    typedef T value_type;
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef T& reference;
    typedef const T& const_reference;
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    
    typedef SLAB_ALLOCATOR_TYPE SlabAllocatorType;
    typedef SimpleAllocator<SlabAllocatorType> UnderlyingAllocatorType;

    // lets us rebind to other types (needed for stl containers)
    template<typename U>
    struct rebind {
        typedef TypeAllocator<U, SLAB_ALLOCATOR_TYPE> other;
    };

private:
    SlabAllocatorType& slabAllocator;
    UnderlyingAllocatorType underlyingAllocator;
    
    static const size_t OBJECT_SIZE = sizeof(T);
    static const size_t OBJECT_ALIGNMENT = alignof(T);

public:
    // constructor
    __host__ __device__
    explicit TypeAllocator(SlabAllocatorType& allocator) 
        : slabAllocator(allocator)
        , underlyingAllocator(allocator, OBJECT_SIZE) {}

    // copy construct
    __host__ __device__
    TypeAllocator(const TypeAllocator& other) 
        : slabAllocator(other.slabAllocator)
        , underlyingAllocator(other.slabAllocator, OBJECT_SIZE) {}

    // rebind copy construct
    template<typename U>
    __host__ __device__
    TypeAllocator(const TypeAllocator<U, SLAB_ALLOCATOR_TYPE>& other)
        : slabAllocator(other.getSlabAllocator())
        , underlyingAllocator(other.getSlabAllocator(), OBJECT_SIZE) {}

    // assignment operator (mostly useless here since we can’t reassign references)
    __host__ __device__
    TypeAllocator& operator=(const TypeAllocator& other) {
        if (this != &other) {
            // nothing we can really do since slabAllocator is a ref...
        }
        return *this;
    }

    // dtor (just default it)
    ~TypeAllocator() = default;

    
    // allocates raw memory for n objects (supports 1 now)
    __host__ __device__
    pointer allocate(size_type n = 1) {
        if (n == 0) return nullptr;
        
        // only single objects for now
        if (n != 1) {
            return nullptr;
        }
        
        void* raw_ptr = underlyingAllocator.alloc();
        if (!raw_ptr) {
            return nullptr;
        }
        
        // double-check alignment
        if (reinterpret_cast<uintptr_t>(raw_ptr) % OBJECT_ALIGNMENT != 0) {
            // shouldn’t happen if slabs are aligned right
            underlyingAllocator.free(raw_ptr);
            return nullptr;
        }
        
        return static_cast<pointer>(raw_ptr);
    }
    
    // frees raw memory
    __host__ __device__
    void deallocate(pointer ptr, size_type n = 1) {
        if (!ptr || n == 0) return;
        
        if (!underlyingAllocator.isValidPtr(ptr)) {
            // invalid pointer — just bail
            return;
        }
        
        underlyingAllocator.free(ptr);
    }
    
    
    // placement...new construct
    template<typename... Args>
    __host__ 
    void construct(pointer ptr, Args&&... args) {
        if (!ptr) return;
        ::new (static_cast<void*>(ptr)) T(static_cast<Args&&>(args)...);
    }
    
    // manually call destructor if needed
    __host__ __device__
    void destroy(pointer ptr) {
        if (!ptr) return;
        if (!is_trivially_destructible<T>::value) {
            ptr->~T();
        }
    }
    
    
    // alloc + construct in one go
    template<typename... Args>
    __host__
    pointer create(Args&&... args) {
        pointer ptr = allocate(1);
        if (!ptr) {
            return nullptr;
        }
        
        #ifdef __CUDA_ARCH__
            construct(ptr, static_cast<Args&&>(args)...);
            return ptr;
        #else
            try {
                construct(ptr, static_cast<Args&&>(args)...);
                return ptr;
            } catch (...) {
                deallocate(ptr, 1);
                return nullptr;
            }
        #endif
    }
    
    // destroy + free in one go
    __host__ __device__
    void destroy_and_deallocate(pointer ptr) {
        if (!ptr) return;
        destroy(ptr);
        deallocate(ptr, 1);
    }
    
    
    __host__ __device__
    size_type max_size() const {
        return SLAB_ALLOCATOR_TYPE::SLAB_COUNT * 
               (SLAB_ALLOCATOR_TYPE::slabType::SIZE / OBJECT_SIZE);
    }
    
    __host__ __device__
    size_type object_size() const {
        return OBJECT_SIZE;
    }
    
    __host__ __device__
    size_type object_alignment() const {
        return OBJECT_ALIGNMENT;
    }
    
    __host__ __device__
    size_type allocated_count() const {
        return const_cast<UnderlyingAllocatorType&>(underlyingAllocator).getAllocatedCount();
    }
    
    __host__ __device__
    void getStats(size_type& activeSlabs, size_type& reusableSlabs, size_type& totalObjects) const {
        const_cast<UnderlyingAllocatorType&>(underlyingAllocator).getReuseStats(activeSlabs, reusableSlabs, totalObjects);
    }
    
    __host__ __device__
    SlabAllocatorType& getSlabAllocator() const {
        return slabAllocator;
    }
    
    // stl wants these comparisons...
    __host__ __device__
    bool operator==(const TypeAllocator& other) const {
        return &slabAllocator == &other.slabAllocator;
    }
    
    __host__ __device__
    bool operator!=(const TypeAllocator& other) const {
        return !(*this == other);
    }
};

// placeholder for array types (not done yet)
template <typename T, size_t N, typename SLAB_ALLOCATOR_TYPE = TestSlabArena>
class ArrayTypeAllocator {
public:
    typedef T value_type[N];
    typedef T* pointer;
    typedef const T* const_pointer;
    // todo: handle arrays of N
};

// shortcut helpers
template<typename T, typename SlabAllocatorType>
__host__ __device__
TypeAllocator<T, SlabAllocatorType> make_type_allocator(SlabAllocatorType& allocator) {
    return TypeAllocator<T, SlabAllocatorType>(allocator);
}

template<typename T, typename SlabAllocatorType, typename... Args>
__host__
T* create_object(SlabAllocatorType& allocator, Args&&... args) {
    TypeAllocator<T, SlabAllocatorType> alloc(allocator);
    return alloc.create(static_cast<Args&&>(args)...);
}

template<typename T, typename SlabAllocatorType>
__host__ __device__
void destroy_object(SlabAllocatorType& allocator, T* obj) {
    TypeAllocator<T, SlabAllocatorType> alloc(allocator);
    alloc.destroy_and_deallocate(obj);
}

// some test structs
struct TestObject {
    int value;
    double data;
    char buffer[32];
    
    __host__ __device__
    TestObject() : value(0), data(0.0) {
        for(int i = 0; i < 32; i++) buffer[i] = 0;
    }
    
    __host__ __device__
    TestObject(int v, double d) : value(v), data(d) {
        for(int i = 0; i < 32; i++) buffer[i] = static_cast<char>(i);
    }
    
    __host__ __device__
    ~TestObject() {
        // no-op
    }
};

struct ComplexObject {
    int* dynamicData;
    size_t size;
    
    __host__ __device__
    ComplexObject(size_t s) : size(s) {
        #ifdef __CUDA_ARCH__
        dynamicData = nullptr; // device alloc would go here
        #else
        dynamicData = new int[size];
        for(size_t i = 0; i < size; i++) {
            dynamicData[i] = static_cast<int>(i);
        }
        #endif
    }
    
    __host__ __device__
    ~ComplexObject() {
        #ifdef __CUDA_ARCH__
        // device free would go here
        #else
        delete[] dynamicData;
        #endif
    }
    
    ComplexObject(const ComplexObject&) = delete;
    ComplexObject& operator=(const ComplexObject&) = delete;
};

// handy aliases
typedef TypeAllocator<int> IntAllocator;
typedef TypeAllocator<double> DoubleAllocator;
typedef TypeAllocator<TestObject> TestObjectAllocator;
typedef TypeAllocator<ComplexObject> ComplexObjectAllocator;
