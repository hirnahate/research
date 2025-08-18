#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <memory>

#include "typedAlloc.h"

void testBasicTypeAllocation() {
    std::cout << "=== Basic Type Allocation Test ===" << std::endl;
    
    TestSlabArena arena;
    
    // Test integer allocation
    {
        TypeAllocator<int> intAlloc(arena);
        
        std::cout << "Integer allocator created" << std::endl;
        std::cout << "Object size: " << intAlloc.object_size() << " bytes" << std::endl;
        std::cout << "Object alignment: " << intAlloc.object_alignment() << " bytes" << std::endl;
        std::cout << "Max possible objects: " << intAlloc.max_size() << std::endl;
        
        // Allocate and construct
        int* ptr1 = intAlloc.create(42);
        int* ptr2 = intAlloc.create(100);
        
        if (ptr1 && ptr2) {
            std::cout << ":) Successfully allocated two integers" << std::endl;
            std::cout << "  ptr1 = " << ptr1 << ", value = " << *ptr1 << std::endl;
            std::cout << "  ptr2 = " << ptr2 << ", value = " << *ptr2 << std::endl;
            
            // Verify they're different
            if (ptr1 != ptr2) {
                std::cout << ":) Different pointers allocated" << std::endl;
            }
            
            // Clean up
            intAlloc.destroy_and_deallocate(ptr1);
            intAlloc.destroy_and_deallocate(ptr2);
            std::cout << ":) Objects destroyed and deallocated" << std::endl;
        } else {
            std::cout << ":( Failed to allocate integers" << std::endl;
        }
    }
    
    std::cout << std::endl;
}

void testComplexObjectAllocation() {
    std::cout << "=== Complex Object Allocation Test ===" << std::endl;
    
    TestSlabArena arena;
    TypeAllocator<TestObject> objAlloc(arena);
    
    std::cout << "TestObject size: " << sizeof(TestObject) << " bytes" << std::endl;
    std::cout << "Allocator object size: " << objAlloc.object_size() << " bytes" << std::endl;
    
    // Test default constructor
    TestObject* obj1 = objAlloc.create();
    if (obj1) {
        std::cout << ":) Default constructed object: value=" << obj1->value 
                  << ", data=" << obj1->data << std::endl;
        objAlloc.destroy_and_deallocate(obj1);
    }
    
    // Test parameterized constructor
    TestObject* obj2 = objAlloc.create(123, 45.67);
    if (obj2) {
        std::cout << ":) Parameterized object: value=" << obj2->value 
                  << ", data=" << obj2->data << std::endl;
        
        // Check buffer was initialized
        bool bufferOk = true;
        for (int i = 0; i < 32; i++) {
            if (obj2->buffer[i] != static_cast<char>(i)) {
                bufferOk = false;
                break;
            }
        }
        
        if (bufferOk) {
            std::cout << ":) Object buffer correctly initialized" << std::endl;
        } else {
            std::cout << ":( Object buffer initialization failed" << std::endl;
        }
        
        objAlloc.destroy_and_deallocate(obj2);
    }
    
    std::cout << std::endl;
}

void testMultipleTypeAllocators() {
    std::cout << "=== Multiple Type Allocators Test ===" << std::endl;
    
    TestSlabArena arena;
    
    // Different types using same arena
    TypeAllocator<int> intAlloc(arena);
    TypeAllocator<double> doubleAlloc(arena);
    TypeAllocator<TestObject> objAlloc(arena);
    
    std::vector<int*> ints;
    std::vector<double*> doubles;
    std::vector<TestObject*> objects;
    
    // Allocate mixed types
    const int count = 10;
    for (int i = 0; i < count; i++) {
        int* intPtr = intAlloc.create(i * 2);
        double* doublePtr = doubleAlloc.create(i * 3.14);
        TestObject* objPtr = objAlloc.create(i, i * 1.5);
        
        if (intPtr && doublePtr && objPtr) {
            ints.push_back(intPtr);
            doubles.push_back(doublePtr);
            objects.push_back(objPtr);
        } else {
            std::cout << ":( Allocation failed at iteration " << i << std::endl;
            break;
        }
    }
    
    std::cout << ":) Allocated " << ints.size() << " ints, " 
              << doubles.size() << " doubles, " 
              << objects.size() << " objects" << std::endl;
    
    // Verify allocations
    bool dataOk = true;
    for (size_t i = 0; i < ints.size(); i++) {
        if (*ints[i] != static_cast<int>(i * 2) ||
            *doubles[i] != (i * 3.14) ||
            objects[i]->value != static_cast<int>(i) ||
            objects[i]->data != (i * 1.5)) {
            dataOk = false;
            break;
        }
    }
    
    if (dataOk) {
        std::cout << ":) All allocated objects have correct values" << std::endl;
    } else {
        std::cout << ":( Data corruption detected!" << std::endl;
    }
    
    // Get statistics
    size_t intCount = intAlloc.allocated_count();
    size_t doubleCount = doubleAlloc.allocated_count();
    size_t objCount = objAlloc.allocated_count();
    
    std::cout << "Current allocations - ints: " << intCount 
              << ", doubles: " << doubleCount 
              << ", objects: " << objCount << std::endl;
    
    // Clean up
    for (int* ptr : ints) intAlloc.destroy_and_deallocate(ptr);
    for (double* ptr : doubles) doubleAlloc.destroy_and_deallocate(ptr);
    for (TestObject* ptr : objects) objAlloc.destroy_and_deallocate(ptr);
    
    std::cout << ":) All objects cleaned up" << std::endl;
    std::cout << std::endl;
}

void testFactoryFunctions() {
    std::cout << "=== Factory Functions Test ===" << std::endl;
    
    TestSlabArena arena;
    
    // Test factory function for object creation
    TestObject* obj1 = create_object<TestObject>(arena, 999, 88.77);
    if (obj1) {
        std::cout << ":) Factory created object: " << obj1->value << ", " << obj1->data << std::endl;
        destroy_object(arena, obj1);
        std::cout << ":) Factory destroyed object" << std::endl;
    }
    
    // Test make_type_allocator
    auto intAlloc = make_type_allocator<int>(arena);
    int* intPtr = intAlloc.create(12345);
    if (intPtr) {
        std::cout << ":) make_type_allocator created int: " << *intPtr << std::endl;
        intAlloc.destroy_and_deallocate(intPtr);
    }
    
    std::cout << std::endl;
}

void testAlignment() {
    std::cout << "=== Alignment Test ===" << std::endl;
    
    TestSlabArena arena;
    
    // Test different types with different alignment requirements
    struct AlignedStruct {
        alignas(16) double data[4];
        AlignedStruct() { 
            for(int i = 0; i < 4; i++) data[i] = i * 1.5; 
        }
    };
    
    TypeAllocator<AlignedStruct> alignedAlloc(arena);
    std::cout << "AlignedStruct size: " << sizeof(AlignedStruct) << std::endl;
    std::cout << "AlignedStruct alignment: " << alignof(AlignedStruct) << std::endl;
    std::cout << "Allocator reports alignment: " << alignedAlloc.object_alignment() << std::endl;
    
    AlignedStruct* ptr = alignedAlloc.create();
    if (ptr) {
        uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
        if (address % alignof(AlignedStruct) == 0) {
            std::cout << ":) Proper alignment maintained: " << std::hex << address << std::dec << std::endl;
        } else {
            std::cout << ":( Alignment violation!" << std::endl;
        }
        alignedAlloc.destroy_and_deallocate(ptr);
    }
    
    std::cout << std::endl;
}

void testErrorHandling() {
    std::cout << "=== Error Handling Test ===" << std::endl;
    
    TestSlabArena arena;
    TypeAllocator<TestObject> objAlloc(arena);
    
    // Test null pointer handling
    objAlloc.destroy_and_deallocate(nullptr);
    std::cout << ":) Null pointer handled gracefully" << std::endl;
    
    // Test invalid pointer (this should be handled gracefully)
    TestObject dummy;
    objAlloc.destroy_and_deallocate(&dummy);  // Not from our allocator
    std::cout << ":) Invalid pointer handled gracefully" << std::endl;
    
    // Test allocation failure (hard to trigger, but method exists)
    TestObject* ptr = objAlloc.allocate(0);  // Zero allocation
    if (!ptr) {
        std::cout << ":) Zero allocation correctly returns null" << std::endl;
    }
    
    std::cout << std::endl;
}

void performanceComparison() {
    std::cout << "=== Performance Comparison ===" << std::endl;
    
    const int iterations = 10000;
    
    // Test our type allocator
    {
        TestSlabArena arena;
        TypeAllocator<TestObject> typeAlloc(arena);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        std::vector<TestObject*> ptrs;
        ptrs.reserve(iterations);
        
        for (int i = 0; i < iterations; i++) {
            TestObject* ptr = typeAlloc.create(i, i * 2.5);
            if (ptr) ptrs.push_back(ptr);
        }
        
        for (TestObject* ptr : ptrs) {
            typeAlloc.destroy_and_deallocate(ptr);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "TypeAllocator: " << iterations << " alloc/free in " 
                  << duration.count() << "μs (" 
                  << (double)duration.count() / iterations << "μs per op)" << std::endl;
    }
    
    // Compare with standard new/delete
    {
        auto start = std::chrono::high_resolution_clock::now();
        
        std::vector<TestObject*> ptrs;
        ptrs.reserve(iterations);
        
        for (int i = 0; i < iterations; i++) {
            TestObject* ptr = new TestObject(i, i * 2.5);
            ptrs.push_back(ptr);
        }
        
        for (TestObject* ptr : ptrs) {
            delete ptr;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "Standard new/delete: " << iterations << " alloc/free in " 
                  << duration.count() << "μs (" 
                  << (double)duration.count() / iterations << "μs per op)" << std::endl;
    }
    
    std::cout << std::endl;
}

int main() {
    std::cout << "Type-Safe Slab Allocator Test Suite" << std::endl;
    std::cout << "====================================" << std::endl << std::endl;
    
    try {
        testBasicTypeAllocation();
        testComplexObjectAllocation();
        testMultipleTypeAllocators();
        testFactoryFunctions();
        testAlignment();
        testErrorHandling();
        performanceComparison();
        
        std::cout << "All type allocator tests completed!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown exception caught!" << std::endl;
        return 1;
    }
    
    return 0;
}
