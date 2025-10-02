cmake_minimum_required(VERSION 3.18)
project(hello_tilelang_cpp VERSION 1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags for optimization
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# Default to Release build if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Print build configuration
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")

# Create executable
add_executable(hello_vector_add hello_vector_add.cpp)

# Enable warnings
target_compile_options(hello_vector_add PRIVATE
    -Wall -Wextra -Wpedantic
)

# Installation
install(TARGETS hello_vector_add
    RUNTIME DESTINATION bin
)

# Print success message
message(STATUS "Configured hello_vector_add C++ example")
message(STATUS "Build with: cmake --build build")
message(STATUS "Run with: ./build/hello_vector_add")
