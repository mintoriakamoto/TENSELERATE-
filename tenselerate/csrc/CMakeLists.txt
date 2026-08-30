# TENSELERATE native kernels.
#
# Two targets:
#   * tenselerate_gemm_test  — host C++ test of the int8 GEMM reference. Always
#     built; needs no CUDA, so every CI runner can run it.
#   * tenselerate_kernels    — the CUDA library (int8_gemm.cu ...). Built only
#     when CUDA is available, and validated against the same reference on a GPU.
#
# Standalone:  cmake -S tenselerate/csrc -B build-kernels && cmake --build build-kernels
cmake_minimum_required(VERSION 3.18)
project(tenselerate_kernels LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- host reference test (no CUDA) -----------------------------------------
add_executable(tenselerate_gemm_test gemm_test.cpp)
target_include_directories(tenselerate_gemm_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
enable_testing()
add_test(NAME gemm_reference COMMAND tenselerate_gemm_test)

# --- CUDA kernels (optional) ------------------------------------------------
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # The one supported machine: dual RTX 2080 Ti = Turing, sm_75. Turing
        # has int8 tensor cores (IMMA) and unthrottled dp4a, so the int8 path
        # applies. Other archs are opt-in via a preset if you build elsewhere.
        set(CMAKE_CUDA_ARCHITECTURES 75-real)
    endif()
    add_library(tenselerate_kernels STATIC int8_gemm.cu)
    target_include_directories(tenselerate_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    set_target_properties(tenselerate_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    # Shared library exposing the C ABI (tenselerate_int8_gemm) for ctypes -
    # tenselerate/backend/int8_gemm.py dlopen's this, never the static target.
    add_library(tenselerate_int8_gemm_c SHARED int8_gemm.cu)
    target_include_directories(tenselerate_int8_gemm_c PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    set_target_properties(tenselerate_int8_gemm_c PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON)

    message(STATUS "TENSELERATE: CUDA kernels enabled (arch ${CMAKE_CUDA_ARCHITECTURES})")
else()
    message(STATUS "TENSELERATE: no CUDA compiler — building host reference test only")
endif()
