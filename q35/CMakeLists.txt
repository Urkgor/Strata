# strata-q35: Strata's engine for Qwen3.6-35B-A3B (llama.cpp's qwen35moe graph, Strata's serve protocol).
#
# A project of its own, so it builds on an old PC too (RHEL 7: CMake 3.14, GCC 9+ from devtoolset, C++17):
#
#     cmake -S q35 -B build-q35 -DCMAKE_BUILD_TYPE=Release -DSTRATA_Q35_CUDA=ON
#     cmake --build build-q35 -j --target strata-q35
#
# The top-level CMakeLists.txt also takes it with -DSTRATA_ENABLE_Q35=ON.  docs/RHEL7.md has the whole recipe.
cmake_minimum_required(VERSION 3.14)
project(strata_q35 VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(STRATA_Q35_CUDA "Build ggml's CUDA backend (NVIDIA cards)" OFF)
option(STRATA_Q35_FIT "Fit the placement of the model to the free VRAM (llama.cpp's own fit, adds libcommon)" ON)
option(STRATA_Q35_PORTABLE "Build for other PCs of the same family: AVX2 baseline instead of this machine's CPU" OFF)
option(STRATA_Q35_STATIC_LIBSTDCXX "Link libstdc++ and libgcc statically (a binary built with devtoolset runs on the stock system)" ON)
set(STRATA_Q35_LLAMACPP_DIR "" CACHE PATH "a llama.cpp checkout (empty: fetch the commit this engine was written against)")
set(STRATA_Q35_LLAMACPP_TAG "3cf03257f219afbe7334045ff7c6a06ac68c627d" CACHE STRING "llama.cpp commit to fetch (the one in third_party/ggml/VERSION.txt)")

# ---- llama.cpp, static, only what this engine uses
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON ${STRATA_Q35_FIT} CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)        # ggml's own thread pool: same speed, no libgomp to carry around
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_CUDA ${STRATA_Q35_CUDA} CACHE BOOL "" FORCE)
if(STRATA_Q35_PORTABLE)
  set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
  foreach(_f GGML_AVX GGML_AVX2 GGML_FMA GGML_F16C GGML_BMI2)
    set(${_f} ON CACHE BOOL "" FORCE)
  endforeach()
else()
  set(GGML_NATIVE ON CACHE BOOL "" FORCE)
endif()

if(STRATA_Q35_LLAMACPP_DIR)
  add_subdirectory(${STRATA_Q35_LLAMACPP_DIR} ${CMAKE_BINARY_DIR}/llama.cpp EXCLUDE_FROM_ALL)
  set(_q35_llama_src ${STRATA_Q35_LLAMACPP_DIR})
else()
  include(FetchContent)
  FetchContent_Declare(strata_q35_llamacpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG ${STRATA_Q35_LLAMACPP_TAG}
    GIT_SHALLOW FALSE)
  FetchContent_GetProperties(strata_q35_llamacpp)
  if(NOT strata_q35_llamacpp_POPULATED)
    FetchContent_Populate(strata_q35_llamacpp)
  endif()
  add_subdirectory(${strata_q35_llamacpp_SOURCE_DIR} ${CMAKE_BINARY_DIR}/llama.cpp EXCLUDE_FROM_ALL)
  set(_q35_llama_src ${strata_q35_llamacpp_SOURCE_DIR})
endif()

# ---- the engine
add_executable(strata-q35 strata_q35.cpp)
target_compile_definitions(strata-q35 PRIVATE STRATA_Q35_VERSION="${PROJECT_VERSION}")
target_link_libraries(strata-q35 PRIVATE llama ggml)
if(STRATA_Q35_FIT)
  target_compile_definitions(strata-q35 PRIVATE STRATA_Q35_FIT=1)
  target_include_directories(strata-q35 PRIVATE ${_q35_llama_src}/common)
  target_link_libraries(strata-q35 PRIVATE llama-common)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(strata-q35 PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()
if(STRATA_Q35_STATIC_LIBSTDCXX AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_link_libraries(strata-q35 PRIVATE -static-libstdc++ -static-libgcc)
endif()
install(TARGETS strata-q35 RUNTIME DESTINATION bin)
