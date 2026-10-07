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

# ---- the patches: the hybrid expert cache changes llama.cpp's graph builder and its CPU kernels (q35/patches)
file(GLOB _q35_patches ${CMAKE_CURRENT_SOURCE_DIR}/patches/*.patch)
foreach(_p ${_q35_patches})
  get_filename_component(_pn ${_p} NAME_WE)
  set(_stamp ${_q35_llama_src}/.strata-patched-${_pn})
  if(NOT EXISTS ${_stamp})
    find_program(_q35_patch_exe patch)
    find_program(_q35_git_exe git)
    if(_q35_patch_exe)
      execute_process(COMMAND ${_q35_patch_exe} -p1 -R --dry-run -s -f INPUT_FILE ${_p} WORKING_DIRECTORY ${_q35_llama_src}
                      RESULT_VARIABLE _already OUTPUT_QUIET ERROR_QUIET)
      if(NOT _already EQUAL 0)
        execute_process(COMMAND ${_q35_patch_exe} -p1 -s -f INPUT_FILE ${_p} WORKING_DIRECTORY ${_q35_llama_src} RESULT_VARIABLE _rc)
      else()
        set(_rc 0)
      endif()
    elseif(_q35_git_exe)
      execute_process(COMMAND ${_q35_git_exe} apply --reverse --check ${_p} WORKING_DIRECTORY ${_q35_llama_src}
                      RESULT_VARIABLE _already OUTPUT_QUIET ERROR_QUIET)
      if(NOT _already EQUAL 0)
        execute_process(COMMAND ${_q35_git_exe} apply ${_p} WORKING_DIRECTORY ${_q35_llama_src} RESULT_VARIABLE _rc)
      else()
        set(_rc 0)
      endif()
    else()
      message(FATAL_ERROR "strata-q35 needs patch(1) or git to apply ${_p} to llama.cpp")
    endif()
    if(NOT _rc EQUAL 0)
      message(FATAL_ERROR "${_p} does not apply to ${_q35_llama_src} (the commit this engine was written against: ${STRATA_Q35_LLAMACPP_TAG})")
    endif()
    file(WRITE ${_stamp} "applied\n")
    message(STATUS "strata-q35: applied ${_pn}")
  endif()
endforeach()

# ---- the engine
add_executable(strata-q35 strata_q35.cpp hybrid_experts.cpp)
target_compile_definitions(strata-q35 PRIVATE STRATA_Q35_VERSION="${PROJECT_VERSION}")
target_include_directories(strata-q35 PRIVATE ${_q35_llama_src}/src)   # llama-hybrid.h (added by the patch)
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

# the expert cache's policy, with no llama.cpp and no GPU:  cmake --build build-q35 --target test-expert-cache
add_executable(test-expert-cache EXCLUDE_FROM_ALL test_expert_cache.cpp)
enable_testing()
add_test(NAME expert-cache COMMAND test-expert-cache)
