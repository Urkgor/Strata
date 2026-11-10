// cuda_cpu_emu.hpp - CUDA on the CPU, enough to run a kernel and its host code without a GPU (a copy of the Gemma 4 engine's g4_emu.hpp; here for tests/core/q36_sample_test.cpp).
//
// A kernel launch runs the blocks one after another on one OS thread; the `blockDim.x` threads of a block are fibers with a stack each,
// __syncthreads() hands over to the scheduler until every thread has arrived (a thread that has finished counts as arrived; a barrier
// that some thread can never reach is reported as a deadlock), __shared__ is a static array (only one block runs at a time) and the warp
// shuffles are exchanges through a staging array between two warp-wide barriers.  "Device" memory is malloc'd host memory, one allocation per cudaMalloc, so AddressSanitizer sees every kernel's
// out-of-bounds access.  What it cannot show: speed, occupancy, shared memory or register limits, coalescing, a divergent __syncthreads (here
// a deadlock, there undefined behaviour: the kernels never do it).
#pragma once

#include <atomic>
#include <barrier>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

struct dim3 {
    unsigned x = 1, y = 1, z = 1;
    dim3() = default;
    dim3(unsigned a, unsigned b = 1, unsigned c = 1) : x(a), y(b), z(c) {}
};
inline thread_local dim3 threadIdx, blockIdx, blockDim, gridDim;

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __shared__ static
#define __constant__ static const
#define __launch_bounds__(...)

namespace g4emu {

// ---- fibers: every CUDA thread of the running block is a stack of its own on this one OS thread; a barrier is a hand-over to the scheduler.
// x86-64 System V: the context switch saves the callee-saved registers and the stack pointer.
#if !defined(__x86_64__)
#error "the CUDA emulation (g4_emu.hpp) switches fibers in x86-64 assembly"
#endif
extern "C" void g4emu_switch(void** save_sp, void* load_sp);
asm(R"(
.text
.globl g4emu_switch
.type g4emu_switch,@function
g4emu_switch:
    pushq %rbp
    pushq %rbx
    pushq %r12
    pushq %r13
    pushq %r14
    pushq %r15
    movq %rsp, (%rdi)
    movq %rsi, %rsp
    popq %r15
    popq %r14
    popq %r13
    popq %r12
    popq %rbx
    popq %rbp
    ret
.size g4emu_switch,.-g4emu_switch
)");

struct Fiber {
    void* sp = nullptr;
    std::vector<unsigned char> stack;
    bool done = false;
    int wait = 0;                // 0 running / runnable, 1 at a block barrier, 2 at a warp barrier
    unsigned wait_gen = 0;
};

struct Sched {
    std::vector<Fiber> fibers;
    void* sched_sp = nullptr;
    int B = 0;
    int cur = 0;
    const std::function<void()>* job = nullptr;
    int active = 0;                       // fibers not finished
    int block_arrived = 0;
    unsigned block_gen = 0;
    std::vector<int> warp_arrived, warp_active;
    std::vector<unsigned> warp_gen;
    alignas(16) unsigned char stage[1 << 14];      // the shuffles' exchange area: up to 1024 threads x 16 bytes
    static Sched*& current() { static Sched* s = nullptr; return s; }
};

inline void fiber_entry() {
    Sched* s = Sched::current();
    Fiber& f = s->fibers[(size_t) s->cur];
    (*s->job)();
    f.done = true;
    const int w = s->cur >> 5;
    --s->active;
    --s->warp_active[(size_t) w];
    // a finished thread counts as arrived at the barriers the others are waiting at
    if (s->active > 0 && s->block_arrived == s->active) { s->block_arrived = 0; ++s->block_gen; }
    if (s->warp_active[(size_t) w] > 0 && s->warp_arrived[(size_t) w] == s->warp_active[(size_t) w]) { s->warp_arrived[(size_t) w] = 0; ++s->warp_gen[(size_t) w]; }
    g4emu_switch(&f.sp, s->sched_sp);
    std::abort();                                   // never resumed
}

inline void yield_to_scheduler() {
    Sched* s = Sched::current();
    Fiber& f = s->fibers[(size_t) s->cur];
    g4emu_switch(&f.sp, s->sched_sp);
}

inline void sync() {
    Sched* s = Sched::current();
    Fiber& f = s->fibers[(size_t) s->cur];
    if (++s->block_arrived == s->active) {
        s->block_arrived = 0;
        ++s->block_gen;
        return;
    }
    f.wait = 1;
    f.wait_gen = s->block_gen;
    yield_to_scheduler();
    f.wait = 0;
}
inline void warp_sync() {
    Sched* s = Sched::current();
    Fiber& f = s->fibers[(size_t) s->cur];
    const size_t w = (size_t) (s->cur >> 5);
    if (++s->warp_arrived[w] == s->warp_active[w]) {
        s->warp_arrived[w] = 0;
        ++s->warp_gen[w];
        return;
    }
    f.wait = 2;
    f.wait_gen = s->warp_gen[w];
    yield_to_scheduler();
    f.wait = 0;
}

inline Sched& sched_for(int b) {
    static std::map<int, std::unique_ptr<Sched>> all;
    auto& p = all[b];
    if (!p) {
        p = std::make_unique<Sched>();
        p->B = b;
        p->fibers.resize((size_t) b);
        for (auto& f : p->fibers) f.stack.resize(160 * 1024);
        p->warp_arrived.resize((size_t) b / 32);
        p->warp_active.resize((size_t) b / 32);
        p->warp_gen.resize((size_t) b / 32);
    }
    return *p;
}

inline void launch(dim3 grid, dim3 block, const std::function<void()>& kernel) {
    if (grid.x == 0 || grid.y == 0 || grid.z == 0) return;
    if (block.y != 1 || block.z != 1 || block.x == 0 || block.x > 1024 || block.x % 32) {
        std::fprintf(stderr, "g4emu: a block of %u x %u x %u threads is not emulated (a multiple of 32, one dimension)\n", block.x, block.y, block.z);
        std::abort();
    }
    Sched& s = sched_for((int) block.x);
    Sched::current() = &s;
    s.job = &kernel;
    blockDim = dim3(block.x, 1, 1);
    gridDim = grid;
    for (unsigned z = 0; z < grid.z; ++z)
        for (unsigned y = 0; y < grid.y; ++y)
            for (unsigned x = 0; x < grid.x; ++x) {
                blockIdx = dim3(x, y, z);
                s.active = s.B;
                s.block_arrived = 0;
                for (size_t w = 0; w < s.warp_active.size(); ++w) { s.warp_active[w] = 32; s.warp_arrived[w] = 0; }
                for (int i = 0; i < s.B; ++i) {
                    Fiber& f = s.fibers[(size_t) i];
                    f.done = false;
                    f.wait = 0;
                    unsigned char* top = f.stack.data() + f.stack.size();
                    top = (unsigned char*) ((uintptr_t) top & ~(uintptr_t) 15);
                    void** sp = (void**) (top - 16);
                    sp[0] = (void*) &fiber_entry;           // the `ret` of the first switch
                    sp[1] = nullptr;
                    sp -= 6;                                // the six saved registers
                    for (int r = 0; r < 6; ++r) sp[r] = nullptr;
                    f.sp = sp;
                }
                int remaining = s.B;
                while (remaining > 0) {
                    bool progressed = false;
                    for (int i = 0; i < s.B; ++i) {
                        Fiber& f = s.fibers[(size_t) i];
                        if (f.done) continue;
                        if (f.wait == 1 && f.wait_gen == s.block_gen) continue;
                        if (f.wait == 2 && f.wait_gen == s.warp_gen[(size_t) (i >> 5)]) continue;
                        s.cur = i;
                        threadIdx = dim3((unsigned) i, 0, 0);
                        g4emu_switch(&s.sched_sp, f.sp);
                        progressed = true;
                        if (f.done) --remaining;
                    }
                    if (!progressed) {
                        std::fprintf(stderr, "g4emu: deadlock in block (%u,%u,%u): %d threads wait at a barrier that not every thread reaches\n", x, y, z, remaining);
                        std::abort();
                    }
                }
            }
}

template <class T>
inline T shfl_xor(T v, int m) {
    static_assert(sizeof(T) <= 16, "");
    Sched* s = Sched::current();
    T* st = reinterpret_cast<T*>(s->stage);
    const int tid = (int) threadIdx.x;
    st[tid] = v;
    warp_sync();
    const T r = st[(tid & ~31) | ((tid & 31) ^ m)];
    warp_sync();
    return r;
}
template <class T>
inline T shfl_down(T v, int d) {
    Sched* s = Sched::current();
    T* st = reinterpret_cast<T*>(s->stage);
    const int tid = (int) threadIdx.x;
    st[tid] = v;
    warp_sync();
    const T r = (tid & 31) + d < 32 ? st[tid + d] : v;
    warp_sync();
    return r;
}
template <class T>
inline T shfl_idx(T v, int src) {
    Sched* s = Sched::current();
    T* st = reinterpret_cast<T*>(s->stage);
    const int tid = (int) threadIdx.x;
    st[tid] = v;
    warp_sync();
    const T r = st[(tid & ~31) | (src & 31)];
    warp_sync();
    return r;
}

}  // namespace g4emu

#define __syncthreads() g4emu::sync()
#define __shfl_xor_sync(mask, v, m) g4emu::shfl_xor((v), (m))
#define __shfl_down_sync(mask, v, d) g4emu::shfl_down((v), (d))
#define __shfl_sync(mask, v, src) g4emu::shfl_idx((v), (src))

inline int min(int a, int b) { return a < b ? a : b; }
inline int max(int a, int b) { return a > b ? a : b; }
inline float rsqrtf_(float x) { return 1.0f / std::sqrt(x); }
#define rsqrtf rsqrtf_
inline float atomicAdd(float* p, float v) { return std::atomic_ref<float>(*p).fetch_add(v); }
inline int atomicAdd(int* p, int v) { return std::atomic_ref<int>(*p).fetch_add(v); }

// ---- the runtime API, as much as the engine calls
using cudaError_t = int;
enum { cudaSuccess = 0, cudaErrorInvalidValue = 1 };
using cudaStream_t = void*;
enum cudaMemcpyKind { cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice, cudaMemcpyDefault };
inline int& g4emu_device() { static thread_local int d = 0; return d; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline const char* cudaGetErrorString(cudaError_t) { return "no error (emulated)"; }
inline cudaError_t cudaMalloc(void** p, size_t n) { *p = std::calloc(1, n ? n : 1); return *p ? cudaSuccess : cudaErrorInvalidValue; }
inline cudaError_t cudaFree(void* p) { std::free(p); return cudaSuccess; }
enum { cudaHostAllocPortable = 1 };
inline cudaError_t cudaHostAlloc(void** p, size_t n, unsigned) { *p = std::calloc(1, n ? n : 1); return *p ? cudaSuccess : cudaErrorInvalidValue; }
inline cudaError_t cudaFreeHost(void* p) { std::free(p); return cudaSuccess; }
inline cudaError_t cudaMemset(void* p, int v, size_t n) { std::memset(p, v, n); return cudaSuccess; }
inline cudaError_t cudaMemcpy(void* d, const void* s, size_t n, cudaMemcpyKind) { std::memmove(d, s, n); return cudaSuccess; }
inline cudaError_t cudaMemcpyAsync(void* d, const void* s, size_t n, cudaMemcpyKind, cudaStream_t) { std::memmove(d, s, n); return cudaSuccess; }
inline cudaError_t cudaMemsetAsync(void* p, int v, size_t n, cudaStream_t) { std::memset(p, v, n); return cudaSuccess; }
inline cudaError_t cudaStreamCreate(cudaStream_t* s) { *s = nullptr; return cudaSuccess; }
inline cudaError_t cudaStreamDestroy(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
inline cudaError_t cudaSetDevice(int d) { g4emu_device() = d; return cudaSuccess; }
inline cudaError_t cudaGetDevice(int* d) { *d = g4emu_device(); return cudaSuccess; }
inline cudaError_t cudaGetDeviceCount(int* n) { const char* e = std::getenv("STRATA_G4_EMU_DEVICES"); *n = e ? std::atoi(e) : 2; return cudaSuccess; }
inline cudaError_t cudaMemGetInfo(size_t* fr, size_t* tot) {
    const char* e = std::getenv("STRATA_G4_EMU_FREE_MB");      // per device: a comma list, else every device has 8 GiB
    double mb = 8192;
    if (e) { int d = g4emu_device(); const char* q = e; for (; d > 0 && std::strchr(q, ','); --d) q = std::strchr(q, ',') + 1; mb = std::atof(q); }
    *fr = (size_t) (mb * 1048576.0);
    *tot = (size_t) ((mb + 512) * 1048576.0);
    return cudaSuccess;
}
struct cudaDeviceProp { char name[256]; int major = 6, minor = 1; size_t totalGlobalMem = 0; };
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* p, int d) { std::snprintf(p->name, sizeof p->name, "emulated device %d", d); return cudaSuccess; }
