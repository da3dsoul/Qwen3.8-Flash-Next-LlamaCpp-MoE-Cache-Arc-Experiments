// Phase 2 validation for ggml_backend_sycl_moe_cached_buffer_type() (see
// src/llama.cpp/ggml/src/ggml-sycl/moe-cache.{hpp,cpp}). Standalone -- not
// part of llama.cpp's own test suite (AGENTS.md asks contributors not to add
// files under tests/ without maintainer approval, and this isn't meant to be
// upstreamed as-is anyway). Links directly against the built ggml/llama
// shared libraries.
//
// Checks, per PLAN.md Phase 2's "validate with direct tensor read/write
// tests, not full model inference":
//   1. A tensor allocated through the new buffer type round-trips known data
//      correctly (host-side set/get).
//   2. is_host() is false and the identity check recognizes only this type.
//   3. A context whose tensors sum to more than the 2 GiB per-buffer cap
//      gets split into multiple underlying allocations (via ggml's own
//      ggml_backend_alloc_ctx_tensors_from_buft, not custom code here), and
//      every tensor across that split still round-trips correctly -- this is
//      the actual test of the "no custom slab->(chunk,offset) map needed"
//      claim in moe-cache.cpp's header comment.
//   4. A real host->device->host round trip: copy a tensor's data from our
//      cold-tier buffer to an ordinary SYCL device buffer, then back to a
//      fresh host buffer, and compare -- confirms the pinned-host pointers
//      this buffer type hands out are genuinely usable by the SYCL runtime
//      for PCIe transfer, not just plain-CPU-accessible memory.

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <random>
#include <thread>
#include <vector>
#include <algorithm>

#include <sycl/sycl.hpp>
#include "dpct/helper.hpp"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-sycl.h"

// Declared in ggml-sycl/moe-cache.hpp, which isn't part of the public
// ggml-sycl.h surface (Phase 3's dispatch hook is the only intended future
// caller) -- redeclare here (plain C++ linkage, matching the header exactly)
// to avoid depending on ggml-sycl's private include path from an external
// test binary.
ggml_backend_buffer_type_t ggml_backend_sycl_moe_cached_buffer_type(void);
bool ggml_backend_buft_is_sycl_moe_cached(ggml_backend_buffer_type_t buft);

// Phase 3d: device-planned admission + gather. `sycl_moe_cache` stays
// opaque here (pointer only, never dereferenced) -- matches how moe-cache.hpp
// itself declares it.
struct sycl_moe_cache;
sycl_moe_cache * ggml_sycl_moe_cache_get_or_create(const ggml_tensor * src0, sycl::queue & compute_q);
const int32_t * ggml_sycl_moe_cache_plan_and_gather(sycl_moe_cache * cache, const ggml_tensor * src0,
                                                     sycl::queue & compute_q, const int32_t * ids_dev,
                                                     int n_ids_per_group);
void * ggml_sycl_moe_cache_pool_base(sycl_moe_cache * cache);
size_t ggml_sycl_moe_cache_slot_stride(sycl_moe_cache * cache);
int    ggml_sycl_moe_cache_debug_last_events(sycl::event * out, int max_out);
int    ggml_sycl_moe_cache_debug_n_misses(sycl_moe_cache * cache, sycl::queue & q);

// A queue in the SAME context as the buffer type's pinned allocations (which
// go through dpct's default queue), but with profiling enabled -- the default
// queue has no profiling property, and nanosecond device timestamps are the
// entire point of this benchmark. In-order, matching the production
// ctx.stream() the real dispatch hook uses, so kernel ordering here is the
// same ordering the real path relies on for correctness.
static sycl::queue make_profiling_queue() {
    sycl::queue base = dpct::dev_mgr::instance().get_device(0).default_queue();
    return sycl::queue(base.get_context(), base.get_device(),
                       sycl::property_list{ sycl::property::queue::in_order(),
                                            sycl::property::queue::enable_profiling() });
}

static double event_ms(const sycl::event & e) {
    const auto t0 = e.get_profiling_info<sycl::info::event_profiling::command_start>();
    const auto t1 = e.get_profiling_info<sycl::info::event_profiling::command_end>();
    return (double) (t1 - t0) / 1.0e6;
}

// Per-launch dispatch cost on this stack, measured rather than assumed: the
// "two kernels must be why it is slow" theory this project has been carrying
// around only holds if a launch actually costs something comparable to the
// work. A single-work-item kernel that writes one int is as close to zero
// device work as a launch gets.
static void test_launch_overhead() {
    sycl::queue q = make_profiling_queue();
    fprintf(stderr, "\n=== launch overhead ===\n");
    fprintf(stderr, "device: %s, max_compute_units=%u\n",
            q.get_device().get_info<sycl::info::device::name>().c_str(),
            q.get_device().get_info<sycl::info::device::max_compute_units>());

    int32_t * scratch = sycl::malloc_device<int32_t>(1, q);
    const int n = 2000;

    auto submit_noop = [&](int i) {
        return q.submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(1), sycl::range<1>(1)),
                             [=](sycl::nd_item<1>) { *scratch = i; });
        });
    };

    // warm up (first submit pays JIT/program-build, which is not per-launch cost)
    submit_noop(0);
    q.wait();

    // (a) pipelined: submit everything, wait once. This is what the real
    // dispatch path does -- it never waits between ops.
    auto w0 = std::chrono::steady_clock::now();
    double dev_ns_total = 0.0;
    std::vector<sycl::event> evs;
    evs.reserve(n);
    for (int i = 0; i < n; i++) evs.push_back(submit_noop(i));
    q.wait();
    auto w1 = std::chrono::steady_clock::now();
    for (auto & e : evs) dev_ns_total += event_ms(e) * 1.0e6;
    const double pipelined_us = std::chrono::duration<double, std::micro>(w1 - w0).count() / n;

    // (b) serialized: submit + wait each time, i.e. what a host round trip costs.
    auto s0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; i++) { submit_noop(i); q.wait(); }
    auto s1 = std::chrono::steady_clock::now();
    const double serialized_us = std::chrono::duration<double, std::micro>(s1 - s0).count() / n;

    fprintf(stderr, "no-op kernel, %d launches:\n", n);
    fprintf(stderr, "  pipelined  (submit x%d, one wait): %.2f us/launch wall, %.2f us/launch device\n",
            n, pipelined_us, dev_ns_total / n / 1000.0);
    fprintf(stderr, "  serialized (submit + wait each)  : %.2f us/launch wall\n", serialized_us);
    fprintf(stderr, "  => a second kernel per MoE op costs ~%.2f us; at 144 ops/token that is %.2f ms/token\n",
            pipelined_us, pipelined_us * 144 / 1000.0);

    sycl::free(scratch, q);
}

static int g_failures = 0;

#define CHECK(cond, msg) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); \
            g_failures++; \
        } else { \
            fprintf(stderr, "OK:   %s\n", msg); \
        } \
    } while (0)

static std::vector<float> random_floats(size_t n, uint32_t seed) {
    std::vector<float> v(n);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto & x : v) {
        x = dist(rng);
    }
    return v;
}

static void test_basic_roundtrip() {
    ggml_backend_buffer_type_t buft = ggml_backend_sycl_moe_cached_buffer_type();

    CHECK(buft != nullptr, "buffer type created");
    CHECK(!ggml_backend_buft_is_host(buft), "is_host() is false for the cache buffer type");
    CHECK(ggml_backend_buft_is_sycl_moe_cached(buft), "identity check recognizes its own type");
    CHECK(!ggml_backend_buft_is_sycl_moe_cached(ggml_backend_cpu_buffer_type()),
          "identity check rejects the plain CPU buffer type");

    struct ggml_init_params params = { /*.mem_size=*/ ggml_tensor_overhead() * 4, /*.mem_buffer=*/ nullptr,
                                        /*.no_alloc=*/ true };
    struct ggml_context * ctx = ggml_init(params);

    const int64_t ne = 4096;
    struct ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    CHECK(buf != nullptr, "buffer allocated for a small single tensor");

    std::vector<float> data = random_floats(ne, 42);
    ggml_backend_tensor_set(t, data.data(), 0, ne * sizeof(float));

    std::vector<float> readback(ne);
    ggml_backend_tensor_get(t, readback.data(), 0, ne * sizeof(float));

    CHECK(memcmp(data.data(), readback.data(), ne * sizeof(float)) == 0,
          "small tensor round-trips byte-identical through the cache buffer type");

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

static void test_multi_buffer_split() {
    // Force more than one 2 GiB chunk: 3 tensors of ~900 MiB each (~2.6 GiB
    // total) so ggml_backend_alloc_ctx_tensors_from_buft_impl must split
    // across at least two underlying sycl::malloc_host allocations.
    const int64_t ne_per_tensor = (900LL * 1024 * 1024) / sizeof(float);
    const int     n_tensors     = 3;

    struct ggml_init_params params = { /*.mem_size=*/ ggml_tensor_overhead() * (n_tensors + 1),
                                        /*.mem_buffer=*/ nullptr, /*.no_alloc=*/ true };
    struct ggml_context * ctx = ggml_init(params);

    std::vector<struct ggml_tensor *> tensors;
    for (int i = 0; i < n_tensors; i++) {
        tensors.push_back(ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne_per_tensor));
    }

    ggml_backend_buffer_type_t buft = ggml_backend_sycl_moe_cached_buffer_type();
    ggml_backend_buffer_t      buf  = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    CHECK(buf != nullptr, "buffer allocated for a >2 GiB context (multiple chunks expected)");

    bool all_ok = true;
    for (int i = 0; i < n_tensors; i++) {
        std::vector<float> data = random_floats(ne_per_tensor, 1000 + i);
        ggml_backend_tensor_set(tensors[i], data.data(), 0, ne_per_tensor * sizeof(float));

        std::vector<float> readback(ne_per_tensor);
        ggml_backend_tensor_get(tensors[i], readback.data(), 0, ne_per_tensor * sizeof(float));

        if (memcmp(data.data(), readback.data(), ne_per_tensor * sizeof(float)) != 0) {
            all_ok = false;
            fprintf(stderr, "  tensor %d mismatched after round-trip\n", i);
        }
    }
    CHECK(all_ok, "every tensor round-trips correctly across the auto-split multi-buffer allocation");

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

static void test_host_device_host_roundtrip() {
    struct ggml_init_params params = { /*.mem_size=*/ ggml_tensor_overhead() * 4, /*.mem_buffer=*/ nullptr,
                                        /*.no_alloc=*/ true };
    struct ggml_context * ctx_cold = ggml_init(params);
    struct ggml_context * ctx_dev  = ggml_init(params);
    struct ggml_context * ctx_host = ggml_init(params);

    const int64_t ne = 65536; // 256 KiB, comfortably a single real expert-slab-sized transfer

    struct ggml_tensor * t_cold = ggml_new_tensor_1d(ctx_cold, GGML_TYPE_F32, ne);
    struct ggml_tensor * t_dev  = ggml_new_tensor_1d(ctx_dev, GGML_TYPE_F32, ne);
    struct ggml_tensor * t_host = ggml_new_tensor_1d(ctx_host, GGML_TYPE_F32, ne);

    ggml_backend_buffer_t buf_cold = ggml_backend_alloc_ctx_tensors_from_buft(ctx_cold, ggml_backend_sycl_moe_cached_buffer_type());
    ggml_backend_buffer_t buf_dev  = ggml_backend_alloc_ctx_tensors_from_buft(ctx_dev, ggml_backend_sycl_buffer_type(0));
    ggml_backend_buffer_t buf_host = ggml_backend_alloc_ctx_tensors_from_buft(ctx_host, ggml_backend_cpu_buffer_type());

    CHECK(buf_cold && buf_dev && buf_host, "cold/device/host buffers all allocated");

    std::vector<float> data = random_floats(ne, 7);
    ggml_backend_tensor_set(t_cold, data.data(), 0, ne * sizeof(float));

    // cold (pinned host, our buffer type) -> device
    ggml_backend_tensor_copy(t_cold, t_dev);
    // device -> host (plain CPU buffer, distinct from the cache's pinned tier)
    ggml_backend_tensor_copy(t_dev, t_host);

    std::vector<float> readback(ne);
    ggml_backend_tensor_get(t_host, readback.data(), 0, ne * sizeof(float));

    CHECK(memcmp(data.data(), readback.data(), ne * sizeof(float)) == 0,
          "cold-tier -> SYCL device -> host round trip is byte-identical");

    ggml_backend_buffer_free(buf_cold);
    ggml_backend_buffer_free(buf_dev);
    ggml_backend_buffer_free(buf_host);
    ggml_free(ctx_cold);
    ggml_free(ctx_dev);
    ggml_free(ctx_host);
}

// Phase 3d benchmark/stress harness, parameterized by MODEL SHAPE.
//
// This started as a hang repro for the device-planned admission kernel and
// grew into the project's only fast (seconds, not a 10+ minute model load)
// measurement loop for the plan and gather kernels. It mirrors real usage as
// closely as a standalone binary can: real per-expert row byte counts, real
// n_as / n_slots / n_ids_per_group, many consecutive calls with randomized
// routing (so hit+miss mixes and same-call eviction-protection cases both
// occur repeatedly), device-side SYCL event timestamps for timing, a
// byte-exact check that each routed expert's slot really holds that expert's
// bytes, and a watchdog so a device hang prints a diagnostic instead of
// silently wedging the binary.
//
// TWO shapes matter, and they behave very differently -- which is the whole
// reason this is parameterized rather than hardcoded to one:
//
//   smoke  (Qwen3.6-35B-A3B, the fast one): 256 experts, 8 routed,
//          Q4_K gate/up 589824 B/expert, Q5_K down 720896 B/expert.
//          Cold tensor total: 256 * 589824 = 151 MiB.
//   flash  (Qwen3.8-Flash-Next, the target): 512 experts, 10 routed,
//          IQ3_XXS gate/up 627200 B/expert (2560x640, 98 B per 256-block),
//          IQ4_NL down 921600 B/expert (640x2560, 18 B per 32-block),
//          Q8_0 down on 5 of 48 layers 1740800 B/expert.
//          Cold tensor total: 512 * 627200 = 306 MiB / 512 * 921600 = 450 MiB.
//
// Every one of those row byte counts is a multiple of 16, so the gather's
// word-width autodetect lands on 16 B for all of them -- verified, not
// assumed (the kernel prints the width it chose).
namespace {

struct moe_shape {
    const char * name;
    int64_t      row_cols;   // f32 tensor ne[0]
    int64_t      row_rows;   // f32 tensor ne[1]; row_bytes = cols*rows*4
    int64_t      n_as;
    int          n_ids;
    int          n_slots;
};

// row_bytes = row_cols * row_rows * 4 must equal the real quantized nb[2].
const moe_shape k_shapes[] = {
    // 2048 * 72 * 4 = 589824  (Q4_K 2048x512 expert)
    { "smoke-q4k-gate",  2048,  72, 256,  8, 96 },
    // 1568 * 100 * 4 = 627200 (IQ3_XXS 2560x640 expert)
    { "flash-iq3xxs-gate", 1568, 100, 512, 10, 96 },
    // 1536 * 150 * 4 = 921600 (IQ4_NL 640x2560 expert)
    { "flash-iq4nl-down", 1536, 150, 512, 10, 96 },
};

std::vector<ggml_backend_buffer_t> g_kept_buffers;
std::vector<ggml_context *>        g_kept_contexts;

struct shape_result {
    double plan_ms   = 0.0;
    double gather_ms = 0.0;
    double misses    = 0.0;
    double gb_s      = 0.0;
    bool   ok        = false;
};

// Content for expert `e`, recomputed rather than kept in a giant host array
// (a 512-expert flash tensor is 450 MiB; three of them plus a verification
// copy would be gigabytes of host RAM on an already-tight shared box).
void fill_expert_row(std::vector<float> & row, int64_t e) {
    for (size_t i = 0; i < row.size(); i++) {
        row[i] = (float) (e * 7919 + (int64_t) (i % 65521));
    }
}

// Copy-engine reference for the same transfer the gather kernel does. Phase
// 3b serviced misses with exactly this (sycl::queue::memcpy from the pinned
// cold tier into the slot pool) and reached 11.2-12.9 t/s on Flash-Next,
// where the kernel gather reaches 4.1 -- so "is the compute-kernel gather
// actually slower than a plain DMA memcpy at THIS shape?" is the single most
// load-bearing question this harness can answer, and it can only answer it
// by measuring both against the same bytes.
double bench_memcpy_gather(sycl::queue & q, const char * cold_base, char * pool_base, size_t row_bytes,
                           const std::vector<int32_t> & experts, const std::vector<int32_t> & slots) {
    std::vector<sycl::event> evs;
    evs.reserve(experts.size());
    for (size_t i = 0; i < experts.size(); i++) {
        evs.push_back(q.memcpy(pool_base + (size_t) slots[i] * row_bytes,
                               cold_base + (size_t) experts[i] * row_bytes, row_bytes));
    }
    q.wait();
    // Wall-clock span across all of them (they are enqueued back to back on
    // one in-order queue, so start-of-first to end-of-last is the real cost).
    uint64_t t0 = UINT64_MAX, t1 = 0;
    for (auto & e : evs) {
        t0 = std::min<uint64_t>(t0, e.get_profiling_info<sycl::info::event_profiling::command_start>());
        t1 = std::max<uint64_t>(t1, e.get_profiling_info<sycl::info::event_profiling::command_end>());
    }
    return (double) (t1 - t0) / 1.0e6;
}

shape_result run_shape(const moe_shape & sh, int n_calls, bool verify) {
    shape_result res;
    sycl::queue q = make_profiling_queue();

    const size_t row_bytes_expected = (size_t) sh.row_cols * sh.row_rows * sizeof(float);

    struct ggml_init_params params = { /*.mem_size=*/ ggml_tensor_overhead() * 2, /*.mem_buffer=*/ nullptr,
                                        /*.no_alloc=*/ true };
    struct ggml_context * ctx = ggml_init(params);
    struct ggml_tensor * src0 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, sh.row_cols, sh.row_rows, sh.n_as);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_sycl_moe_cached_buffer_type());
    if (buf == nullptr) {
        fprintf(stderr, "FAIL: %s: cold-tier tensor allocation failed (%.1f MiB)\n",
                sh.name, (double) (row_bytes_expected * sh.n_as) / 1048576.0);
        g_failures++;
        ggml_free(ctx);
        return res;
    }

    const size_t floats_per_expert = (size_t) sh.row_cols * sh.row_rows;
    std::vector<float> row(floats_per_expert);
    for (int64_t e = 0; e < sh.n_as; e++) {
        fill_expert_row(row, e);
        ggml_backend_tensor_set(src0, row.data(), (size_t) e * row_bytes_expected, row_bytes_expected);
    }

    sycl_moe_cache * cache = ggml_sycl_moe_cache_get_or_create(src0, q);
    if (cache == nullptr) {
        fprintf(stderr, "FAIL: %s: cache creation failed\n", sh.name);
        g_failures++;
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return res;
    }

    const size_t row_bytes = ggml_sycl_moe_cache_slot_stride(cache);
    char * const pool_base = (char *) ggml_sycl_moe_cache_pool_base(cache);
    const char * cold_base = (const char *) src0->data;

    int word_bytes = 16;
    while (word_bytes > 1 &&
           ((row_bytes % (size_t) word_bytes) != 0 ||
            ((uintptr_t) cold_base % (uintptr_t) word_bytes) != 0 ||
            ((uintptr_t) pool_base % (uintptr_t) word_bytes) != 0)) {
        word_bytes /= 2;
    }

    fprintf(stderr, "\n=== shape %s ===\n", sh.name);
    fprintf(stderr, "row_bytes=%zu, n_as=%lld, n_slots=%d, n_ids=%d, calls=%d\n",
            row_bytes, (long long) sh.n_as, sh.n_slots, sh.n_ids, n_calls);
    fprintf(stderr, "cold tier %.1f MiB @ %p, slot pool %.1f MiB @ %p, gather word width %d B\n",
            (double) (row_bytes * sh.n_as) / 1048576.0, (const void *) cold_base,
            (double) (row_bytes * sh.n_slots) / 1048576.0, (void *) pool_base, word_bytes);
    if (row_bytes != row_bytes_expected) {
        fprintf(stderr, "FAIL: %s: row_bytes %zu != intended %zu\n", sh.name, row_bytes, row_bytes_expected);
        g_failures++;
    }

    int32_t * ids_dev = (int32_t *) sycl::malloc_device(sh.n_ids * sizeof(int32_t), q);

    std::mt19937 rng(123);
    std::uniform_int_distribution<int32_t> hot_dist(0, 15);
    std::uniform_int_distribution<int32_t> full_dist(0, (int32_t) sh.n_as - 1);

    bool                 hung = false;
    std::vector<int32_t> host_ids(sh.n_ids);
    std::vector<int32_t> remapped_host(sh.n_ids);
    std::vector<char>    slot_readback(row_bytes);
    std::vector<float>   expect_row(floats_per_expert);
    double               plan_ms_total  = 0.0;
    double               gath_ms_total  = 0.0;
    double               wall_ms_total  = 0.0;
    long long            miss_total     = 0;
    int                  gather_verifications = 0;
    bool                 gather_ok = true;

    for (int call = 0; call < n_calls && !hung; ++call) {
        for (int i = 0; i < sh.n_ids; i++) {
            host_ids[i] = (call % 3 == 0) ? hot_dist(rng) : full_dist(rng);
        }
        q.memcpy(ids_dev, host_ids.data(), sh.n_ids * sizeof(int32_t)).wait();

        const auto t0 = std::chrono::steady_clock::now();
        const int32_t * remapped = ggml_sycl_moe_cache_plan_and_gather(cache, src0, q, ids_dev, sh.n_ids);

        std::atomic<bool> done{false};
        std::thread waiter([&]() { q.wait(); done = true; });
        for (int waited_ms = 0; waited_ms < 15000 && !done; waited_ms += 50) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        if (!done) {
            fprintf(stderr, "FAIL: %s: call %d did NOT complete within 15s -- HANG\n", sh.name, call);
            g_failures++;
            hung = true;
            waiter.detach();
            break;
        }
        waiter.join();
        const auto t1 = std::chrono::steady_clock::now();
        wall_ms_total += std::chrono::duration<double, std::milli>(t1 - t0).count();

        sycl::event evs[2];
        const int   n_evs = ggml_sycl_moe_cache_debug_last_events(evs, 2);
        const double plan_ms = n_evs >= 1 ? event_ms(evs[0]) : 0.0;
        const double gath_ms = n_evs >= 2 ? event_ms(evs[1]) : 0.0;
        plan_ms_total += plan_ms;
        gath_ms_total += gath_ms;
        miss_total    += ggml_sycl_moe_cache_debug_n_misses(cache, q);

        q.memcpy(remapped_host.data(), remapped, sh.n_ids * sizeof(int32_t)).wait();
        bool all_valid = true;
        for (int i = 0; i < sh.n_ids; i++) {
            if (remapped_host[i] < 0 || remapped_host[i] >= sh.n_slots) all_valid = false;
        }
        if (!all_valid) {
            fprintf(stderr, "  %s call %3d: remapped OUT OF RANGE\n", sh.name, call);
            g_failures++;
        }

        if (verify && all_valid && call % 25 == 0) {
            for (int i = 0; i < sh.n_ids && gather_ok; i++) {
                q.memcpy(slot_readback.data(), pool_base + (size_t) remapped_host[i] * row_bytes, row_bytes).wait();
                fill_expert_row(expect_row, host_ids[i]);
                if (memcmp(slot_readback.data(), expect_row.data(), row_bytes) != 0) {
                    fprintf(stderr, "  %s call %3d: slot %d does NOT contain expert %d's bytes\n",
                            sh.name, call, remapped_host[i], host_ids[i]);
                    gather_ok = false;
                }
            }
            gather_verifications++;
        }
    }

    if (!hung) {
        char msg[256];
        snprintf(msg, sizeof(msg), "%s: all calls completed without hanging", sh.name);
        CHECK(true, msg);
        if (verify) {
            snprintf(msg, sizeof(msg), "%s: gathered slots contain exactly the routed experts' bytes", sh.name);
            CHECK(gather_ok && gather_verifications > 0, msg);
        }

        res.plan_ms   = plan_ms_total / n_calls;
        res.gather_ms = gath_ms_total / n_calls;
        res.misses    = (double) miss_total / n_calls;
        const double bytes_per_call = res.misses * (double) row_bytes;
        res.gb_s      = res.gather_ms > 0.0 ? bytes_per_call / (res.gather_ms / 1000.0) / 1e9 : 0.0;
        res.ok        = gather_ok;

        const double dev_avg = res.plan_ms + res.gather_ms;
        fprintf(stderr, "  device per call: plan %.3f ms + gather %.3f ms = %.3f ms\n",
                res.plan_ms, res.gather_ms, dev_avg);
        fprintf(stderr, "  wall per call (incl. host wait): %.3f ms\n", wall_ms_total / n_calls);
        fprintf(stderr, "  misses/call %.2f of %d routed (%.0f%% hit) => %.2f MiB/call\n",
                res.misses, sh.n_ids, 100.0 * (1.0 - res.misses / sh.n_ids), bytes_per_call / 1048576.0);
        fprintf(stderr, "  GATHER KERNEL BANDWIDTH: %.2f GB/s\n", res.gb_s);
        fprintf(stderr, "  x144 ops/token => %.1f ms/token of cache work alone (%.1f t/s ceiling)\n",
                dev_avg * 144.0, dev_avg > 0.0 ? 1000.0 / (dev_avg * 144.0) : 0.0);

        // Copy-engine A/B at exactly this shape and miss count.
        const int n_miss_typ = (int) (res.misses + 0.5);
        if (n_miss_typ > 0) {
            std::vector<int32_t> exps(n_miss_typ), slts(n_miss_typ);
            for (int i = 0; i < n_miss_typ; i++) {
                exps[i] = full_dist(rng);
                slts[i] = i % sh.n_slots;
            }
            bench_memcpy_gather(q, cold_base, pool_base, row_bytes, exps, slts); // warm
            double best = 1e30;
            for (int rep = 0; rep < 10; rep++) {
                for (int i = 0; i < n_miss_typ; i++) exps[i] = full_dist(rng);
                best = std::min(best, bench_memcpy_gather(q, cold_base, pool_base, row_bytes, exps, slts));
            }
            const double cp_bytes = (double) n_miss_typ * row_bytes;
            fprintf(stderr, "  copy-engine reference (%d x memcpy, best of 10): %.3f ms => %.2f GB/s\n",
                    n_miss_typ, best, cp_bytes / (best / 1000.0) / 1e9);
        }
    }

    sycl::free(ids_dev, q);
    // DELIBERATELY NOT FREED. ggml_sycl_moe_cache_get_or_create keys its
    // registry on the ggml_tensor POINTER, and ggml_init hands the next
    // context's first tensor the same address a just-freed one had -- so
    // freeing here made shape 2 and 3 silently reuse shape 1's cache (wrong
    // stride, wrong n_as, wrong pool). That was caught only because the
    // harness asserts row_bytes and verifies gathered bytes; it is worth
    // knowing the production registry has the same aliasing hazard if a
    // model is ever unloaded and reloaded in one process.
    g_kept_buffers.push_back(buf);
    g_kept_contexts.push_back(ctx);
    return res;
}

} // namespace

// ---------------------------------------------------------------------------
// Host-USM FOOTPRINT sweep.
//
// The per-shape benchmark above allocates ONE cold tensor (306-450 MiB) and
// measures ~24 GB/s, i.e. PCIe 4.0 x16 line rate, at both models' shapes. The
// real runs do not look like that at all: every cached expert tensor in the
// model is its own pinned allocation and a token touches ALL of them in turn.
//   35B-A3B  : 144 tensors, 256 experts   -> ~22 GiB of pinned host
//   Flash-Next: 144 tensors, 512 experts  -> ~53 GiB of pinned host
// The single-tensor benchmark cannot see anything that depends on that total,
// which is exactly the axis on which the two models differ most. This sweeps
// it directly: same 16 B-word device-wide-grid copy, same row size, random
// rows drawn uniformly across a cold region of growing total size.
//
// Both engines are measured at every size, because they are the two designs
// this project has actually shipped: Phase 3b serviced misses with
// sycl::queue::memcpy (the Level Zero copy engine) and Phase 3d with a
// compute kernel. A divergence that only opens up at large footprints would
// explain why Phase 3d wins on the 22 GiB model and loses on the 53 GiB one.
namespace {

struct alignas(16) sweep_word16 { uint32_t v[4]; };

sycl::event submit_sweep_gather(sycl::queue & q, const char * const * src_rows_dev, char * dst_base,
                                size_t row_bytes, int n_rows) {
    constexpr int wg_size = 256;
    const size_t words_per_row  = row_bytes / sizeof(sweep_word16);
    const size_t blocks_per_row = (words_per_row + wg_size - 1) / wg_size;
    const size_t n_wg           = blocks_per_row * (size_t) n_rows;

    return q.submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(n_wg * wg_size), sycl::range<1>(wg_size)),
            [=](sycl::nd_item<1> item) {
                const size_t lid = item.get_local_id(0);
                const size_t blk = item.get_group(0);
                const size_t r   = blk / blocks_per_row;
                const size_t c   = blk - r * blocks_per_row;
                const sweep_word16 * src = (const sweep_word16 *) src_rows_dev[r];
                sweep_word16 *       dst = (sweep_word16 *) (dst_base + r * row_bytes);
                const size_t idx = c * (size_t) wg_size + lid;
                if (idx < words_per_row) {
                    dst[idx] = src[idx];
                }
            });
    });
}

void test_host_usm_footprint_sweep() {
    sycl::queue q = make_profiling_queue();

    const size_t row_bytes   = 921600;               // real IQ4_NL ffn_down expert row
    const size_t chunk_bytes = 450ull * 1024 * 1024; // ~ one real 512-expert cold tensor
    const size_t rows_per_chunk = chunk_bytes / row_bytes;
    const int    n_rows      = 8;                    // ~ misses per op at 96/512 residency

    const char * cap_env  = getenv("MOE_TEST_MAX_FOOTPRINT_GIB");
    size_t max_gib = cap_env ? (size_t) atoll(cap_env) : 16;

    fprintf(stderr, "\n=== host-USM footprint sweep ===\n");
    fprintf(stderr, "row_bytes=%zu, rows/copy=%d, chunk=%zu MiB, max footprint %zu GiB\n",
            row_bytes, n_rows, chunk_bytes / 1048576, max_gib);

    char * dst_base = (char *) sycl::malloc_device(row_bytes * n_rows, q);
    const char ** src_rows_dev = (const char **) sycl::malloc_device(sizeof(char *) * n_rows, q);
    std::vector<const char *> src_rows_host(n_rows);
    std::vector<char *> chunks;

    std::mt19937 rng(4242);
    size_t allocated_gib = 0;

    while (allocated_gib < max_gib) {
        // Grow the pinned footprint by ~1 GiB, then measure.
        for (int i = 0; i < 2 && allocated_gib < max_gib; i++) {
            char * p = (char *) sycl::malloc_host(chunk_bytes, q);
            if (p == nullptr) {
                fprintf(stderr, "  malloc_host failed at %zu GiB -- stopping sweep\n", allocated_gib);
                max_gib = allocated_gib;
                break;
            }
            memset(p, (int) (chunks.size() & 0xFF), chunk_bytes); // commit every page
            chunks.push_back(p);
            allocated_gib = chunks.size() * chunk_bytes / (1024ull * 1024 * 1024);
        }
        if (chunks.empty()) break;

        auto pick_rows = [&]() {
            for (int r = 0; r < n_rows; r++) {
                const size_t ch  = rng() % chunks.size();
                const size_t row = rng() % rows_per_chunk;
                src_rows_host[r] = chunks[ch] + row * row_bytes;
            }
            q.memcpy((void *) src_rows_dev, src_rows_host.data(), sizeof(char *) * n_rows).wait();
        };

        // Kernel gather, best of 20 (best, not mean, so a scheduler hiccup on
        // this shared box cannot masquerade as a bandwidth cliff).
        double best_kernel = 1e30;
        for (int rep = 0; rep < 20; rep++) {
            pick_rows();
            sycl::event e = submit_sweep_gather(q, src_rows_dev, dst_base, row_bytes, n_rows);
            q.wait();
            best_kernel = std::min(best_kernel, event_ms(e));
        }

        // Copy engine, same rows, same total bytes.
        double best_copy = 1e30;
        for (int rep = 0; rep < 20; rep++) {
            pick_rows();
            std::vector<sycl::event> evs;
            for (int r = 0; r < n_rows; r++) {
                evs.push_back(q.memcpy(dst_base + (size_t) r * row_bytes, src_rows_host[r], row_bytes));
            }
            q.wait();
            uint64_t t0 = UINT64_MAX, t1 = 0;
            for (auto & e : evs) {
                t0 = std::min<uint64_t>(t0, e.get_profiling_info<sycl::info::event_profiling::command_start>());
                t1 = std::max<uint64_t>(t1, e.get_profiling_info<sycl::info::event_profiling::command_end>());
            }
            best_copy = std::min(best_copy, (double) (t1 - t0) / 1.0e6);
        }

        const double bytes = (double) row_bytes * n_rows;
        fprintf(stderr, "  footprint %5.1f GiB (%3zu chunks): kernel %6.3f ms = %5.2f GB/s | copy-engine %6.3f ms = %5.2f GB/s\n",
                (double) (chunks.size() * chunk_bytes) / (1024.0 * 1024 * 1024), chunks.size(),
                best_kernel, bytes / (best_kernel / 1000.0) / 1e9,
                best_copy,   bytes / (best_copy   / 1000.0) / 1e9);
    }

    for (char * p : chunks) {
        sycl::free(p, q);
    }
    sycl::free(dst_base, q);
    sycl::free((void *) src_rows_dev, q);
}

} // namespace

int main() {
    ggml_backend_load_all();

    // The buffer-type conformance tests (notably the 2.6 GiB multi-buffer
    // split) take far longer than the benchmarks they precede; MOE_TEST_QUICK
    // skips them so a gather A/B sweep is seconds per variant.
    if (getenv("MOE_TEST_QUICK") == nullptr) {
        test_basic_roundtrip();
        test_multi_buffer_split();
        test_host_device_host_roundtrip();
    }
    if (getenv("MOE_TEST_NO_LAUNCH_BENCH") == nullptr) {
        test_launch_overhead();
    }

    const char * calls_env = getenv("MOE_TEST_CALLS");
    const int    n_calls   = calls_env ? atoi(calls_env) : 200;
    const char * only      = getenv("MOE_TEST_SHAPES"); // substring filter, e.g. "flash"
    const bool   verify    = getenv("MOE_TEST_NO_VERIFY") == nullptr;

    if (getenv("MOE_TEST_FOOTPRINT") != nullptr) {
        test_host_usm_footprint_sweep();
        fprintf(stderr, "\n%s (%d failures)\n", g_failures == 0 ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED", g_failures);
        return g_failures == 0 ? 0 : 1;
    }

    for (const moe_shape & sh : k_shapes) {
        if (only != nullptr && strstr(sh.name, only) == nullptr) {
            continue;
        }
        run_shape(sh, n_calls, verify);
    }

    fprintf(stderr, "\n%s (%d failures)\n", g_failures == 0 ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED", g_failures);
    return g_failures == 0 ? 0 : 1;
}
