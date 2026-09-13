#include <sycl/sycl.hpp>
#include <cstdio>
int main() {
    auto platforms = sycl::platform::get_platforms();
    for (auto &p : platforms) {
        for (auto &d : p.get_devices()) {
            if (!d.is_gpu()) continue;
            printf("Device: %s\n", d.get_info<sycl::info::device::name>().c_str());
            printf("  aspect::ext_oneapi_graph = %d\n", d.has(sycl::aspect::ext_oneapi_graph));
            printf("  aspect::ext_oneapi_limited_graph = %d\n", d.has(sycl::aspect::ext_oneapi_limited_graph));
#ifdef SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
            printf("  SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC = defined\n");
#else
            printf("  SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC = NOT defined\n");
#endif
        }
    }
    return 0;
}
