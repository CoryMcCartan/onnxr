#include <cpp11.hpp>
#include <string>

// Include the ORT C API header for type declarations
#include "onnxruntime/onnxruntime_c_api.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

// Resolved function pointer to the real OrtGetApiBase
static const OrtApiBase* (*ort_get_api_base_fn)(void) = nullptr;

#ifdef _WIN32
static HMODULE ort_lib_handle = nullptr;
#else
static void* ort_lib_handle = nullptr;
#endif

// Provide the OrtGetApiBase symbol that the C++ headers expect.
// This forwards to the dynamically loaded implementation.
ORT_EXPORT const OrtApiBase* ORT_API_CALL OrtGetApiBase(void) NO_EXCEPTION {
    if (!ort_get_api_base_fn) {
        return nullptr;
    }
    return ort_get_api_base_fn();
}

// Check that ORT is loaded, stop with helpful message if not
void ort_check_loaded() {
    if (!ort_lib_handle) {
        cpp11::stop(
            "ONNX Runtime not loaded. "
            "Run ort_install() to download it, or install ORT system-wide."
        );
    }
}

[[cpp11::register]]
bool ort_load_lib(std::string path) {
    // Prevent double-loading and handle leak
    if (ort_lib_handle) return true;

#ifdef _WIN32
    ort_lib_handle = LoadLibraryA(path.c_str());
    if (!ort_lib_handle) return false;
    ort_get_api_base_fn = (const OrtApiBase* (*)(void))
        GetProcAddress(ort_lib_handle, "OrtGetApiBase");
#else
    ort_lib_handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!ort_lib_handle) return false;
    ort_get_api_base_fn = (const OrtApiBase* (*)(void))
        dlsym(ort_lib_handle, "OrtGetApiBase");
#endif
    if (!ort_get_api_base_fn) {
        // Loaded the library but couldn't find the symbol
#ifdef _WIN32
        FreeLibrary(ort_lib_handle);
        ort_lib_handle = nullptr;
#else
        dlclose(ort_lib_handle);
        ort_lib_handle = nullptr;
#endif
        return false;
    }
    return true;
}

[[cpp11::register]]
bool ort_is_loaded() {
    return ort_lib_handle != nullptr;
}

[[cpp11::register]]
void ort_unload_lib() {
    if (ort_lib_handle) {
#ifdef _WIN32
        FreeLibrary(ort_lib_handle);
#else
        dlclose(ort_lib_handle);
#endif
        ort_lib_handle = nullptr;
        ort_get_api_base_fn = nullptr;
    }
}
