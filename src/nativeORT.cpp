#include <cpp11.hpp>
#include <string>
#include "ort_loader.h"
#include "onnxruntime/onnxruntime_cxx_api.h"

[[cpp11::register]]
std::string ort_version() {
    ort_check_loaded();
    return std::string(OrtGetApiBase()->GetVersionString());
}
