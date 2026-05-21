#include <cpp11.hpp>
#include <string>

#ifdef HAVE_ORT
#include <onnxruntime_cxx_api.h>
#endif

[[cpp11::register]]
std::string ort_version() {
#ifdef HAVE_ORT
  return std::string(OrtGetApiBase()->GetVersionString());
#else
  cpp11::stop(
    "ONNX Runtime not installed.\n"
    "Run ort_install() then re-install nativeORT to enable inference"
  );
  return "";
#endif
}
