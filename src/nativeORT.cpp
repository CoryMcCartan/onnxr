#include <Rcpp.h>

#ifdef HAVE_ORT
#include <onnxruntime_cxx_api.h>

// [[Rcpp::export]]
std::string ort_version() {
  return std::string(OrtGetApiBase()->GetVersionString());
}

#else

std::string ort_version() {
  Rcpp::stop(
    "ONNX Runtime not installed.\n"
    "Run ort_install() then re-install nativeORT to enable inference"
  );
  return "";
}

#endif
