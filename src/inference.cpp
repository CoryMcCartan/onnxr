#include <Rcpp.h>
#include <vector>

#ifdef HAVE_ORT
#include <onnxruntime_cxx_api.h>

// [[Rcpp::export]]
Rcpp::NumericVector ort_run(
  SEXP session_ptr,
  Rcpp::NumericVector input_array,
  std::vector<int> input_shape,
  std::string input_name,
  std::string output_name
) {
  Rcpp::XPtr<Ort::Session> session(session_ptr);

  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
    OrtArenaAllocator,
    OrtMemTypeDefault
  );

  std::vector<int64_t> shape64(input_shape.begin(), input_shape.end());

  // zero copy wrapper
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
    memory_info,
    reinterpret_cast<float*>(input_array.begin()),
    input_array.size(),
    shape64.data(),
    shape64.size()
  );

  const char* input_names[] = { input_name.c_str() };
  const char* output_names[] = { output_name.c_str() };

  auto output_tensors = session->Run(
    Ort::RunOptions{nullptr},
    input_names,
    &input_tensor,
    (size_t)1,
    output_names,
    (size_t)1
  );

  float* output_data = output_tensors[0].GetTensorMutableData<float>();
  auto output_info = output_tensors[0].GetTensorTypeAndShapeInfo();
  auto output_shape = output_info.GetShape();
  size_t output_size = 1;
  for (auto dim: output_shape) output_size *= dim;

  // copy back to R
  Rcpp::NumericVector result(output_size);
  for (size_t i = 0; i < output_size; i++) {
    result[i] = static_cast<double>(output_data[i]);
  }
  Rcpp::IntegerVector shape_attr(output_shape.size());
  for (size_t i = 0; i < output_shape.size(); i++) {
    shape_attr[i] = static_cast<int>(output_shape[i]);
  }
  result.attr("dim") = shape_attr;

  return result;
}

#else

Rcpp::NumericVector ort_run(
    SEXP session_ptr,
    Rcpp::NumericVector input_array,
    std::vector<int> input_shape,
    std::string input_name,
    std::string output_name
) {
  Rcpp::stop(
    "ORT not installed. Run ort_install() then reinstall nativeORT."
  );
}

#endif
