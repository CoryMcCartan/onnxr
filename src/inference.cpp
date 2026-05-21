#include <cpp11.hpp>
#include <vector>
#include <string>

#ifdef HAVE_ORT
#include <onnxruntime_cxx_api.h>

// Permute flat array from column-major to row-major, double -> float
static void permute_colrow(const double* src, float* dst,
                           const std::vector<int64_t>& shape) {
  size_t ndim = shape.size();
  size_t n = 1;
  for (auto d : shape) n *= d;

  if (ndim <= 1) {
    for (size_t i = 0; i < n; i++) dst[i] = static_cast<float>(src[i]);
    return;
  }

  std::vector<size_t> rm_stride(ndim);
  rm_stride[ndim - 1] = 1;
  for (int k = ndim - 2; k >= 0; k--)
    rm_stride[k] = rm_stride[k + 1] * shape[k + 1];

  std::vector<size_t> cm_stride(ndim);
  cm_stride[0] = 1;
  for (size_t k = 1; k < ndim; k++)
    cm_stride[k] = cm_stride[k - 1] * shape[k - 1];

  for (size_t rm = 0; rm < n; rm++) {
    size_t tmp = rm;
    size_t cm = 0;
    for (size_t k = 0; k < ndim; k++) {
      size_t idx = tmp / rm_stride[k];
      tmp %= rm_stride[k];
      cm += idx * cm_stride[k];
    }
    dst[rm] = static_cast<float>(src[cm]);
  }
}

// Permute flat array from row-major to column-major, float -> double
static void permute_rowcol(const float* src, double* dst,
                           const std::vector<int64_t>& shape) {
  size_t ndim = shape.size();
  size_t n = 1;
  for (auto d : shape) n *= d;

  if (ndim <= 1) {
    for (size_t i = 0; i < n; i++) dst[i] = static_cast<double>(src[i]);
    return;
  }

  std::vector<size_t> rm_stride(ndim);
  rm_stride[ndim - 1] = 1;
  for (int k = ndim - 2; k >= 0; k--)
    rm_stride[k] = rm_stride[k + 1] * shape[k + 1];

  std::vector<size_t> cm_stride(ndim);
  cm_stride[0] = 1;
  for (size_t k = 1; k < ndim; k++)
    cm_stride[k] = cm_stride[k - 1] * shape[k - 1];

  for (size_t cm = 0; cm < n; cm++) {
    size_t tmp = cm;
    size_t rm = 0;
    for (int k = ndim - 1; k >= 0; k--) {
      size_t idx = tmp / cm_stride[k];
      tmp %= cm_stride[k];
      rm += idx * rm_stride[k];
    }
    dst[cm] = static_cast<double>(src[rm]);
  }
}

static cpp11::writable::doubles ort_run_impl(
  SEXP session_ptr,
  cpp11::doubles input_array,
  cpp11::integers input_shape,
  std::string input_name,
  std::string output_name
) {
  cpp11::external_pointer<Ort::Session> session(session_ptr);

  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
    OrtArenaAllocator,
    OrtMemTypeDefault
  );

  std::vector<int64_t> shape64(input_shape.begin(), input_shape.end());

  // Convert double -> float and column-major (R) -> row-major (ONNX)
  std::vector<float> float_data(input_array.size());
  permute_colrow(REAL(input_array), float_data.data(), shape64);

  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
    memory_info,
    float_data.data(),
    float_data.size(),
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
  for (auto dim : output_shape) output_size *= dim;

  // Convert float -> double and row-major (ONNX) -> column-major (R)
  cpp11::writable::doubles result(output_size);
  permute_rowcol(output_data, REAL(result), output_shape);

  cpp11::writable::integers shape_attr(output_shape.size());
  for (size_t i = 0; i < output_shape.size(); i++) {
    shape_attr[i] = static_cast<int>(output_shape[i]);
  }
  result.attr("dim") = shape_attr;

  return result;
}
#endif

[[cpp11::register]]
cpp11::writable::doubles ort_run(
  SEXP session_ptr,
  cpp11::doubles input_array,
  cpp11::integers input_shape,
  std::string input_name,
  std::string output_name
) {
#ifdef HAVE_ORT
  return ort_run_impl(session_ptr, input_array, input_shape, input_name, output_name);
#else
  cpp11::stop("ORT not installed. Run ort_install() then reinstall nativeORT.");
#endif
}
