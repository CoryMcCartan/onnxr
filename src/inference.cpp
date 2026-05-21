#include <cpp11.hpp>
#include <vector>
#include <string>
#include "ort_loader.h"
#include "onnxruntime/onnxruntime_cxx_api.h"

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

// Permute row-major -> column-major for floating point output
template <typename T>
static void permute_rowcol_to_double(const T* src, double* dst,
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

// Permute row-major -> column-major for integer output
template <typename T>
static void permute_rowcol_to_int(const T* src, int* dst,
                                  const std::vector<int64_t>& shape) {
    size_t ndim = shape.size();
    size_t n = 1;
    for (auto d : shape) n *= d;

    if (ndim <= 1) {
        for (size_t i = 0; i < n; i++) dst[i] = static_cast<int>(src[i]);
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
        dst[cm] = static_cast<int>(src[rm]);
    }
}

// Helper to set dim attribute on a vector
static void set_dim_attr(SEXP result, const std::vector<int64_t>& shape) {
    cpp11::writable::integers shape_attr(shape.size());
    for (size_t i = 0; i < shape.size(); i++) {
        shape_attr[i] = static_cast<int>(shape[i]);
    }
    Rf_setAttrib(result, R_DimSymbol, shape_attr);
}

[[cpp11::register]]
SEXP ort_run_(
    SEXP session_ptr,
    cpp11::doubles input_array,
    cpp11::integers input_shape,
    std::string input_name,
    std::string output_name,
    int output_type
) {
    ort_check_loaded();
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

    auto output_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    auto output_shape = output_info.GetShape();
    size_t output_size = 1;
    for (auto dim : output_shape) output_size *= dim;

    // Dispatch on output element type
    switch (output_type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
            float* data = output_tensors[0].GetTensorMutableData<float>();
            cpp11::writable::doubles result(output_size);
            permute_rowcol_to_double(data, REAL(result), output_shape);
            set_dim_attr(result, output_shape);
            return result;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: {
            double* data = output_tensors[0].GetTensorMutableData<double>();
            cpp11::writable::doubles result(output_size);
            permute_rowcol_to_double(data, REAL(result), output_shape);
            set_dim_attr(result, output_shape);
            return result;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
            int32_t* data = output_tensors[0].GetTensorMutableData<int32_t>();
            cpp11::writable::integers result(output_size);
            permute_rowcol_to_int(data, INTEGER(result), output_shape);
            set_dim_attr(result, output_shape);
            return result;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
            int64_t* data = output_tensors[0].GetTensorMutableData<int64_t>();
            cpp11::writable::integers result(output_size);
            permute_rowcol_to_int(data, INTEGER(result), output_shape);
            set_dim_attr(result, output_shape);
            return result;
        }
        default:
            cpp11::stop("Unsupported output tensor type (code %d). "
                        "Supported types: float, double, int32, int64.",
                        output_type);
    }
}
