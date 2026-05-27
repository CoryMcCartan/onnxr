#include <cpp11.hpp>
#include <vector>
#include <string>
#include "ort_loader.h"
#include "onnxruntime/onnxruntime_cxx_api.h"

// ---- Permutation helpers ----
// R stores arrays column-major, ONNX expects row-major.
// These functions convert between the two layouts while also casting types.

// Compute strides for row-major and column-major layouts.
// Caller must ensure ndim >= 2.
static void compute_strides(const std::vector<int64_t>& shape,
                            std::vector<size_t>& rm_stride,
                            std::vector<size_t>& cm_stride) {
    size_t ndim = shape.size();
    rm_stride.resize(ndim);
    cm_stride.resize(ndim);
    rm_stride[ndim - 1] = 1;
    for (size_t k = ndim - 1; k > 0; k--)
        rm_stride[k - 1] = rm_stride[k] * shape[k];
    cm_stride[0] = 1;
    for (size_t k = 1; k < ndim; k++)
        cm_stride[k] = cm_stride[k - 1] * shape[k - 1];
}

// Column-major src → row-major dst (for inputs: R → ORT)
template <typename Src, typename Dst>
static void permute_colrow(const Src* src, Dst* dst,
                           const std::vector<int64_t>& shape) {
    size_t ndim = shape.size();
    size_t n = 1;
    for (auto d : shape) n *= d;

    if (ndim <= 1) {
        for (size_t i = 0; i < n; i++) dst[i] = static_cast<Dst>(src[i]);
        return;
    }

    std::vector<size_t> rm_stride, cm_stride;
    compute_strides(shape, rm_stride, cm_stride);

    for (size_t rm = 0; rm < n; rm++) {
        size_t tmp = rm, cm = 0;
        for (size_t k = 0; k < ndim; k++) {
            size_t idx = tmp / rm_stride[k];
            tmp %= rm_stride[k];
            cm += idx * cm_stride[k];
        }
        dst[rm] = static_cast<Dst>(src[cm]);
    }
}

// Row-major src → column-major dst (for outputs: ORT → R)
template <typename Src, typename Dst>
static void permute_rowcol(const Src* src, Dst* dst,
                           const std::vector<int64_t>& shape) {
    size_t ndim = shape.size();
    size_t n = 1;
    for (auto d : shape) n *= d;

    if (ndim <= 1) {
        for (size_t i = 0; i < n; i++) dst[i] = static_cast<Dst>(src[i]);
        return;
    }

    std::vector<size_t> rm_stride, cm_stride;
    compute_strides(shape, rm_stride, cm_stride);

    for (size_t cm = 0; cm < n; cm++) {
        size_t tmp = cm, rm = 0;
        for (size_t k = ndim; k > 0; k--) {
            size_t idx = tmp / cm_stride[k - 1];
            tmp %= cm_stride[k - 1];
            rm += idx * rm_stride[k - 1];
        }
        dst[cm] = static_cast<Dst>(src[rm]);
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

// Check whether an R integer vector contains any NA values
static bool has_int_na(SEXP x) {
    int* p = INTEGER(x);
    R_xlen_t len = Rf_xlength(x);
    for (R_xlen_t i = 0; i < len; i++) {
        if (p[i] == NA_INTEGER) return true;
    }
    return false;
}

// Check whether an R logical vector contains any NA values
// (LGLSXP stores values as int, with NA_LOGICAL == NA_INTEGER)
static bool has_lgl_na(SEXP x) {
    int* p = LOGICAL(x);
    R_xlen_t len = Rf_xlength(x);
    for (R_xlen_t i = 0; i < len; i++) {
        if (p[i] == NA_LOGICAL) return true;
    }
    return false;
}

// ---- Input tensor creation ----
// Create an ORT input tensor from an R vector, dispatching on input_type.
// Manages its own buffer lifetime via the out_buf pointer.

static Ort::Value make_input_tensor(
    SEXP input_data, const std::vector<int64_t>& shape,
    int input_type, Ort::MemoryInfo& memory_info,
    std::vector<float>& buf_f, std::vector<double>& buf_d,
    std::vector<int32_t>& buf_i32, std::vector<int64_t>& buf_i64,
    std::vector<uint8_t>& buf_b
) {
    size_t n = 1;
    for (auto d : shape) n *= d;

    // Validate input length matches shape
    R_xlen_t input_len = Rf_xlength(input_data);
    if (static_cast<size_t>(input_len) != n) {
        cpp11::stop("Input has %d elements but shape requires %d.",
                     static_cast<int>(input_len), static_cast<int>(n));
    }

    // Check for NAs in integer/logical input
    if (TYPEOF(input_data) == INTSXP && has_int_na(input_data)) {
        cpp11::stop("NA values are not supported in ONNX model inputs.");
    }
    if (TYPEOF(input_data) == LGLSXP && has_lgl_na(input_data)) {
        cpp11::stop("NA values are not supported in ONNX model inputs.");
    }

    switch (input_type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
            buf_f.resize(n);
            if (TYPEOF(input_data) == REALSXP) {
                permute_colrow(REAL(input_data), buf_f.data(), shape);
            } else if (TYPEOF(input_data) == INTSXP) {
                permute_colrow(INTEGER(input_data), buf_f.data(), shape);
            } else {
                cpp11::stop("Input must be numeric or integer.");
            }
            return Ort::Value::CreateTensor<float>(
                memory_info, buf_f.data(), n, shape.data(), shape.size());
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: {
            buf_d.resize(n);
            if (TYPEOF(input_data) == REALSXP) {
                permute_colrow(REAL(input_data), buf_d.data(), shape);
            } else if (TYPEOF(input_data) == INTSXP) {
                permute_colrow(INTEGER(input_data), buf_d.data(), shape);
            } else {
                cpp11::stop("Input must be numeric or integer.");
            }
            return Ort::Value::CreateTensor<double>(
                memory_info, buf_d.data(), n, shape.data(), shape.size());
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
            buf_i32.resize(n);
            if (TYPEOF(input_data) == INTSXP) {
                permute_colrow(INTEGER(input_data), buf_i32.data(), shape);
            } else if (TYPEOF(input_data) == REALSXP) {
                permute_colrow(REAL(input_data), buf_i32.data(), shape);
            } else {
                cpp11::stop("Input must be numeric or integer.");
            }
            return Ort::Value::CreateTensor<int32_t>(
                memory_info, buf_i32.data(), n, shape.data(), shape.size());
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
            buf_i64.resize(n);
            if (TYPEOF(input_data) == INTSXP) {
                permute_colrow(INTEGER(input_data), buf_i64.data(), shape);
            } else if (TYPEOF(input_data) == REALSXP) {
                permute_colrow(REAL(input_data), buf_i64.data(), shape);
            } else {
                cpp11::stop("Input must be numeric or integer.");
            }
            return Ort::Value::CreateTensor<int64_t>(
                memory_info, buf_i64.data(), n, shape.data(), shape.size());
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: {
            buf_b.resize(n);
            if (TYPEOF(input_data) == LGLSXP) {
                permute_colrow(LOGICAL(input_data), buf_b.data(), shape);
            } else if (TYPEOF(input_data) == INTSXP) {
                permute_colrow(INTEGER(input_data), buf_b.data(), shape);
            } else {
                cpp11::stop("Input must be logical or integer.");
            }
            return Ort::Value::CreateTensor(
                memory_info, buf_b.data(), n,
                shape.data(), shape.size(),
                ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);
        }
        default:
            cpp11::stop("Unsupported input tensor type (code %d). "
                        "Supported: float, double, int32, int64, bool.", input_type);
    }
}

// ---- Output tensor reading ----

static SEXP read_output_tensor(Ort::Value& tensor) {
    auto info = tensor.GetTensorTypeAndShapeInfo();
    auto shape = info.GetShape();
    auto etype = info.GetElementType();
    size_t n = 1;
    for (auto d : shape) n *= d;

    switch (etype) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
            float* data = tensor.GetTensorMutableData<float>();
            cpp11::writable::doubles result(n);
            permute_rowcol(data, REAL(result), shape);
            set_dim_attr(result, shape);
            return result;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: {
            double* data = tensor.GetTensorMutableData<double>();
            cpp11::writable::doubles result(n);
            permute_rowcol(data, REAL(result), shape);
            set_dim_attr(result, shape);
            return result;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
            int32_t* data = tensor.GetTensorMutableData<int32_t>();
            cpp11::writable::integers result(n);
            permute_rowcol(data, INTEGER(result), shape);
            set_dim_attr(result, shape);
            return result;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
            // R has no native int64; return as doubles (exact up to 2^53)
            int64_t* data = tensor.GetTensorMutableData<int64_t>();
            cpp11::writable::doubles result(n);
            permute_rowcol(data, REAL(result), shape);
            set_dim_attr(result, shape);
            return result;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: {
            uint8_t* data = tensor.GetTensorMutableData<uint8_t>();
            cpp11::writable::logicals result(n);
            permute_rowcol(data, LOGICAL(result), shape);
            set_dim_attr(result, shape);
            return result;
        }
        default:
            cpp11::stop("Unsupported output tensor type (code %d). "
                        "Supported: float, double, int32, int64, bool.",
                        static_cast<int>(etype));
    }
}

// ---- Main inference function ----

[[cpp11::register]]
cpp11::writable::list onnx_run_(
    SEXP session_ptr,
    cpp11::list inputs,
    cpp11::list input_shapes,
    cpp11::strings input_names,
    cpp11::integers input_types,
    cpp11::strings output_names
) {
    onnx_check_loaded();
    cpp11::external_pointer<Ort::Session> session(session_ptr);

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    size_t n_inputs = inputs.size();
    size_t n_outputs = output_names.size();

    // Build input tensors
    // Buffers must outlive the session->Run() call
    std::vector<std::vector<float>> bufs_f(n_inputs);
    std::vector<std::vector<double>> bufs_d(n_inputs);
    std::vector<std::vector<int32_t>> bufs_i32(n_inputs);
    std::vector<std::vector<int64_t>> bufs_i64(n_inputs);
    std::vector<std::vector<uint8_t>> bufs_b(n_inputs);

    std::vector<Ort::Value> input_tensors;
    input_tensors.reserve(n_inputs);
    std::vector<const char*> c_input_names(n_inputs);

    // We need to keep the std::string objects alive for c_str() pointers
    std::vector<std::string> input_name_strs(n_inputs);

    for (size_t i = 0; i < n_inputs; i++) {
        SEXP input_data = inputs[i];
        cpp11::integers shp(input_shapes[i]);
        std::vector<int64_t> shape64(shp.begin(), shp.end());

        input_name_strs[i] = std::string(input_names[i]);
        c_input_names[i] = input_name_strs[i].c_str();

        input_tensors.push_back(
            make_input_tensor(input_data, shape64, input_types[i],
                              memory_info, bufs_f[i], bufs_d[i],
                              bufs_i32[i], bufs_i64[i], bufs_b[i]));
    }

    // Build output names
    std::vector<std::string> output_name_strs(n_outputs);
    std::vector<const char*> c_output_names(n_outputs);
    for (size_t i = 0; i < n_outputs; i++) {
        output_name_strs[i] = std::string(output_names[i]);
        c_output_names[i] = output_name_strs[i].c_str();
    }

    // Run
    auto output_tensors = session->Run(
        Ort::RunOptions{nullptr},
        c_input_names.data(),
        input_tensors.data(),
        n_inputs,
        c_output_names.data(),
        n_outputs
    );

    // Read outputs
    cpp11::writable::list results(n_outputs);
    cpp11::writable::strings result_names(n_outputs);
    for (size_t i = 0; i < n_outputs; i++) {
        results[i] = read_output_tensor(output_tensors[i]);
        result_names[i] = output_names[i];
    }
    results.names() = result_names;

    return results;
}
