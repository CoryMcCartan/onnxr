#include <cpp11.hpp>
#include <string>
#include <vector>
#include <set>
#include "onnx_proto.h"

using namespace onnx_proto;

static std::string dtype_to_hlo(DataType t) {
    switch (t) {
        case FLOAT:  return "f32";
        case DOUBLE: return "f64";
        case FLOAT16: return "f16";
        case INT8:   return "i8";
        case INT16:  return "i16";
        case INT32:  return "i32";
        case INT64:  return "i64";
        case UINT8:  return "ui8";
        case UINT16: return "ui16";
        case UINT32: return "ui32";
        case UINT64: return "ui64";
        case BOOL:   return "bool";
        default:     return "unknown";
    }
}

// Convert a TensorProto to an R numeric/integer array with dim + dtype attrs
static SEXP tensor_to_r(TensorProto& tp, const std::string& model_dir) {
    // Load external data if needed
    if (!tp.external_location.empty()) {
        load_external_data(tp, model_dir);
    }

    size_t n = 1;
    for (auto d : tp.dims) n *= d;

    if (n == 0) {
        // Empty tensor — return NULL with dtype attribute
        SEXP result = R_NilValue;
        return result;
    }

    SEXP result;
    std::string hlo_dtype = dtype_to_hlo(tp.data_type);

    // Check if tensor has any data
    bool has_data = !tp.raw_data.empty() || !tp.float_data.empty() ||
                    !tp.double_data.empty() || !tp.int32_data.empty() ||
                    !tp.int64_data.empty();
    if (!has_data) {
        // No inline data — might need to skip (external data loaded separately)
        cpp11::writable::doubles r(static_cast<R_xlen_t>(0));
        Rf_setAttrib(r, Rf_install("dtype"),
            cpp11::writable::strings({dtype_to_hlo(tp.data_type)}));
        return r;
    }

    if (tp.data_type == FLOAT) {
        cpp11::writable::doubles r(n);
        if (!tp.raw_data.empty()) {
            const float* data = reinterpret_cast<const float*>(tp.raw_data.data());
            for (size_t i = 0; i < n; i++) r[i] = static_cast<double>(data[i]);
        } else {
            for (size_t i = 0; i < n; i++) r[i] = static_cast<double>(tp.float_data[i]);
        }
        result = r;
    } else if (tp.data_type == DOUBLE) {
        cpp11::writable::doubles r(n);
        if (!tp.raw_data.empty()) {
            const double* data = reinterpret_cast<const double*>(tp.raw_data.data());
            for (size_t i = 0; i < n; i++) r[i] = data[i];
        } else {
            for (size_t i = 0; i < n; i++) r[i] = tp.double_data[i];
        }
        result = r;
    } else if (tp.data_type == INT32) {
        cpp11::writable::integers r(n);
        if (!tp.raw_data.empty()) {
            const int32_t* data = reinterpret_cast<const int32_t*>(tp.raw_data.data());
            for (size_t i = 0; i < n; i++) r[i] = data[i];
        } else {
            for (size_t i = 0; i < n; i++) r[i] = tp.int32_data[i];
        }
        result = r;
    } else if (tp.data_type == INT64) {
        cpp11::writable::doubles r(n);
        if (!tp.raw_data.empty()) {
            const int64_t* data = reinterpret_cast<const int64_t*>(tp.raw_data.data());
            for (size_t i = 0; i < n; i++) r[i] = static_cast<double>(data[i]);
        } else {
            for (size_t i = 0; i < n; i++) r[i] = static_cast<double>(tp.int64_data[i]);
        }
        result = r;
    } else {
        // Store raw bytes for other types
        cpp11::writable::raws r(tp.raw_data.size());
        memcpy(RAW(r), tp.raw_data.data(), tp.raw_data.size());
        result = r;
    }

    if (result != R_NilValue) {
        if (!tp.dims.empty()) {
            cpp11::writable::integers dim_attr(tp.dims.size());
            for (size_t i = 0; i < tp.dims.size(); i++)
                dim_attr[i] = static_cast<int>(tp.dims[i]);
            Rf_setAttrib(result, R_DimSymbol, dim_attr);
        }
        Rf_setAttrib(result, Rf_install("dtype"),
            cpp11::writable::strings({hlo_dtype}));
    }

    return result;
}

static SEXP attr_to_r(const Attribute& attr) {
    // Infer type from populated fields if type field is unset
    AttrType t = attr.type;
    if (t == ATTR_UNDEFINED) {
        if (!attr.floats.empty()) t = ATTR_FLOATS;
        else if (!attr.ints.empty()) t = ATTR_INTS;
        else if (!attr.strings.empty()) t = ATTR_STRINGS;
        else if (!attr.s.empty()) t = ATTR_STRING;
        else if (attr.f != 0) t = ATTR_FLOAT;
        else if (attr.i != 0) t = ATTR_INT;
    }

    switch (t) {
        case ATTR_FLOAT:
            return cpp11::writable::doubles({static_cast<double>(attr.f)});
        case ATTR_INT:
            return cpp11::writable::integers({static_cast<int>(attr.i)});
        case ATTR_STRING:
            return cpp11::writable::strings({attr.s});
        case ATTR_FLOATS: {
            cpp11::writable::doubles r(attr.floats.size());
            for (size_t i = 0; i < attr.floats.size(); i++)
                r[i] = static_cast<double>(attr.floats[i]);
            return r;
        }
        case ATTR_INTS: {
            cpp11::writable::integers r(attr.ints.size());
            for (size_t i = 0; i < attr.ints.size(); i++)
                r[i] = static_cast<int>(attr.ints[i]);
            return r;
        }
        case ATTR_STRINGS: {
            cpp11::writable::strings r;
            for (auto& s : attr.strings) r.push_back(s);
            return r;
        }
        default:
            return cpp11::writable::strings({"<unsupported>"});
    }
}

[[cpp11::register]]
cpp11::writable::list ort_read_graph(std::string model_path) {
    using namespace cpp11::literals;

    onnx_proto::ModelProto model;
    try {
        model = onnx_proto::load(model_path);
    } catch (const std::exception& e) {
        cpp11::stop("Failed to parse ONNX file: %s", e.what());
    }
    auto& graph = model.graph;
    std::string model_dir = model_path.substr(0, model_path.find_last_of("/\\"));

    // Build set of initializer names
    std::set<std::string> init_names_set;
    for (auto& init : graph.initializers) {
        init_names_set.insert(init.name);
    }

    // -- Inputs (skip initializers) --
    cpp11::writable::list r_inputs;
    for (auto& vi : graph.inputs) {
        if (init_names_set.count(vi.name)) continue;
        cpp11::writable::list r_vi;
        r_vi.push_back({"name"_nm = cpp11::writable::strings({vi.name})});
        r_vi.push_back({"dtype"_nm = cpp11::writable::strings({dtype_to_hlo(vi.type.elem_type)})});
        cpp11::writable::integers r_shape(vi.type.shape.size());
        for (size_t j = 0; j < vi.type.shape.size(); j++)
            r_shape[j] = static_cast<int>(vi.type.shape[j].value);
        r_vi.push_back({"shape"_nm = r_shape});
        r_inputs.push_back(r_vi);
    }

    // -- Outputs --
    cpp11::writable::list r_outputs(graph.outputs.size());
    for (size_t i = 0; i < graph.outputs.size(); i++) {
        auto& vi = graph.outputs[i];
        cpp11::writable::list r_vi;
        r_vi.push_back({"name"_nm = cpp11::writable::strings({vi.name})});
        r_vi.push_back({"dtype"_nm = cpp11::writable::strings({dtype_to_hlo(vi.type.elem_type)})});
        cpp11::writable::integers r_shape(vi.type.shape.size());
        for (size_t j = 0; j < vi.type.shape.size(); j++)
            r_shape[j] = static_cast<int>(vi.type.shape[j].value);
        r_vi.push_back({"shape"_nm = r_shape});
        r_outputs[i] = r_vi;
    }

    // -- Initializers --
    cpp11::writable::list r_inits;
    cpp11::writable::strings r_init_names;
    for (auto& init : graph.initializers) {
        r_init_names.push_back(init.name);
        r_inits.push_back(tensor_to_r(init, model_dir));
    }
    if (r_init_names.size() > 0) {
        r_inits.names() = r_init_names;
    }

    // -- Nodes --
    cpp11::writable::list r_nodes(graph.nodes.size());
    for (size_t i = 0; i < graph.nodes.size(); i++) {
        auto& node = graph.nodes[i];
        cpp11::writable::list r_node;

        r_node.push_back({"op_type"_nm = cpp11::writable::strings({node.op_type})});
        r_node.push_back({"name"_nm = cpp11::writable::strings({node.name})});

        cpp11::writable::strings r_inputs_v;
        for (auto& s : node.inputs) r_inputs_v.push_back(s);
        r_node.push_back({"inputs"_nm = r_inputs_v});

        cpp11::writable::strings r_outputs_v;
        for (auto& s : node.outputs) r_outputs_v.push_back(s);
        r_node.push_back({"outputs"_nm = r_outputs_v});

        cpp11::writable::list r_attrs;
        cpp11::writable::strings r_attr_names;
        for (auto& attr : node.attributes) {
            r_attr_names.push_back(attr.name);
            r_attrs.push_back(attr_to_r(attr));
        }
        if (r_attr_names.size() > 0) {
            r_attrs.names() = r_attr_names;
        }
        r_node.push_back({"attrs"_nm = r_attrs});

        r_nodes[i] = r_node;
    }

    // -- Assemble --
    cpp11::writable::list result;
    result.push_back({"inputs"_nm = r_inputs});
    result.push_back({"outputs"_nm = r_outputs});
    result.push_back({"initializers"_nm = r_inits});
    result.push_back({"nodes"_nm = r_nodes});

    return result;
}
