#include <cpp11.hpp>
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include "ort_loader.h"
#include "onnxruntime/onnxruntime_cxx_api.h"

// Scan an .onnx file (a protobuf) for the locations of external data files.
// Each external initializer encodes its file path as a StringStringEntryProto
// nested inside TensorProto.external_data with key="location". On the wire
// this is the fixed 10-byte sequence:
//   0x0a 0x08 'l' 'o' 'c' 'a' 't' 'i' 'o' 'n'   (field 1, len 8 string "location")
//   0x12 <varint length> <value bytes>          (field 2, the location string)
// Scanning for that fixed prefix is robust without linking protobuf.
static std::vector<std::string> find_external_data_locations(
    const std::vector<char>& bytes
) {
    static const unsigned char prefix[] = {
        0x0a, 0x08, 'l', 'o', 'c', 'a', 't', 'i', 'o', 'n', 0x12
    };
    const size_t plen = sizeof(prefix);
    const size_t n = bytes.size();
    std::vector<std::string> locations;
    size_t i = 0;
    while (i + plen < n) {
        if (static_cast<unsigned char>(bytes[i]) == prefix[0] &&
            std::memcmp(bytes.data() + i, prefix, plen) == 0) {
            // Decode varint for value length.
            size_t j = i + plen;
            uint64_t len = 0;
            int shift = 0;
            while (j < n && shift < 35) {
                unsigned char b = static_cast<unsigned char>(bytes[j++]);
                len |= static_cast<uint64_t>(b & 0x7f) << shift;
                if ((b & 0x80) == 0) break;
                shift += 7;
            }
            if (len > 0 && len < 1024 && j + len <= n) {
                locations.emplace_back(bytes.data() + j, len);
            }
            i = j + len;
        } else {
            i++;
        }
    }
    // De-duplicate while preserving order.
    std::vector<std::string> unique;
    for (const auto& s : locations) {
        if (std::find(unique.begin(), unique.end(), s) == unique.end()) {
            unique.push_back(s);
        }
    }
    return unique;
}

[[cpp11::register]]
SEXP onnx_create_env() {
    onnx_check_loaded();
    Ort::Env* env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "onnxr");
    return cpp11::external_pointer<Ort::Env>(env);
}

[[cpp11::register]]
SEXP onnx_create_session(SEXP env_ptr,
                        std::string model_path,
                        std::string provider,
                        std::string cache_dir,
                        int threads,
                        int opt_level)
{
    onnx_check_loaded();
    cpp11::external_pointer<Ort::Env> env(env_ptr);
    Ort::SessionOptions session_options;

    if (threads > 0){
        session_options.SetIntraOpNumThreads(threads);
        session_options.SetInterOpNumThreads(threads);
    }

    GraphOptimizationLevel ort_opt;
    switch(opt_level) {
        case 0: ort_opt = GraphOptimizationLevel::ORT_DISABLE_ALL; break;
        case 1: ort_opt = GraphOptimizationLevel::ORT_ENABLE_BASIC; break;
        default: ort_opt = GraphOptimizationLevel::ORT_ENABLE_ALL; break;
    }
    session_options.SetGraphOptimizationLevel(ort_opt);

    // Discover and pre-load external data files referenced by the model
    // so that non-CPU backends (especially CoreML) can access them without
    // resolving relative paths at session-construction time.
    std::vector<std::vector<char>> data_buffers;
    {
        std::ifstream model_ifs(model_path, std::ios::binary | std::ios::ate);
        if (!model_ifs) {
            cpp11::stop("Failed to read model file: %s", model_path.c_str());
        }
        size_t msize = model_ifs.tellg();
        model_ifs.seekg(0, std::ios::beg);
        std::vector<char> model_bytes(msize);
        model_ifs.read(model_bytes.data(), msize);

        std::vector<std::string> locations =
            find_external_data_locations(model_bytes);

        if (!locations.empty()) {
            std::string model_dir;
            size_t slash = model_path.find_last_of("/\\");
            if (slash != std::string::npos) {
                model_dir = model_path.substr(0, slash + 1);
            }

            std::vector<std::basic_string<ORTCHAR_T>> file_names;
            std::vector<char*> buffer_ptrs;
            std::vector<size_t> buffer_lengths;
            data_buffers.resize(locations.size());

            for (size_t i = 0; i < locations.size(); i++) {
                const std::string& loc = locations[i];
                std::string fpath = model_dir + loc;
                std::ifstream ifs(fpath, std::ios::binary | std::ios::ate);
                if (!ifs) {
                    cpp11::stop(
                        "Model references external data file not found: %s",
                        fpath.c_str());
                }
                size_t fsize = ifs.tellg();
                ifs.seekg(0, std::ios::beg);
                data_buffers[i].resize(fsize);
                ifs.read(data_buffers[i].data(), fsize);

                // Key passed to ORT must match the 'location' string verbatim
                file_names.emplace_back(loc.begin(), loc.end());
                buffer_ptrs.push_back(data_buffers[i].data());
                buffer_lengths.push_back(fsize);
            }

            session_options.AddExternalInitializersFromFilesInMemory(
                file_names, buffer_ptrs, buffer_lengths);
        }
    }

    if (provider != "cpu") {
        std::unordered_map<std::string, std::string> options;

        // Provider-specific default options
        if (provider == "coreml") {
            options["ModelFormat"] = "MLProgram";
            options["MLComputeUnits"] = "CPUAndNeuralEngine";
            if (!cache_dir.empty()) {
                options["ModelCacheDirectory"] = cache_dir;
            }
        }

        // Map short names to ORT provider names
        std::string ort_name;
        if (provider == "coreml") ort_name = "CoreML";
        else if (provider == "cuda") ort_name = "CUDA";
        else if (provider == "xnnpack") ort_name = "XNNPACK";
        else if (provider == "openvino") ort_name = "OpenVINO";
        else ort_name = provider;

        session_options.AppendExecutionProvider(ort_name, options);
    }

    std::basic_string<ORTCHAR_T> ort_path(model_path.begin(), model_path.end());
    Ort::Session* session = new Ort::Session(
        *env,
        ort_path.c_str(),
        session_options
    );

    return cpp11::external_pointer<Ort::Session>(session);
}

[[cpp11::register]]
int onnx_session_input_count(SEXP session_ptr) {
    onnx_check_loaded();
    cpp11::external_pointer<Ort::Session> session(session_ptr);
    return session->GetInputCount();
}

[[cpp11::register]]
int onnx_session_output_count(SEXP session_ptr) {
    onnx_check_loaded();
    cpp11::external_pointer<Ort::Session> session(session_ptr);
    return session->GetOutputCount();
}

[[cpp11::register]]
cpp11::writable::strings onnx_session_input_names(SEXP session_ptr) {
    onnx_check_loaded();
    cpp11::external_pointer<Ort::Session> session(session_ptr);
    Ort::AllocatorWithDefaultOptions allocator;

    size_t count = session->GetInputCount();
    cpp11::writable::strings names;
    names.reserve(count);

    for (size_t i = 0; i < count; i++) {
        auto name = session->GetInputNameAllocated(i, allocator);
        names.push_back(std::string(name.get()));
    }

    return names;
}

[[cpp11::register]]
cpp11::writable::strings onnx_session_output_names(SEXP session_ptr) {
    onnx_check_loaded();
    cpp11::external_pointer<Ort::Session> session(session_ptr);
    Ort::AllocatorWithDefaultOptions allocator;

    size_t count = session->GetOutputCount();
    cpp11::writable::strings names;
    names.reserve(count);

    for (size_t i = 0; i < count; i++) {
        auto name = session->GetOutputNameAllocated(i, allocator);
        names.push_back(std::string(name.get()));
    }
    return names;
}

// Helper to extract shapes from TypeInfo objects
static cpp11::writable::list get_shapes(Ort::Session* session, size_t count, bool is_input) {
    cpp11::writable::list shapes(count);
    for (size_t i = 0; i < count; i++) {
        auto type_info = is_input ? session->GetInputTypeInfo(i)
                                  : session->GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();
        cpp11::writable::integers r_shape(shape.size());
        for (size_t j = 0; j < shape.size(); j++) {
            r_shape[j] = static_cast<int>(shape[j]); // -1 for dynamic dims
        }
        shapes[i] = r_shape;
    }
    return shapes;
}

// Helper to extract element types
static cpp11::writable::integers get_types(Ort::Session* session, size_t count, bool is_input) {
    cpp11::writable::integers types(count);
    for (size_t i = 0; i < count; i++) {
        auto type_info = is_input ? session->GetInputTypeInfo(i)
                                  : session->GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        types[i] = static_cast<int>(tensor_info.GetElementType());
    }
    return types;
}

[[cpp11::register]]
cpp11::writable::list onnx_session_input_shapes(SEXP session_ptr) {
    onnx_check_loaded();
    cpp11::external_pointer<Ort::Session> session(session_ptr);
    return get_shapes(session.get(), session->GetInputCount(), true);
}

[[cpp11::register]]
cpp11::writable::list onnx_session_output_shapes(SEXP session_ptr) {
    onnx_check_loaded();
    cpp11::external_pointer<Ort::Session> session(session_ptr);
    return get_shapes(session.get(), session->GetOutputCount(), false);
}

[[cpp11::register]]
cpp11::writable::integers onnx_session_input_types(SEXP session_ptr) {
    onnx_check_loaded();
    cpp11::external_pointer<Ort::Session> session(session_ptr);
    return get_types(session.get(), session->GetInputCount(), true);
}

// Logical vector, one per input, TRUE if the input is marked as an
// optional graph input (ORT lets these be omitted at Run time).
[[cpp11::register]]
cpp11::writable::logicals onnx_session_input_optional(SEXP session_ptr) {
    onnx_check_loaded();
    cpp11::external_pointer<Ort::Session> session(session_ptr);
    size_t count = session->GetInputCount();
    cpp11::writable::logicals out(count);
    // IsOptionalGraphInput() is only supported on some session types in
    // current ORT builds. If unavailable, fall back to treating every input
    // as required (conservative; matches pre-existing behavior).
    try {
        auto infos = session->GetInputs();
        for (size_t i = 0; i < infos.size(); i++) {
            out[i] = infos[i].IsOptionalGraphInput();
        }
    } catch (const Ort::Exception&) {
        for (size_t i = 0; i < count; i++) out[i] = FALSE;
    }
    return out;
}

[[cpp11::register]]
cpp11::writable::integers onnx_session_output_types(SEXP session_ptr) {
    onnx_check_loaded();
    cpp11::external_pointer<Ort::Session> session(session_ptr);
    return get_types(session.get(), session->GetOutputCount(), false);
}
