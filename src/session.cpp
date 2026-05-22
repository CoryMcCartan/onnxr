#include <cpp11.hpp>
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include "ort_loader.h"
#include "onnxruntime/onnxruntime_cxx_api.h"

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
                        int opt_level,
                        cpp11::strings external_data_files)
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

    // Load external data files into memory so providers can access them
    // without resolving relative paths from the model file
    std::vector<std::vector<char>> data_buffers;
    if (external_data_files.size() > 0) {
        std::vector<std::basic_string<ORTCHAR_T>> file_names;
        std::vector<char*> buffer_ptrs;
        std::vector<size_t> buffer_lengths;

        data_buffers.resize(external_data_files.size());
        for (int i = 0; i < external_data_files.size(); i++) {
            std::string fpath(external_data_files[i]);
            // Use basename only — must match the 'location' field in the protobuf
            std::string fname = fpath.substr(fpath.find_last_of("/\\") + 1);
            file_names.push_back(
                std::basic_string<ORTCHAR_T>(fname.begin(), fname.end()));


            std::ifstream ifs(fpath, std::ios::binary | std::ios::ate);
            if (!ifs) {
                cpp11::stop("Failed to read external data file: %s", fpath.c_str());
            }
            size_t fsize = ifs.tellg();
            ifs.seekg(0, std::ios::beg);
            data_buffers[i].resize(fsize);
            ifs.read(data_buffers[i].data(), fsize);

            buffer_ptrs.push_back(data_buffers[i].data());
            buffer_lengths.push_back(fsize);
        }

        session_options.AddExternalInitializersFromFilesInMemory(
            file_names, buffer_ptrs, buffer_lengths);
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

[[cpp11::register]]
cpp11::writable::integers onnx_session_output_types(SEXP session_ptr) {
    onnx_check_loaded();
    cpp11::external_pointer<Ort::Session> session(session_ptr);
    return get_types(session.get(), session->GetOutputCount(), false);
}
