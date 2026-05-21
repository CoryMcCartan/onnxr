#include <cpp11.hpp>
#include <string>
#include <vector>
#include <unordered_map>
#include "ort_loader.h"
#include "onnxruntime/onnxruntime_cxx_api.h"

[[cpp11::register]]
SEXP ort_create_env() {
    ort_check_loaded();
    Ort::Env* env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "nativeORT");
    return cpp11::external_pointer<Ort::Env>(env);
}

[[cpp11::register]]
SEXP ort_create_session(SEXP env_ptr,
                        std::string model_path,
                        std::string provider,
                        std::string cache_dir,
                        int threads,
                        int opt_level)
{
    ort_check_loaded();
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

    if (provider == "coreml") {
        std::unordered_map<std::string, std::string> options;
        options["ModelFormat"] = "MLProgram";
        options["MLComputeUnits"] = "CPUAndNeuralEngine";

        if (!cache_dir.empty()) {
            options["ModelCacheDirectory"] = cache_dir;
        }

        session_options.AppendExecutionProvider("CoreML", options);
    } else if (provider != "cpu") {
        cpp11::warning("Unknown provider/Unsupported GPU support. Falling back to CPU");
    }

    Ort::Session* session = new Ort::Session(
        *env,
        model_path.c_str(),
        session_options
    );

    return cpp11::external_pointer<Ort::Session>(session);
}

[[cpp11::register]]
int ort_session_input_count(SEXP session_ptr) {
    ort_check_loaded();
    cpp11::external_pointer<Ort::Session> session(session_ptr);
    return session->GetInputCount();
}

[[cpp11::register]]
int ort_session_output_count(SEXP session_ptr) {
    ort_check_loaded();
    cpp11::external_pointer<Ort::Session> session(session_ptr);
    return session->GetOutputCount();
}

[[cpp11::register]]
cpp11::writable::strings ort_session_input_names(SEXP session_ptr) {
    ort_check_loaded();
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
cpp11::writable::strings ort_session_output_names(SEXP session_ptr) {
    ort_check_loaded();
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
cpp11::writable::list ort_session_input_shapes(SEXP session_ptr) {
    ort_check_loaded();
    cpp11::external_pointer<Ort::Session> session(session_ptr);
    return get_shapes(session.get(), session->GetInputCount(), true);
}

[[cpp11::register]]
cpp11::writable::list ort_session_output_shapes(SEXP session_ptr) {
    ort_check_loaded();
    cpp11::external_pointer<Ort::Session> session(session_ptr);
    return get_shapes(session.get(), session->GetOutputCount(), false);
}

[[cpp11::register]]
cpp11::writable::integers ort_session_input_types(SEXP session_ptr) {
    ort_check_loaded();
    cpp11::external_pointer<Ort::Session> session(session_ptr);
    return get_types(session.get(), session->GetInputCount(), true);
}

[[cpp11::register]]
cpp11::writable::integers ort_session_output_types(SEXP session_ptr) {
    ort_check_loaded();
    cpp11::external_pointer<Ort::Session> session(session_ptr);
    return get_types(session.get(), session->GetOutputCount(), false);
}
