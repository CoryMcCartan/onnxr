#include <cpp11.hpp>
#include <string>
#include <vector>
#include <unordered_map>

#ifdef HAVE_ORT
#include <onnxruntime_cxx_api.h>
#endif

[[cpp11::register]]
SEXP ort_create_env() {
#ifdef HAVE_ORT
  Ort::Env* env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "nativeORT");
  return cpp11::external_pointer<Ort::Env>(env);
#else
  cpp11::stop("ORT not installed. Run ort_install() then reinstall nativeORT.");
#endif
}

[[cpp11::register]]
SEXP ort_create_session(SEXP env_ptr,
                        std::string model_path,
                        std::string provider,
                        std::string cache_dir,
                        int threads,
                        int opt_level)
{
#ifdef HAVE_ORT
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
#else
  cpp11::stop("ORT not installed. Run ort_install() then reinstall nativeORT.");
#endif
}

[[cpp11::register]]
int ort_session_input_count(SEXP session_ptr) {
#ifdef HAVE_ORT
  cpp11::external_pointer<Ort::Session> session(session_ptr);
  return session->GetInputCount();
#else
  cpp11::stop("ORT not installed. Run ort_install() then reinstall nativeORT.");
#endif
}

[[cpp11::register]]
int ort_session_output_count(SEXP session_ptr) {
#ifdef HAVE_ORT
  cpp11::external_pointer<Ort::Session> session(session_ptr);
  return session->GetOutputCount();
#else
  cpp11::stop("ORT not installed. Run ort_install() then reinstall nativeORT.");
#endif
}

[[cpp11::register]]
cpp11::writable::strings ort_session_input_names(SEXP session_ptr) {
#ifdef HAVE_ORT
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
#else
  cpp11::stop("ORT not installed. Run ort_install() then reinstall nativeORT.");
#endif
}

[[cpp11::register]]
cpp11::writable::strings ort_session_output_names(SEXP session_ptr) {
#ifdef HAVE_ORT
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
#else
  cpp11::stop("ORT not installed. Run ort_install() then reinstall nativeORT.");
#endif
}
