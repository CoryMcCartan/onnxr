#include <Rcpp.h>

#ifdef HAVE_ORT
#include <onnxruntime_cxx_api.h>

// [[Rcpp::export]]
SEXP ort_create_env() {
  Ort::Env* env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "nativeORT");
  Rcpp::XPtr<Ort::Env> ptr(env, true);
  return ptr;
}

// [[Rcpp::export]]
SEXP ort_create_session(SEXP env_ptr,
                        std::string model_path,
                        std::string provider = "cpu",
                        std::string cache_dir = "",
                        int threads = 0,
                        int opt_level=99)
{
  Rcpp::XPtr<Ort::Env> env(env_ptr);
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

  // add in CoreML options for apple silicon users
  if (provider == "coreml") {
    std::unordered_map<std::string, std::string> options;
    options["ModelFormat"] = "MLProgram";
    options["MLComputeUnits"] = "CPUAndNeuralEngine";

    if (!cache_dir.empty()) {
      options["ModelCacheDirectory"] = cache_dir;
    }

    session_options.AppendExecutionProvider("CoreML", options);
  } else if (provider != "cpu") {
    Rcpp::warning("Unknown provider/Unsupported GPU support. Falling back to CPU");
  }

  Ort::Session* session = new Ort::Session(
    *env,
    model_path.c_str(),
    session_options
  );

  Rcpp::XPtr<Ort::Session> ptr(session, true);
  return ptr;
}

// [[Rcpp::export]]
int ort_session_input_count(SEXP session_ptr) {
  Rcpp::XPtr<Ort::Session> session(session_ptr);
  return session->GetInputCount();
}

// [[Rcpp::export]]
int ort_session_output_count(SEXP session_ptr) {
  Rcpp::XPtr<Ort::Session> session(session_ptr);
  return session->GetOutputCount();
}

// [[Rcpp::export]]
std::vector<std::string> ort_session_input_names(SEXP session_ptr) {
  Rcpp::XPtr<Ort::Session> session(session_ptr);
  Ort::AllocatorWithDefaultOptions allocator;

  size_t count = session->GetInputCount();
  std::vector<std::string> names;
  names.reserve(count);

  for (size_t i = 0; i < count; i++) {
    auto name = session->GetInputNameAllocated(i, allocator);
    names.push_back(std::string(name.get()));
  }

  return names;
}

// [[Rcpp::export]]
std::vector<std::string> ort_session_output_names(SEXP session_ptr) {
  Rcpp::XPtr<Ort::Session> session(session_ptr);
  Ort::AllocatorWithDefaultOptions allocator;

  size_t count = session->GetOutputCount();
  std::vector<std::string> names;
  names.reserve(count);

  for (size_t i = 0; i < count; i++) {
    auto name = session->GetOutputNameAllocated(i, allocator);
    names.push_back(std::string(name.get()));
  }
  return names;
}

#else

SEXP ort_create_env() {
  Rcpp::stop("ORT not installed. Run ort_install() then reinstall nativeORT.");
}

SEXP ort_create_session(SEXP env_ptr,
                        std::string model_path,
                        std::string provider = "cpu",
                        std::string cache_dir = "",
                        int threads = 0,
                        int opt_level=99) {
  Rcpp::stop("ORT not installed. Run ort_install() then reinstall nativeORT.");
}

int ort_session_input_count(SEXP session_ptr) {
  Rcpp::stop("ORT not installed. Run ort_install() then reinstall nativeORT.");
}

int ort_session_output_count(SEXP session_ptr) {
  Rcpp::stop("ORT not installed. Run ort_install() then reinstall nativeORT.");
}

std::vector<std::string> ort_session_input_names(SEXP session_ptr) {
  Rcpp::stop("ORT not installed. Run ort_install() then reinstall nativeORT.");
}

std::vector<std::string> ort_session_output_names(SEXP session_ptr) {
  Rcpp::stop("ORT not installed. Run ort_install() then reinstall nativeORT.");
}

#endif
