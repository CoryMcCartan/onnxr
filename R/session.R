#' Create an ONNX Runtime inference session
#'
#' Loads an `.onnx` model file and creates a session for running inference.
#'
#' @param path Path to an `.onnx` model file.
#' @param provider Execution provider: `"cpu"` (default) or `"coreml"`
#'   (Apple Silicon only).
#' @param cache_dir Optional directory for CoreML model cache.
#' @param threads Number of threads. `0` (default) uses all available;
#'   a positive integer sets a fixed thread count.
#' @param opt_level Graph optimization level. `99` (default) enables all
#'   optimizations; `1` for basic only; `0` to disable.
#'
#' @returns An `"ort_session"` object (a named list) with model metadata
#'   and internal pointers used by [ort_infer_raw()].
#' @export
#'
#' @examples \dontrun{
#' sess <- ort_session("model.onnx")
#' sess
#' }
ort_session <- function(
    path,
    provider = "cpu",
    cache_dir = NULL,
    threads = 0L,
    opt_level = 99L
) {
    if (!file.exists(path)) {
        stop("Model file not found")
    }
    if (!grepl("\\.onnx$", path, ignore.case = TRUE)) {
        stop("File must be an .onnx model")
    }
    if (provider == "coreml" && .Platform$OS.type != "unix") {
        warning("CoreML is only available on macOS, falling back to CPU")
        provider <- "cpu"
    }
    cache <- if (!is.null(cache_dir)) {
        normalizePath(cache_dir, mustWork = FALSE)
    } else {
        ""
    }

    env <- ort_create_env()
    sess <- ort_create_session(
        env_ptr = env,
        model_path = normalizePath(path),
        provider = provider,
        cache_dir = cache,
        threads = as.integer(threads),
        opt_level = as.integer(opt_level)
    )

    structure(
        list(
            ptr = sess,
            env = env,
            path = path,
            provider = provider,
            threads = as.integer(threads),
            opt_level = as.integer(opt_level),
            input_names = ort_session_input_names(sess),
            output_names = ort_session_output_names(sess),
            n_inputs = ort_session_input_count(sess),
            n_outputs = ort_session_output_count(sess)
        ),
        class = "ort_session"
    )
}

#' @export
print.ort_session <- function(x, ...) {
    cat("nativeORT session\n")
    cat("  model:  ", x$path, "\n")
    cat("  threads:", ifelse(x$threads == 0, "auto", x$threads), "\n")
    cat("  inputs: ", paste(x$input_names, collapse = ", "), "\n")
    cat("  outputs:", paste(x$output_names, collapse = ", "), "\n")
    invisible(x)
}
