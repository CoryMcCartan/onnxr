#' Load an ONNX model
#'
#' Loads an `.onnx` model file and creates a model object for running inference.
#'
#' @param path Path to an `.onnx` model file.
#' @param provider Execution provider. Available options depend on the
#'   platform and ORT build:
#'   - `"cpu"` — Default, available everywhere.
#'   - `"coreml"` — Apple Neural Engine + CPU (macOS/iOS only).
#'   - `"cuda"` — NVIDIA GPU (requires CUDA-enabled ORT build).
#'   - `"xnnpack"` — Optimized CPU kernels (mobile/embedded).
#'   - `"openvino"` — Intel hardware acceleration.
#' @param cache_dir Optional directory for CoreML model cache. Set to `NULL` to disable caching.
#' @param threads Number of threads. `0` uses all available;
#'   a positive integer sets a fixed thread count.
#' @param opt_level Graph optimization level. `99` (default) enables all
#'   optimizations; `1` for basic only; `0` to disable.
#'
#' @returns An `"ort_model"` object (a named list) with model metadata
#'   and internal pointers used by [ort_run()].
#' @export
#'
#' @examples \donttest{
#' model_path <- system.file("extdata", "lm_iris.onnx", package = "nativeORT")
#' if (ort_is_loaded() && nzchar(model_path)) {
#'     sess <- ort_model(model_path)
#'     sess
#' }
#' }
ort_model <- function(
    path,
    provider = c("cpu", "coreml", "cuda", "xnnpack", "openvino"),
    cache_dir = tools::R_user_dir("nativeORT", "cache"),
    threads = 1L,
    opt_level = 99L
) {
    provider <- match.arg(provider)
    if (!file.exists(path)) {
        stop("Model file not found")
    }
    if (!grepl("\\.onnx$", path, ignore.case = TRUE)) {
        stop("File must be an .onnx model")
    }
    if (provider == "coreml" && Sys.info()[["sysname"]] != "Darwin") {
        warning("CoreML is only available on macOS, falling back to CPU")
        provider <- "cpu"
    }
    cache <- if (!is.null(cache_dir)) {
        normalizePath(cache_dir, mustWork = FALSE)
    } else {
        ""
    }

    model_path <- normalizePath(path)

    # Detect external data files (.onnx_data) alongside the model.
    # These are pre-loaded into memory so that non-CPU providers
    # (especially CoreML) can access them without resolving relative paths.
    model_dir <- dirname(model_path)
    external_data <- list.files(model_dir, pattern = "[.]onnx_data$",
        full.names = TRUE)

    env <- ort_create_env()
    sess <- ort_create_session(
        env_ptr = env,
        model_path = model_path,
        provider = provider,
        cache_dir = cache,
        threads = as.integer(threads),
        opt_level = as.integer(opt_level),
        external_data_files = external_data
    )

    input_shapes <- ort_session_input_shapes(sess)
    output_shapes <- ort_session_output_shapes(sess)
    input_names <- ort_session_input_names(sess)
    output_names <- ort_session_output_names(sess)
    names(input_shapes) <- input_names
    names(output_shapes) <- output_names

    structure(
        list(
            ptr = sess,
            env = env,
            path = path,
            provider = provider,
            threads = as.integer(threads),
            opt_level = as.integer(opt_level),
            input_names = input_names,
            output_names = output_names,
            n_inputs = ort_session_input_count(sess),
            n_outputs = ort_session_output_count(sess),
            input_shapes = input_shapes,
            output_shapes = output_shapes,
            input_types = ort_session_input_types(sess),
            output_types = ort_session_output_types(sess)
        ),
        class = "ort_model"
    )
}

# Map ORT element type codes to human-readable names
.ort_type_names <- c(
    "undefined",
    "float",
    "uint8",
    "int8",
    "uint16",
    "int16",
    "int32",
    "int64",
    "string",
    "bool",
    "float16",
    "double",
    "uint32",
    "uint64"
)
.ort_type_name <- function(code) {
    ifelse(
        code >= 0L & code < length(.ort_type_names),
        .ort_type_names[code + 1L],
        paste0("type(", code, ")")
    )
}

# Format a shape vector as a string, e.g. "[?, 3]"
.fmt_shape <- function(shape) {
    dims <- ifelse(shape < 0L, "?", as.character(shape))
    paste0("[", paste(dims, collapse = ", "), "]")
}


#' @export
print.ort_model <- function(x, ...) {
    cat("nativeORT model\n")
    cat("  model:  ", x$path, "\n")
    cat("  provider:", x$provider, " threads:", ifelse(x$threads == 0, "auto", x$threads), "\n")
    for (i in seq_len(x$n_inputs)) {
        cat(sprintf(
            "  input:  %s %s <%s>\n",
            x$input_names[i],
            .fmt_shape(x$input_shapes[[i]]),
            .ort_type_name(x$input_types[i])
        ))
    }
    for (i in seq_len(x$n_outputs)) {
        cat(sprintf(
            "  output: %s %s <%s>\n",
            x$output_names[i],
            .fmt_shape(x$output_shapes[[i]]),
            .ort_type_name(x$output_types[i])
        ))
    }
    invisible(x)
}
