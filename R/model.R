#' Load an ONNX model
#'
#' Loads an `.onnx` model file and creates a model object.
#'
#' @param path Path to an `.onnx` model file.
#' @param backend Execution backend. Available options depend on the
#'   platform and ORT build:
#'   - `"cpu"` — Default, available everywhere.
#'   - `"coreml"` — Apple Neural Engine + CPU (macOS/iOS only).
#'   - `"cuda"` — NVIDIA GPU (Linux x64 and Windows x64 only). Requires
#'     CUDA toolkit and the CUDA-enabled ORT build from
#'     `onnx_install(cuda = TRUE)`.
#'   - `"xnnpack"` — Optimized CPU kernels (mobile/embedded). Requires
#'     an ORT build with XNNPACK support (not provided by [onnx_install()]).
#'   - `"openvino"` — Intel hardware acceleration. Requires OpenVINO
#'     installation and ORT build with OpenVINO EP (not provided by [onnx_install()]).
#' @param cache_dir Optional directory for CoreML model cache. Set to `NULL` to disable caching.
#' @param threads Number of threads. `0` uses all available;
#'   a positive integer sets a fixed thread count.
#' @param opt_level Graph optimization level. `99` (default) enables all
#'   optimizations; `1` for basic only; `0` to disable.
#'
#' @returns An `"onnx_model"` object (a named list) with model metadata
#'   and internal pointers used by [onnx_run()].
#' @export
#'
#' @examples \donttest{
#' model_path <- system.file("extdata", "lm_iris.onnx", package = "onnxr")
#' if (onnx_is_loaded() && nzchar(model_path)) {
#'     sess <- onnx_model(model_path)
#'     sess
#' }
#' }
onnx_model <- function(
    path,
    backend = c("cpu", "coreml", "cuda", "xnnpack", "openvino"),
    cache_dir = tools::R_user_dir("onnxr", "cache"),
    threads = 1L,
    opt_level = 99L
) {
    backend <- match.arg(backend)
    if (!file.exists(path)) {
        stop("Model file not found")
    }
    if (!grepl("\\.onnx$", path, ignore.case = TRUE)) {
        stop("File must be an .onnx model")
    }
    if (backend == "coreml" && Sys.info()[["sysname"]] != "Darwin") {
        warning("CoreML is only available on macOS, falling back to CPU")
        backend <- "cpu"
    }
    cache <- if (!is.null(cache_dir)) {
        normalizePath(cache_dir, mustWork = FALSE)
    } else {
        ""
    }

    model_path <- normalizePath(path)

    env <- onnx_create_env()
    sess <- onnx_create_session(
        env_ptr = env,
        model_path = model_path,
        provider = backend,
        cache_dir = cache,
        threads = as.integer(threads),
        opt_level = as.integer(opt_level)
    )

    input_shapes <- onnx_session_input_shapes(sess)
    output_shapes <- onnx_session_output_shapes(sess)
    input_names <- onnx_session_input_names(sess)
    output_names <- onnx_session_output_names(sess)
    names(input_shapes) <- input_names
    names(output_shapes) <- output_names

    structure(
        list(
            ptr = sess,
            env = env,
            path = path,
            backend = backend,
            threads = as.integer(threads),
            opt_level = as.integer(opt_level),
            input_names = input_names,
            output_names = output_names,
            n_inputs = onnx_session_input_count(sess),
            n_outputs = onnx_session_output_count(sess),
            input_shapes = input_shapes,
            output_shapes = output_shapes,
            input_types = onnx_session_input_types(sess),
            output_types = onnx_session_output_types(sess),
            input_optional = onnx_session_input_optional(sess)
        ),
        class = "onnx_model"
    )
}

# Map ORT element type codes to human-readable names
.onnx_type_names <- c(
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
.onnx_type_name <- function(code) {
    ifelse(
        code >= 0L & code < length(.onnx_type_names),
        .onnx_type_names[code + 1L],
        paste0("type(", code, ")")
    )
}

# Format a shape vector as a string, e.g. "[?, 3]"
.fmt_shape <- function(shape) {
    dims <- ifelse(shape < 0L, "?", as.character(shape))
    paste0("[", paste(dims, collapse = ", "), "]")
}


#' @export
print.onnx_model <- function(x, ...) {
    cat("onnxr model\n")
    cat("  model:  ", x$path, "\n")
    cat("  backend:", x$backend, " threads:", ifelse(x$threads == 0, "auto", x$threads), "\n")
    for (i in seq_len(x$n_inputs)) {
        cat(sprintf(
            "  input:  %s %s <%s>%s\n",
            x$input_names[i],
            .fmt_shape(x$input_shapes[[i]]),
            .onnx_type_name(x$input_types[i]),
            if (isTRUE(x$input_optional[i])) " (optional)" else ""
        ))
    }
    for (i in seq_len(x$n_outputs)) {
        cat(sprintf(
            "  output: %s %s <%s>\n",
            x$output_names[i],
            .fmt_shape(x$output_shapes[[i]]),
            .onnx_type_name(x$output_types[i])
        ))
    }
    invisible(x)
}
