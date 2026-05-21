#' session.R
#'
#' @param path where the .onnx is located
#' @param type detection/classificaton/segmentation
#' @param provider cpu or coreml (only for apple silicon)
#' @param cache_dir where coreml cache be directed
#' @param threads =  0 for all, integer otherwise
#' @param opt_level = 99 for all ops, 1 for basics
#'
#' @returns ORT session object
#' @export
#'
#' @examples \dontrun{ort_session('./yolov8n.onnx', 'detection')}
ort_session <- function(
    path,
    type = 'detection',
    provider = "cpu",
    cache_dir = NULL,
    threads = 0L,
    opt_level = 99L
) {
    # file verifications
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
            type = type,
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


#' print.ort_session
#'
#' @param x session object
#' @param ... extra params
#'
#' @returns invisible session
#' @export
#'
#' @examples \dontrun{print(session)}
print.ort_session <- function(x, ...) {
    cat("nativeORT session\n")
    cat("  model: ", x$path, "\n")
    cat(" threads", ifelse(x$threads == 0, "auto", x$threads), "\n")
    cat("  inputs: ", paste(x$input_names, collapse = ", "), "\n")
    cat("  outputs: ", paste(x$output_names, collapse = ", "), "\n")
    invisible(x)
}
