#' Run inference on an ONNX model
#'
#' Passes `input` through the model and returns the results. For models
#' with a single output, returns the output array directly. For models
#' with multiple outputs, returns a named list of arrays.
#'
#' Handles conversion between R's column-major arrays and ONNX's
#' row-major tensors, and between R's numeric types and the model's
#' declared element types (float, double, int32, int64).
#'
#' @param session An `"ort_model"` object created by [ort_model()].
#' @param ... Input arrays, either as unnamed arguments (matched to model
#'   inputs by position) or as named arguments (matched by name). Each
#'   input must be a numeric or integer matrix/array with dimensions
#'   matching the model's expected input shape. For single-input models,
#'   a single array can be passed directly.
#' @param simplify If `TRUE`, return the output array directly for
#'   single-output models instead of a length-1 named list.
#'
#' @returns A named list of output arrays, or (if `simplify = TRUE` and
#'   the model has a single output) the output array directly.
#' @export
#'
#' @examples \donttest{
#' model_path <- system.file("extdata", "lm_iris.onnx", package = "nativeORT")
#' if (ort_is_loaded() && nzchar(model_path)) {
#'     sess <- ort_model(model_path)
#'     input <- as.matrix(iris[1:5, c("Sepal.Length", "Sepal.Width", "Petal.Length")])
#'     ort_run(sess, input)
#' }
#' }
ort_run <- function(session, ..., simplify = FALSE) {
    args <- list(...)

    # Match inputs to model input names
    if (length(args) == 1L && is.null(names(args))) {
        # Single unnamed input → first model input
        input_list <- args
        input_order <- 1L
    } else if (!is.null(names(args)) && all(nzchar(names(args)))) {
        # All named → match by name
        input_order <- match(names(args), session$input_names)
        if (anyNA(input_order)) {
            bad <- names(args)[is.na(input_order)]
            stop("Unknown input name(s): ", paste(bad, collapse = ", "),
                ". Model inputs: ", paste(session$input_names, collapse = ", "))
        }
        input_list <- args
    } else {
        # Positional
        if (length(args) != session$n_inputs) {
            stop(sprintf("Model has %d input(s) but %d provided.",
                session$n_inputs, length(args)))
        }
        input_list <- args
        input_order <- seq_len(session$n_inputs)
    }

    # Validate each input's type and dimensions
    for (j in seq_along(input_list)) {
        i <- input_order[j]
        if (!is.numeric(input_list[[j]]) && !is.integer(input_list[[j]])) {
            stop(sprintf("Input '%s' must be numeric or integer, not %s.",
                session$input_names[i], typeof(input_list[[j]])))
        }
        d <- dim(input_list[[j]])
        expected <- session$input_shapes[[i]]
        if (is.null(d)) {
            stop(sprintf("Input '%s' must have a dim attribute (use matrix or array).",
                session$input_names[i]))
        }
        if (length(d) != length(expected)) {
            stop(sprintf("Input '%s' has %d dimensions but model expects %d.",
                session$input_names[i], length(d), length(expected)))
        }
        for (k in seq_along(expected)) {
            if (expected[k] > 0L && d[k] != expected[k]) {
                stop(sprintf("Input '%s' dimension %d is %d but model expects %d.",
                    session$input_names[i], k, d[k], expected[k]))
            }
        }
    }

    # Build lists for C++
    input_shapes <- lapply(input_list, function(x) as.integer(dim(x)))
    input_types <- session$input_types[input_order]
    input_nms <- session$input_names[input_order]

    results <- ort_run_(
        session_ptr = session$ptr,
        inputs = input_list,
        input_shapes = input_shapes,
        input_names = input_nms,
        input_types = input_types,
        output_names = session$output_names
    )
    names(results) <- session$output_names

    if (isTRUE(simplify) && length(results) == 1L) {
        results[[1L]]
    } else {
        results
    }
}
