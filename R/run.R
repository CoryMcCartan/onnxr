#' Run or predict from an ONNX model
#'
#' Passes `input` through the model and returns the results. For models
#' with a single output, returns the output array directly. For models
#' with multiple outputs, returns a named list of arrays.
#'
#' Handles conversion between R's column-major arrays and ONNX's
#' row-major tensors, and between R's numeric types and the model's
#' declared element types (float, double, int32, int64). Note that int64 outputs
#' are cast to doubles, which may lose precision for large integers.
#'
#' @param model An `"onnx_model"` object created by [onnx_model()].
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
#' model_path <- system.file("extdata", "lm_iris.onnx", package = "onnxr")
#' if (onnx_is_loaded() && nzchar(model_path)) {
#'     sess <- onnx_model(model_path)
#'     input <- as.matrix(iris[1:5, c("Sepal.Length", "Sepal.Width", "Petal.Length")])
#'     onnx_run(sess, input)
#' }
#' }
onnx_run <- function(model, ..., simplify = FALSE) {
    args <- list(...)

    # Match inputs to model input names
    if (length(args) == 1L && is.null(names(args))) {
        # Single unnamed input → first model input
        input_list <- args
        input_order <- 1L
    } else if (!is.null(names(args)) && all(nzchar(names(args)))) {
        # All named → match by name
        input_order <- match(names(args), model$input_names)
        if (anyNA(input_order)) {
            bad <- names(args)[is.na(input_order)]
            # fmt: skip
            stop(
                "Unknown input name(s): ", paste(bad, collapse = ", "), ". ", 
                "Model inputs: ", paste(model$input_names, collapse = ", ")
            )
        }
        input_list <- args
    } else {
        # Positional
        if (length(args) != model$n_inputs) {
            stop("Model has ", model$n_inputs, " input(s) but ", length(args), " provided.")
        }
        input_list <- args
        input_order <- seq_len(model$n_inputs)
    }

    # Validate each input's type and dimensions
    for (j in seq_along(input_list)) {
        i <- input_order[j]
        if (!is.numeric(input_list[[j]]) && !is.integer(input_list[[j]])) {
            # fmt: skip
            stop(
                "Input '", model$input_names[i], "' must be numeric or integer, ", 
                "not ", typeof(input_list[[j]]), "."
            )
        }
        d <- dim(input_list[[j]])
        expected <- model$input_shapes[[i]]
        if (is.null(d)) {
            # fmt: skip
            stop("Input '", model$input_names[i], "' must have a dim attribute (use matrix or array).")
        }
        if (length(d) != length(expected)) {
            # fmt: skip
            stop(
                "Input '", model$input_names[i], "' has ", length(d),
                " dimensions but model expects ", length(expected), "."
            )
        }
        for (k in seq_along(expected)) {
            if (expected[k] > 0L && d[k] != expected[k]) {
                # fmt: skip
                stop(
                    "Input '", model$input_names[i], "' dimension ", k,
                    " is ", d[k], " but model expects ", expected[k], "."
                )
            }
        }
    }

    results <- onnx_run_(
        session_ptr = model$ptr,
        inputs = input_list,
        input_shapes = lapply(input_list, function(x) as.integer(dim(x))),
        input_names = model$input_names[input_order],
        input_types = model$input_types[input_order],
        output_names = model$output_names
    )

    if (isTRUE(simplify) && length(results) == 1L) {
        results[[1L]]
    } else {
        results
    }
}
