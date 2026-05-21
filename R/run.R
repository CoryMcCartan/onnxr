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
#' @param session An `"ort_session"` object created by [ort_session()].
#' @param input A numeric matrix or array to use as model input.
#'   Must have a `dim` attribute matching the model's expected input shape.
#'
#' @returns A numeric or integer array (single-output models), or a named
#'   list of arrays (multi-output models).
#' @export
#'
#' @examples \dontrun{
#' sess <- ort_session("model.onnx")
#' ort_run(sess, input_matrix)
#' }
ort_run <- function(session, input) {
    d <- dim(input)
    expected <- session$input_shapes[[1L]]

    # Validate number of dimensions
    if (length(d) != length(expected)) {
        stop(sprintf(
            "Input has %d dimensions but model expects %d.",
            length(d), length(expected)
        ))
    }
    # Validate non-batch dimensions (those != -1 in declared shape)
    for (i in seq_along(expected)) {
        if (expected[i] > 0L && d[i] != expected[i]) {
            stop(sprintf(
                "Input dimension %d is %d but model expects %d.",
                i, d[i], expected[i]
            ))
        }
    }

    input_vec <- as.numeric(input)
    input_shp <- as.integer(d)
    input_nm <- session$input_names[1L]

    outputs <- lapply(seq_len(session$n_outputs), function(i) {
        ort_run_(
            session_ptr = session$ptr,
            input_array = input_vec,
            input_shape = input_shp,
            input_name = input_nm,
            output_name = session$output_names[i],
            output_type = session$output_types[i]
        )
    })
    names(outputs) <- session$output_names

    if (length(outputs) == 1L) outputs[[1L]] else outputs
}
