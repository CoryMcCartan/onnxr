#' Run inference on an ONNX model
#'
#' Passes `input` through the model's first input/output pair and returns
#' the result. Handles conversion between R's column-major arrays and
#' ONNX's row-major tensors automatically.
#'
#' @param session An `"ort_session"` object created by [ort_session()].
#' @param input A numeric matrix or array to use as model input.
#'   Must have a `dim` attribute matching the model's expected input shape.
#'
#' @returns A numeric array with dimensions matching the model's output shape.
#' @export
#'
#' @examples \dontrun{
#' sess <- ort_session("model.onnx")
#' ort_infer_raw(sess, input_matrix)
#' }
ort_infer_raw <- function(session, input) {
    ort_run(
        session_ptr = session$ptr,
        input_array = as.numeric(input),
        input_shape = as.integer(dim(input)),
        input_name = session$input_names[1],
        output_name = session$output_names[1]
    )
}
