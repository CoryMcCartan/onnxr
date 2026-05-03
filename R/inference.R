#' ort_infer_raw
#'
#' @param session ORT session
#' @param input tensor to infer on
#'
#' @returns ORT output
#' @export
#'
#' @examples \dontrun{ort_infer_raw(session, input)}
ort_infer_raw <- function(session, input){
  ort_run(
    session_ptr = session$ptr,
    input_array = as.numeric(input),
    input_shape = as.integer(dim(input)),
    input_name = session$input_names[1],
    output_name = session$output_names[1]
  )
}
