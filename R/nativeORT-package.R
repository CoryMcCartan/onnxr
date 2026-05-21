## usethis namespace: start
#' @useDynLib nativeORT, .registration = TRUE
## usethis namespace: end
NULL

.onLoad <- function(libname, pkgname) {
    lib_path <- ort_find_lib()
    if (!is.null(lib_path)) {
        if (!ort_load_lib(lib_path)) {
            warning("Found ONNX Runtime at ", lib_path, " but failed to load it.")
        }
    }
}
