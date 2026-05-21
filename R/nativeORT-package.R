## usethis namespace: start
#' @useDynLib nativeORT, .registration = TRUE
## usethis namespace: end
NULL

.onLoad <- function(libname, pkgname) {
    lib_path <- ort_find_lib()
    if (!is.null(lib_path)) {
        if (!ort_load_lib(lib_path)) {
            warning("Found ONNX Runtime at ", lib_path, " but failed to load it.")
        } else {
            loaded_ver <- package_version(ort_version())
            min_ver <- package_version(.ort_version)
            if (loaded_ver < min_ver) {
                warning(
                    "ONNX Runtime version ", loaded_ver, " is older than ",
                    "the minimum required (", min_ver, "). ",
                    "Run ort_install() to get a compatible version."
                )
            }
        }
    } else {
        packageStartupMessage(
            "ONNX Runtime not found. Run ort_install() to download it."
        )
    }
}

.onUnload <- function(libpath) {
    ort_unload_lib()
}
