## usethis namespace: start
#' @useDynLib onnxr, .registration = TRUE
## usethis namespace: end
NULL

.onLoad <- function(libname, pkgname) {
    lib_path <- onnx_find_lib()
    if (!is.null(lib_path)) {
        if (!onnx_load_lib(lib_path)) {
            warning("Found ONNX Runtime at ", lib_path, " but failed to load it.")
        } else {
            loaded_ver <- package_version(onnx_version())
            min_ver <- package_version(.onnx_version)
            if (loaded_ver < min_ver) {
                # fmt: skip
                warning(
                    "ONNX Runtime version ", loaded_ver, " is older than ",
                    "the minimum required (", min_ver, "). ",
                    "Run onnx_install() to get a compatible version."
                )
            }
        }
    }
}

.onAttach <- function(libname, pkgname) {
    if (!onnx_is_loaded()) {
        packageStartupMessage(
            "ONNX Runtime not found. Run onnx_install() to download it."
        )
    }
}

.onUnload <- function(libpath) {
    onnx_unload_lib()
}
