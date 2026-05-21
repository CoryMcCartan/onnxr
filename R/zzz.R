.onAttach <- function(libname, pkgname) {
    if (!ort_is_installed()) {
        packageStartupMessage(
            "nativeORT: ONNX Runtime not found.\n",
            "  Run ort_install() to download it \n.",
            "  Then re-install nativeoRT to enable inference."
        )
    }
}
