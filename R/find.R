# Platform-specific shared library filename.
.ort_lib_name <- function() {
    os <- ort_detect_os()
    switch(os,
        "osx-arm64" = , "osx-x86_64" = "libonnxruntime.dylib",
        "linux-x64" = , "linux-aarch64" = "libonnxruntime.so",
        "win-x64" = "onnxruntime.dll"
    )
}

#' Find the ONNX Runtime shared library
#'
#' Searches for the ONNX Runtime shared library in standard locations:
#' the `ORT_ROOT` environment variable, common system library paths,
#' the per-user install from [ort_install()], the Python `onnxruntime`
#' package, and `pkg-config`.
#'
#' @returns Full path to the shared library, or `NULL` if not found.
#' @export
#'
#' @examples
#' ort_find_lib()
ort_find_lib <- function() {
    lib_name <- .ort_lib_name()

    # 1. ORT_ROOT env var
    ort_root <- Sys.getenv("ORT_ROOT", "")
    if (nzchar(ort_root)) {
        path <- file.path(ort_root, "lib", lib_name)
        if (file.exists(path)) return(normalizePath(path))
    }

    # 2. Common system paths
    sys_dirs <- c("/usr/local/lib", "/usr/lib")
    if (grepl("arm64|aarch64", R.version$arch)) {
        sys_dirs <- c("/opt/homebrew/lib", sys_dirs)
    }
    for (dir in sys_dirs) {
        path <- file.path(dir, lib_name)
        if (file.exists(path)) return(normalizePath(path))
    }

    # 3. Package's R_user_dir (from ort_install())
    pkg_path <- file.path(ort_install_dir(), "lib", lib_name)
    if (file.exists(pkg_path)) return(normalizePath(pkg_path))

    # 4. Python onnxruntime package
    python <- Sys.which("python3")
    if (!nzchar(python)) python <- Sys.which("python")
    if (nzchar(python)) {
        py_path <- tryCatch({
            out <- system2(python, c("-c",
                shQuote("import onnxruntime, os; print(os.path.dirname(onnxruntime.__file__))")),
                stdout = TRUE, stderr = FALSE)
            if (length(out) == 1 && nzchar(out)) {
                capi_path <- file.path(out, "capi", lib_name)
                if (file.exists(capi_path)) capi_path else NULL
            }
        }, error = function(e) NULL, warning = function(w) NULL)
        if (!is.null(py_path)) return(normalizePath(py_path))
    }

    # 5. pkg-config
    pkgconf <- Sys.which("pkg-config")
    if (nzchar(pkgconf)) {
        pc_out <- tryCatch({
            system2("pkg-config", c("--libs-only-L", "onnxruntime"),
                stdout = TRUE, stderr = FALSE)
        }, error = function(e) NULL, warning = function(w) NULL)
        if (!is.null(pc_out) && length(pc_out) == 1 && nzchar(pc_out)) {
            lib_dir <- sub("^-L", "", trimws(pc_out))
            path <- file.path(lib_dir, lib_name)
            if (file.exists(path)) return(normalizePath(path))
        }
    }

    NULL
}
