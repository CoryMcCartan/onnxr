.ort_version <- "1.25.1"
.ort_version_short <- "1.25"

# ---- Exported functions ----

#' Check whether ONNX Runtime is loaded
#'
#' @returns `TRUE` if the ONNX Runtime shared library has been loaded
#'   in the current R session, `FALSE` otherwise.
#' @export
#' @name ort_is_loaded
#'
#' @examples
#' ort_is_loaded()
NULL

#' Check whether ONNX Runtime is available
#'
#' @returns `TRUE` if the ONNX Runtime shared library can be found
#'   in any of the standard search locations, `FALSE` otherwise.
#' @export
#'
#' @examples
#' ort_is_installed()
ort_is_installed <- function() {
    !is.null(ort_find_lib())
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

#' Install ONNX Runtime
#'
#' Downloads pre-built ONNX Runtime binaries for the current platform
#' and installs them to a per-user data directory. The library is loaded
#' immediately after installation, so there is no need to restart R.
#'
#' @returns Invisibly, the path to the installation directory.
#' @export
#'
#' @examples \dontrun{
#' ort_install()
#' }
ort_install <- function() {
    if (ort_is_installed()) {
        message("onnxruntime ", .ort_version, " is already installed.")
        return(invisible(ort_install_dir()))
    }

    os <- ort_detect_os()
    url <- ort_binary_url()
    dest <- ort_install_dir()

    ort_download(url, dest)

    extracted <- file.path(dest, paste0("onnxruntime-", os, "-", .ort_version))
    file.copy(file.path(extracted, "include"), dest, recursive = TRUE)
    file.copy(file.path(extracted, "lib"), dest, recursive = TRUE)
    unlink(extracted, recursive = TRUE)

    if (os == "osx-arm64") {
        ort_codesign(file.path(dest, "lib"))
    }

    message("onnxruntime ", .ort_version, " installed successfully.")
    message("location: ", dest)

    # Load the library immediately so it's available without restarting R
    lib_path <- ort_find_lib()
    if (!is.null(lib_path) && !ort_is_loaded()) {
        ort_load_lib(lib_path)
    }

    invisible(dest)
}

# ---- Internal helpers ----

# Detect the current OS and architecture as an ORT platform string.
ort_detect_os <- function() {
    os_name <- Sys.info()[["sysname"]]
    arch <- R.version$arch
    if (os_name == "Darwin") {
        if (grepl("aarch64|arm64", arch)) {
            "osx-arm64"
        } else {
            "osx-arm64"
        }
    } else if (os_name == "Linux") {
        if (grepl("aarch64|arm64", arch)) {
            "linux-aarch64"
        } else {
            "linux-x64"
        }
    } else if (os_name == "Windows") {
        "win-x64"
    } else {
        stop("Unsupported platform")
    }
}

# Platform-specific shared library filename.
.ort_lib_name <- function() {
    os <- ort_detect_os()
    switch(os,
        "osx-arm64" = , "osx-x86_64" = "libonnxruntime.dylib",
        "linux-x64" = , "linux-aarch64" = "libonnxruntime.so",
        "win-x64" = "onnxruntime.dll"
    )
}

# Path to the per-user nativeORT data directory.
ort_install_dir <- function() {
    tools::R_user_dir("nativeORT", which = "data")
}

# Construct the GitHub release URL for the current platform and ORT version.
ort_binary_url <- function() {
    os <- ort_detect_os()
    base <- "https://github.com/microsoft/onnxruntime/releases/download"
    paste0(base, "/v", .ort_version, "/onnxruntime-", os, "-", .ort_version, ".tgz")
}

# Sign downloaded dylibs on macOS to satisfy Gatekeeper.
ort_codesign <- function(lib_dir) {
    dylibs <- list.files(lib_dir, pattern = "\\.dylib$", full.names = TRUE)
    if (length(dylibs) == 0) {
        warning("No .dylib files found!")
    }
    message("Signing libraries (macOS)...")
    for (lib in dylibs) {
        system2("xattr", c("-dr", "com.apple.quarantine", shQuote(lib)))
        system2("codesign", c("--force", "--deep", "--sign", "-", shQuote(lib)))
    }
    invisible(lib_dir)
}

# Download an ORT release tarball.
ort_download <- function(url, dest_dir) {
    dir.create(dest_dir, recursive = TRUE, showWarnings = FALSE)
    tgz_path <- file.path(dest_dir, basename(url))

    message("Downloading ONNX Runtime ", .ort_version, "...")
    utils::download.file(url, tgz_path, mode = "wb")

    message("Extracting...")
    utils::untar(tgz_path, exdir = dest_dir)
    unlink(tgz_path)

    invisible(dest_dir)
}
