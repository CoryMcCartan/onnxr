.onnx_version <- "1.25.1"
.onnx_version_short <- "1.25"

# ---- Exported functions ----

#' Check whether ONNX Runtime is loaded
#'
#' @returns `TRUE` if the ONNX Runtime shared library has been loaded
#'   in the current R session, `FALSE` otherwise.
#' @export
#' @name onnx_is_loaded
#'
#' @examples
#' onnx_is_loaded()
NULL

#' Check whether ONNX Runtime is available
#'
#' @returns `TRUE` if the ONNX Runtime shared library can be found
#'   in any of the standard search locations, `FALSE` otherwise.
#' @export
#'
#' @examples
#' onnx_is_installed()
onnx_is_installed <- function() {
    !is.null(onnx_find_lib())
}

#' Find the ONNX Runtime shared library
#'
#' Searches for the ONNX Runtime shared library in standard locations:
#' the `ORT_ROOT` environment variable, common system library paths,
#' the per-user install from [onnx_install()], the Python `onnxruntime`
#' package, and `pkg-config`.
#'
#' @returns Full path to the shared library, or `NULL` if not found.
#' @export
#'
#' @examples
#' onnx_find_lib()
onnx_find_lib <- function() {
    lib_name <- .onnx_lib_name()

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

    # 3. Package's R_user_dir (from onnx_install())
    pkg_path <- file.path(onnx_install_dir(), "lib", lib_name)
    if (file.exists(pkg_path)) {
        return(normalizePath(pkg_path))
    }

    # 4. pkg-config
    pkgconf <- Sys.which("pkg-config")
    if (nzchar(pkgconf)) {
        pc_out <- tryCatch(
            system2(
                "pkg-config",
                c("--libs-only-L", "onnxruntime"),
                stdout = TRUE,
                stderr = FALSE
            ),
            error = function(e) NULL,
            warning = function(w) NULL
        )
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
#' @param cuda Whether to install the CUDA-enabled build for GPU acceleration.
#'   - `NULL` (default): auto-detect by checking for `nvidia-smi` on the
#'     system `PATH`. Installs the CUDA build if a GPU is found and the
#'     platform is supported (Linux x64 or Windows x64), otherwise falls
#'     back to the CPU build.
#'   - `TRUE`: force the CUDA build. Errors if the platform is not
#'     Linux x64 or Windows x64.
#'   - `FALSE`: always install the CPU-only build.
#'
#' @returns Invisibly, the path to the installation directory.
#' @export
#'
#' @examples \dontrun{
#' onnx_install()
#' onnx_install(cuda = TRUE)
#' }
onnx_install <- function(cuda = NULL) {
    if (onnx_is_installed()) {
        message("onnxruntime ", .onnx_version, " is already installed.")
        return(invisible(onnx_install_dir()))
    }

    os <- onnx_detect_os()

    if (is.null(cuda)) {
        cuda <- os %in% c("linux-x64", "win-x64") && nzchar(Sys.which("nvidia-smi"))
        if (cuda) message("NVIDIA GPU detected; installing CUDA build.")
    }

    if (cuda && !os %in% c("linux-x64", "win-x64")) {
        stop("CUDA builds are only available for Linux x64 and Windows x64, ",
             "not ", os)
    }

    url <- onnx_binary_url(cuda = cuda)
    dest <- onnx_install_dir()

    onnx_download(url, dest)

    # The extracted directory name includes "-gpu" for CUDA builds
    dir_suffix <- if (cuda) paste0(os, "-gpu") else os
    extracted <- file.path(dest, paste0("onnxruntime-", dir_suffix, "-", .onnx_version))
    if (!all(file.copy(file.path(extracted, "include"), dest, recursive = TRUE))) {
        stop("Failed to copy ONNX Runtime include files to ", dest)
    }
    if (!all(file.copy(file.path(extracted, "lib"), dest, recursive = TRUE))) {
        stop("Failed to copy ONNX Runtime library files to ", dest)
    }
    unlink(extracted, recursive = TRUE)

    if (grepl("^osx-", os)) {
        onnx_codesign(file.path(dest, "lib"))
    }

    message("onnxruntime ", .onnx_version,
            if (cuda) " (CUDA)" else "",
            " installed successfully.")
    message("location: ", dest)

    # Load the library immediately so it's available without restarting R
    lib_path <- onnx_find_lib()
    if (!is.null(lib_path) && !onnx_is_loaded()) {
        onnx_load_lib(lib_path)
    }

    invisible(dest)
}

# ---- Internal helpers ----

# Detect the current OS and architecture as an ORT platform string.
onnx_detect_os <- function() {
    os_name <- Sys.info()[["sysname"]]
    arch <- R.version$arch
    if (os_name == "Darwin") {
        if (grepl("aarch64|arm64", arch)) {
            "osx-arm64"
        } else {
            "osx-x86_64"
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
.onnx_lib_name <- function() {
    os <- onnx_detect_os()
    switch(
        os,
        "osx-arm64" = ,
        "osx-x86_64" = "libonnxruntime.dylib",
        "linux-x64" = ,
        "linux-aarch64" = "libonnxruntime.so",
        "win-x64" = "onnxruntime.dll"
    )
}

# Path to the per-user onnxr data directory.
onnx_install_dir <- function() {
    tools::R_user_dir("onnxr", which = "data")
}

# Construct the GitHub release URL for the current platform and ORT version.
onnx_binary_url <- function(cuda = FALSE) {
    os <- onnx_detect_os()
    base <- "https://github.com/microsoft/onnxruntime/releases/download"
    gpu_suffix <- if (cuda) "-gpu" else ""
    ext <- if (os == "win-x64") ".zip" else ".tgz"
    paste0(base, "/v", .onnx_version, "/onnxruntime-", os, gpu_suffix, "-", .onnx_version, ext)
}

# Sign downloaded dylibs on macOS to satisfy Gatekeeper.
onnx_codesign <- function(lib_dir) {
    dylibs <- list.files(lib_dir, pattern = "\\.dylib$", full.names = TRUE)
    if (length(dylibs) == 0) {
        warning("No .dylib files found!")
    }
    message("Signing libraries (macOS)...")
    for (lib in dylibs) {
        system2("xattr", c("-dr", "com.apple.quarantine", shQuote(lib)), stderr = FALSE) # xattr may fail if no quarantine flag; that's OK
        ret <- system2(
            "codesign",
            c("--force", "--deep", "--sign", "-", shQuote(lib)),
            stdout = TRUE, stderr = TRUE
        )
        if (!is.null(attr(ret, "status"))) {
            warning("codesign failed for ", basename(lib), ": ", paste(ret, collapse = "\n"))
        }
    }
    invisible(lib_dir)
}

# Download and extract an ORT release archive.
onnx_download <- function(url, dest_dir) {
    dir.create(dest_dir, recursive = TRUE, showWarnings = FALSE)
    archive_path <- file.path(dest_dir, basename(url))

    message("Downloading ONNX Runtime ", .onnx_version, "...")
    utils::download.file(url, archive_path, mode = "wb")

    message("Extracting...")
    if (grepl("\\.zip$", archive_path)) {
        utils::unzip(archive_path, exdir = dest_dir)
    } else {
        utils::untar(archive_path, exdir = dest_dir)
    }
    unlink(archive_path)

    invisible(dest_dir)
}
