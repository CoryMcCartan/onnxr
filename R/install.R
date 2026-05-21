.ort_version <- "1.25.1"
.ort_version_short <- "1.25"
.ort_checksums <- list(
    "osx-arm64" = "18987ec3187b5f29ba798109750f6135060560ad4e0a52678fcc753ee8fb3091",
    "linux-aarch64" = "daa71b56b00c4ab34798a3d96ca41a32ece4d3e302dc2386d3cca83fd4491214",
    "linux-x64" = "eb566a49cfc49ef0642f809b69340b5bb656c7c4905ba873526d226f2c005816",
    "win-x64" = "33f2e8a63774811f99a5fc224cac32f4eed8c27643d46c6cc685319fa8f18019"
)

# Detect the current OS and architecture as an ORT platform string.
# Returns one of "osx-arm64", "linux-aarch64", "linux-x64", or "win-x64".
ort_detect_os <- function() {
    os_name <- Sys.info()[["sysname"]]
    arch <- R.version$arch
    if (os_name == "Darwin") {
        if (grepl("aarch64|arm64", arch)) {
            return("osx-arm64")
        } else {
            return("osx-arm64")
        }
    } else if (os_name == "Linux") {
        if (grepl("aarch64|arm64", arch)) {
            return("linux-aarch64")
        } else {
            return("linux-x64")
        }
    } else if (os_name == "Windows") {
        return("win-x64")
    } else {
        stop("Unsupported platform")
    }
}

# Path to the per-user nativeORT data directory.
ort_install_dir <- function() {
    tools::R_user_dir("nativeORT", which = "data")
}

# Construct the GitHub release URL for the current platform and ORT version.
ort_binary_url <- function() {
    os <- ort_detect_os()
    base <- "https://github.com/microsoft/onnxruntime/releases/download"
    glue::glue("{base}/v{.ort_version}/onnxruntime-{os}-{.ort_version}.tgz")
}

#' Check whether ONNX Runtime is installed locally
#'
#' Checks for the presence of the ONNX Runtime shared library in the
#' per-user data directory managed by [ort_install()].
#'
#' @returns `TRUE` if the library file exists, `FALSE` otherwise.
#' @export
#'
#' @examples
#' ort_is_installed()
ort_is_installed <- function() {
    lib_file <- .ort_lib_name()
    lib <- file.path(ort_install_dir(), "lib", lib_file)
    file.exists(lib)
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

# Download an ORT release tarball and verify its SHA-256 checksum.
ort_download <- function(url, dest_dir) {
    dir.create(dest_dir, recursive = TRUE, showWarnings = FALSE)
    tgz_path <- file.path(dest_dir, basename(url))

    message("Downloading ONNX Runtime ", .ort_version, "...")
    utils::download.file(url, tgz_path, mode = "wb")

    message("Verifying download...")
    os <- ort_detect_os()
    expected_sum <- .ort_checksums[[os]]
    received_sum <- digest::digest(tgz_path, algo = "sha256", file = TRUE)

    if (!identical(expected_sum, received_sum)) {
        unlink(tgz_path)
        stop("Checksum mismatch! Download may be corrupt, try again.")
    }

    message("Checksum verified!")
    message("Extracting...")

    utils::untar(tgz_path, exdir = dest_dir)
    unlink(tgz_path)

    invisible(dest_dir)
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

    message("onnxruntime ", .ort_version, " installed successfully!")
    message("location: ", dest)

    # Load the library immediately so it's available without restarting R
    lib_path <- ort_find_lib()
    if (!is.null(lib_path) && !ort_is_loaded()) {
        ort_load_lib(lib_path)
    }

    invisible(dest)
}
