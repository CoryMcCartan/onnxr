# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Overview

nativeORT is an R package providing native bindings to ONNX Runtime for ML model inference without Python dependencies. It uses Rcpp to interface with the ONNX Runtime C++ API and supports CPU and CoreML (Apple Silicon) execution providers.

## Build and Check Commands

```bash
# Install from source (runs configure script to find/link libonnxruntime)
R CMD INSTALL .

# Build source tarball
R CMD build .

# CRAN check
R CMD check nativeORT_*.tar.gz

# Regenerate Rcpp bindings after modifying C++ [[Rcpp::export]] functions
Rscript -e "Rcpp::compileAttributes()"

# Generate roxygen2 docs after modifying R docstrings
Rscript -e "roxygen2::roxygenise()"
```

There is no test suite (`tests/` does not exist).

## Architecture

### Build System

The `configure` shell script searches for libonnxruntime in three locations (in order): pkg-config, the `ORT_ROOT` env var, and the user cache dir (`tools::R_user_dir('nativeORT', which='data')`). It substitutes `@ORT_CFLAGS@` and `@ORT_LIBS@` into `src/Makevars.in` to produce `src/Makevars`. If ORT is not found, the package still builds but all C++ functions become stubs guarded by `#ifdef HAVE_ORT`, returning errors that direct users to call `ort_install()`.

### C++ Layer (src/)

- **session.cpp** — Creates ORT environment and session objects, manages CoreML provider setup, exposes model metadata (input/output names and counts). Objects are stored as `Rcpp::XPtr<>` external pointers.
- **inference.cpp** — `ort_run()` converts R numeric vectors to float tensors, runs inference, returns results as dimensioned R numeric vectors.
- **nativeORT.cpp** — `ort_version()` returns the linked ORT version string.
- **RcppExports.cpp** — Auto-generated; do not edit directly.

### R Layer (R/)

- **session.R** — `ort_session()` wraps C++ session creation with R-level metadata and an S3 print method.
- **inference.R** — `ort_infer_raw()` wraps the C++ `ort_run()` call.
- **install.R** — Platform detection, binary download (ORT v1.25.1 with SHA256 verification), extraction, and macOS code signing. Hard-coded version and checksums.
- **zzz.R** — `.onAttach` startup hook.

### Key Design Patterns

- All C++ code is conditionally compiled with `#ifdef HAVE_ORT` so the package installs even without ORT libraries present.
- ONNX Runtime objects (Env, Session) use RAII via `Rcpp::XPtr` for automatic memory management.
- The two-phase install pattern: package installs first (possibly as stubs), then `ort_install()` downloads ORT binaries, then the user reinstalls to link against them.
