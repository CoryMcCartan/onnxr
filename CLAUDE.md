# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Overview

onnxr is an R package providing native bindings to ONNX Runtime for ML model inference without Python dependencies. It uses cpp11 to interface with the ONNX Runtime C++ API and supports multiple execution providers (CPU, CoreML, CUDA, XNNPACK, OpenVINO).

## Build and Check Commands

```bash
# Document, rebuild, and test (preferred workflow)
Rscript -e "devtools::document()"
Rscript -e "devtools::test()"

# Install to system library
Rscript -e "devtools::install()"

# CRAN check
Rscript -e "devtools::check()"

# Update vendored ORT headers (pulls latest release, strips CRAN-flagged pragmas)
bash tools/update-ort-headers.sh          # latest
bash tools/update-ort-headers.sh 1.25.1   # specific version
```

`devtools::document()` handles cpp11 registration, roxygen2 docs, NAMESPACE updates, and recompilation automatically.

## Tests

Tests live in `tests/testthat/` and use `testthat` (edition 3). They validate ONNX inference against R model outputs using two pre-built ONNX models in `inst/extdata/`:

- `lm_iris.onnx` — Linear regression (Petal.Width ~ 3 features), compared with `lm()`
- `glm_iris.onnx` — Logistic regression (versicolor vs virginica), compared with `glm()`. Multi-output model (int64 labels + float probabilities).

The Python script `data-raw/build_models.py` generates these models using sklearn + skl2onnx. Tests skip automatically if ORT is not loaded.

## Exported API

- `onnx_model(path, backend, ...)` — Load an ONNX model. Returns an S3 object with model metadata (shapes, types, names).
- `onnx_run(session, input)` — Run inference. Validates input dimensions, handles all outputs. Returns array (single output) or named list (multi-output).
- `onnx_install()` — Download ORT binaries and load immediately.
- `onnx_is_installed()` / `onnx_is_loaded()` — Check ORT availability.
- `onnx_find_lib()` — Search for ORT shared library across system locations.

## Architecture

### Runtime Loading (no link-time dependency)

The package ships 5 vendored ORT C/C++ headers in `src/onnxruntime/` and loads the ORT shared library at runtime via `dlopen`/`LoadLibrary`. There is no configure script and no `-lonnxruntime` link flag.

- **`src/ort_loader.cpp`** — Provides a `dlopen` shim that defines `OrtGetApiBase()` (the single entry point the header-only C++ API requires). Exposes `onnx_load_lib(path)` and `onnx_is_loaded()` to R.
- **`R/runtime.R`** — All runtime lifecycle: exported functions (`onnx_is_loaded`, `onnx_is_installed`, `onnx_find_lib`, `onnx_install`) at top, `.onLoad` and internal helpers (`onnx_detect_os`, `.onnx_lib_name`, `onnx_install_dir`, `onnx_binary_url`, `onnx_codesign`, `onnx_download`) below. `.onLoad` calls `onnx_find_lib()` then `onnx_load_lib()`.

`onnx_find_lib()` search order: `ORT_ROOT` env var → system paths (`/usr/local/lib`, `/opt/homebrew/lib`) → per-user R data dir → `pkg-config`.

### C++ Layer (src/)

All C++ uses cpp11 (`[[cpp11::register]]`, `cpp11::external_pointer`, `cpp11::doubles`, etc.). Each function calls `onnx_check_loaded()` before using ORT.

- **ort_loader.cpp** — `dlopen` shim, `OrtGetApiBase()` override, `onnx_check_loaded()`.
- **session.cpp** — Creates ORT environment and session objects, manages execution provider setup (CPU, CoreML, CUDA, XNNPACK, OpenVINO via generic `AppendExecutionProvider` API), exposes model metadata (names, shapes, types). Objects stored as `cpp11::external_pointer<>`.
- **inference.cpp** — `onnx_run_()` (internal, called from R's `onnx_run()`): handles double↔float conversion, column-major (R) ↔ row-major (ONNX) permutation, and type-aware output (float, double → R double; int32, int64 → R integer).
- **ort_loader.cpp** — `onnx_version()` returns the loaded ORT version string.
- **cpp11.cpp** — Auto-generated; do not edit directly.

### R Layer (R/)

- **model.R** — `onnx_model(path, backend)` wraps C++ session creation. Stores shapes, types, names on the S3 object. `print.onnx_model()` shows input/output metadata.
- **run.R** — `onnx_run(session, input)` validates input dimensions against declared shapes, runs inference on all outputs, returns array (single output) or named list (multi-output).
- **runtime.R** — ORT lifecycle: find, load, install, platform detection.
- **onnxr-package.R** — `useDynLib` registration and `.onLoad`.
- **cpp11.R** — Auto-generated; do not edit directly.

### Key Design Patterns

- ORT is loaded at runtime via `dlopen`, not linked at compile time. The package always compiles and installs; ORT is only needed at runtime.
- A single `OrtGetApiBase()` definition in `ort_loader.cpp` satisfies the linker. The header-only C++ API calls through it transparently.
- `onnx_run_()` in C++ handles both data type conversion (R double → ORT float; ORT int64 → R integer) and memory layout conversion (column-major ↔ row-major).
- Execution providers use the generic `AppendExecutionProvider(name, options)` ORT API. Provider-specific options (e.g., CoreML MLProgram format) are set in `session.cpp`.
- ONNX Runtime objects (Env, Session) use RAII via `cpp11::external_pointer` for automatic memory management.
