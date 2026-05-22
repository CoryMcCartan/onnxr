# Load an ONNX model

Loads an `.onnx` model file and creates a model object.

## Usage

``` r
onnx_model(
  path,
  backend = c("cpu", "coreml", "cuda", "xnnpack", "openvino"),
  cache_dir = tools::R_user_dir("onnxr", "cache"),
  threads = 1L,
  opt_level = 99L
)
```

## Arguments

- path:

  Path to an `.onnx` model file.

- backend:

  Execution backend. Available options depend on the platform and ORT
  build:

  - `"cpu"` — Default, available everywhere.

  - `"coreml"` — Apple Neural Engine + CPU (macOS/iOS only).

  - `"cuda"` — NVIDIA GPU (Linux x64 and Windows x64 only). Requires
    CUDA toolkit and the CUDA-enabled ORT build from
    [onnx_install](http://corymccartan.com/onnxr/reference/onnx_install.md)`(cuda = TRUE)`.

  - `"xnnpack"` — Optimized CPU kernels (mobile/embedded). Requires an
    ORT build with XNNPACK support (not provided by
    [`onnx_install()`](http://corymccartan.com/onnxr/reference/onnx_install.md)).

  - `"openvino"` — Intel hardware acceleration. Requires OpenVINO
    installation and ORT build with OpenVINO EP (not provided by
    [`onnx_install()`](http://corymccartan.com/onnxr/reference/onnx_install.md)).

- cache_dir:

  Optional directory for CoreML model cache. Set to `NULL` to disable
  caching.

- threads:

  Number of threads. `0` uses all available; a positive integer sets a
  fixed thread count.

- opt_level:

  Graph optimization level. `99` (default) enables all optimizations;
  `1` for basic only; `0` to disable.

## Value

An `"onnx_model"` object (a named list) with model metadata and internal
pointers used by
[`onnx_run()`](http://corymccartan.com/onnxr/reference/onnx_run.md).

## Examples

``` r
# \donttest{
model_path <- system.file("extdata", "lm_iris.onnx", package = "onnxr")
if (onnx_is_loaded() && nzchar(model_path)) {
    sess <- onnx_model(model_path)
    sess
}
#> onnxr model
#>   model:   /home/runner/work/_temp/Library/onnxr/extdata/lm_iris.onnx 
#>   backend: cpu  threads: 1 
#>   input:  X [?, 3] <float>
#>   output: variable [?, 1] <float>
# }
```
