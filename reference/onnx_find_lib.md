# Find the ONNX Runtime shared library

Searches for the ONNX Runtime shared library in standard locations: the
`ORT_ROOT` environment variable, common system library paths, the
per-user install from
[`onnx_install()`](http://corymccartan.com/onnxr/reference/onnx_install.md),
the Python `onnxruntime` package, and `pkg-config`.

## Usage

``` r
onnx_find_lib()
```

## Value

Full path to the shared library, or `NULL` if not found.

## Examples

``` r
onnx_find_lib()
#> [1] "/home/runner/.local/share/R/onnxr/lib/libonnxruntime.so"
```
