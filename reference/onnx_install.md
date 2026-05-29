# Install ONNX Runtime

Downloads pre-built ONNX Runtime binaries for the current platform and
installs them to a per-user data directory. The library is loaded
immediately after installation, so there is no need to restart R.

## Usage

``` r
onnx_install(cuda = NULL)
```

## Arguments

- cuda:

  Whether to install the CUDA-enabled build for GPU acceleration.

  - `NULL` (default): auto-detect by checking for `nvidia-smi` on the
    system `PATH`. Installs the CUDA build if a GPU is found and the
    platform is supported (Linux x64 or Windows x64), otherwise falls
    back to the CPU build.

  - `TRUE`: force the CUDA build. Errors if the platform is not Linux
    x64 or Windows x64.

  - `FALSE`: always install the CPU-only build.

## Value

Invisibly, the path to the installation directory.

## Examples

``` r
# \donttest{
# Downloads files, which can take time depending on internet speed
onnx_install()
#> onnxruntime 1.25.1 is already installed.
# }
```
