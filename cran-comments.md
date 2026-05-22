## Test environments

* local R installation (macOS 15.7), R 4.6.0
* ubuntu-latest (on GitHub Actions), (oldrel-1, devel, and release)
* windows-latest (on GitHub Actions), (release)
* macOS-latest (on GitHub Actions), (release)
* Windows (on Winbuilder), (devel)

## R CMD check results

0 errors | 0 warnings | 1 note

* This is a new release.

* Examples are \donttest{} in onnx_model() and onnx_run() because they depend on
  libonnxruntime, which is not available on CRAN. These examples are tested 
  locally and on CI systems where libonnxruntime is available.

* Examples are \dontrun{} in onnx_install() because they download compiled libraries.
  The underlying sources are too large (> 100MB) to bundle with the package or
  download at install time. The compiled libraries are < 100MB on most systems,
  except if CUDA support is enabled. This approach follows the similar 'nativeORT'
  package and the 'torch' package.