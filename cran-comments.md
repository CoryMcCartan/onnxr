## Test environments

* local R installation (macOS 15.7), R 4.6.0
* ubuntu-latest (on GitHub Actions), (oldrel-1, devel, and release)
* windows-latest (on GitHub Actions), (release)
* macOS-latest (on GitHub Actions), (release)
* Windows (on Winbuilder), (devel)

## R CMD check results

0 errors | 0 warnings | 1 note

* This is a new release, resubmitted after helpful feedback from CRAN.

* Examples are \donttest{} in onnx_model() and onnx_run() because they depend on
  'libonnxruntime', which is not available on CRAN. These examples are tested 
  locally and on CI systems where libonnxruntime is available.

* Per instructions from CRAN, there is no example for onnx_install() because it 
  downloads relatively large compiled libraries, which takes time.

* There are no references describing methods in this package to be included
  in the DESCRIPTION.

* The call to `par()` in the 'image' vignette now correctly saves the existing
  settings and restores them after the plot is produced.
