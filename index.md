# onnxr

The **onnxr** package provides native R access to the [Open Neural
Network Exchange (ONNX) Runtime](https://onnxruntime.ai), which is a
performant engine for running machine learning models that are saved to
a standardized format. Rather than interfacing with ONNX via Python, as
in the official [onnx](https://cran.r-project.org/package=onnx) R
package, this package directly interfaces with the runtime’s C++ API via
[cpp11](https://cran.r-project.org/package=cpp11). This uses far less
memory and does not require a Python and TensorFlow installation.

Models saved to `.onnx` files can be loaded and run on various backends,
including CPUs and Apple’s CoreML library.

## Installation

You can install **onnxr** from
[CRAN](https://cran.r-project.org/package=onnxr) with:

``` r

install.packages("onnxr")
```

Or install the development version from [GitHub](https://github.com/)
with:

``` r

# install.packages("pak")
pak::pak("CoryMcCartan/onnxr")
```

## Getting started

On first load of the package, you will need to call
[`onnx_install()`](http://corymccartan.com/onnxr/reference/onnx_install.md)
if `libonnxruntime` is not already installed on your system. This will
download and install the latest version of the ONNX Runtime library,
which is roughly 35 MB.

``` r

onnxr::onnx_install()
```

The package comes with an example `.onnx` GLM model for predicting
*versicolor* vs *virginica* in the `iris` data.

``` r

library(onnxr)

model_path <- system.file("extdata", "glm_iris.onnx", package = "onnxr")
model <- onnx_model(model_path)
```

By default, the model will run on the CPU. Other backends are available
with the `backend` argument to
[`onnx_model()`](http://corymccartan.com/onnxr/reference/onnx_model.md).
For example, setting `onnx_model(model_path, backend = "coreml")` would
load the model to run via Apple’s CoreML library on Apple Silicon
devices.

Printing the model object shows information on the input and output
arrays.

``` r

model
#> onnxr model
#>   model:   /private/var/folders/64/lv8c__115kj6hxqc1f9sq5zr0000gn/T/Rtmpacscxg/temp_libpath8f8e2fc81c86/onnxr/extdata/glm_iris.onnx 
#>   backend: cpu  threads: 1 
#>   input:  X [?, 4] <float>
#>   output: label [?] <int64>
#>   output: probabilities [?, 2] <float>
```

The model can be called by using
[`onnx_run()`](http://corymccartan.com/onnxr/reference/onnx_run.md),
which accepts named or positional arguments matching the model’s inputs.

``` r

X = model.matrix(Species ~ 0 + ., data = iris)
output = onnx_run(model, X = X)
str(output)
#> List of 2
#>  $ label        : num [1:150(1d)] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ probabilities: num [1:150, 1:2] 1 1 1 1 1 1 1 1 1 1 ...
```

See the vignettes for a more involved end-to-end example.

## Prior Art

This package is based off of the
[nativeORT](https://cran.r-project.org/package=nativeORT) package by
[Caleb Carr](https://github.com/calebmcarr).
