# Run inference on an ONNX model

Passes `input` through the model and returns the results. For models
with a single output, returns the output array directly. For models with
multiple outputs, returns a named list of arrays.

## Usage

``` r
onnx_run(model, ..., simplify = FALSE)
```

## Arguments

- model:

  An `"onnx_model"` object created by
  [`onnx_model()`](http://corymccartan.com/onnxr/reference/onnx_model.md).

- ...:

  Input arrays, either as unnamed arguments (matched to model inputs by
  position) or as named arguments (matched by name). Each input must be
  a numeric or integer matrix/array with dimensions matching the model's
  expected input shape. For single-input models, a single array can be
  passed directly.

- simplify:

  If `TRUE`, return the output array directly for single-output models
  instead of a length-1 named list.

## Value

A named list of output arrays, or (if `simplify = TRUE` and the model
has a single output) the output array directly.

## Details

Handles conversion between R's column-major arrays and ONNX's row-major
tensors, and between R's numeric types and the model's declared element
types (float, double, int32, int64).

## Examples

``` r
# \donttest{
model_path <- system.file("extdata", "lm_iris.onnx", package = "onnxr")
if (onnx_is_loaded() && nzchar(model_path)) {
    sess <- onnx_model(model_path)
    input <- as.matrix(iris[1:5, c("Sepal.Length", "Sepal.Width", "Petal.Length")])
    onnx_run(sess, input)
}
#> $variable
#>           [,1]
#> [1,] 0.2162523
#> [2,] 0.1462912
#> [3,] 0.1799020
#> [4,] 0.2831622
#> [5,] 0.2592619
#> 
# }
```
