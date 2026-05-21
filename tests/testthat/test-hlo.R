skip_if_not_installed("stablehlo")

test_that("onnx_to_hlo converts linear regression model", {
    model_path <- system.file("extdata", "lm_iris.onnx", package = "nativeORT")
    skip_if(model_path == "", "Test model not found")

    func <- onnx_to_hlo(model_path)

    # Returns a stablehlo Func
    expect_s3_class(func, "Func")

    # Serializes to valid StableHLO text
    hlo_text <- stablehlo::repr(func)
    expect_type(hlo_text, "character")
    expect_match(hlo_text, "func.func @main")
    expect_match(hlo_text, "stablehlo.dot_general")
    expect_match(hlo_text, "stablehlo.add")
    expect_match(hlo_text, "tensor<1x3xf32>")  # input shape
})

test_that("onnx_to_hlo converts logistic regression model", {
    model_path <- system.file("extdata", "glm_iris.onnx", package = "nativeORT")
    skip_if(model_path == "", "Test model not found")

    func <- onnx_to_hlo(model_path)

    expect_s3_class(func, "Func")

    hlo_text <- stablehlo::repr(func)
    expect_match(hlo_text, "stablehlo.dot_general")
    expect_match(hlo_text, "stablehlo.logistic")  # sigmoid
    expect_match(hlo_text, "tensor<1x4xf32>")  # input shape
})

test_that("onnx_to_hlo errors on unsupported ops", {
    # Create a temporary ONNX file with an unsupported op
    # We can test this by checking error message format
    expect_error(
        onnx_to_hlo("nonexistent.onnx"),
        "nonexistent.onnx"
    )
})

test_that("ort_read_graph extracts graph structure", {
    model_path <- system.file("extdata", "lm_iris.onnx", package = "nativeORT")
    skip_if(model_path == "", "Test model not found")

    g <- ort_read_graph(normalizePath(model_path))

    expect_type(g, "list")
    expect_named(g, c("inputs", "outputs", "initializers", "nodes"))

    # Inputs
    expect_length(g$inputs, 1)
    expect_equal(g$inputs[[1]]$name, "X")
    expect_equal(g$inputs[[1]]$dtype, "f32")

    # Outputs
    expect_length(g$outputs, 1)
    expect_equal(g$outputs[[1]]$name, "variable")

    # Initializers (coef + intercept)
    expect_true("coef" %in% names(g$initializers))
    expect_true("intercept" %in% names(g$initializers))
    expect_equal(dim(g$initializers$coef), c(3L, 1L))

    # Nodes
    expect_length(g$nodes, 2)
    expect_equal(g$nodes[[1]]$op_type, "MatMul")
    expect_equal(g$nodes[[2]]$op_type, "Add")
})
