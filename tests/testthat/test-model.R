test_that("onnx_model loads an ONNX model with metadata", {
    skip_if_not(onnx_is_loaded(), "ONNX Runtime not loaded")

    model_path <- system.file("extdata", "lm_iris.onnx", package = "onnxr")
    skip_if(model_path == "", "Test model not found")

    sess <- onnx_model(model_path)
    expect_s3_class(sess, "onnx_model")
    expect_equal(sess$n_inputs, 1L)
    expect_equal(sess$n_outputs, 1L)
    expect_type(sess$input_names, "character")
    expect_type(sess$output_names, "character")
    expect_type(sess$input_shapes, "list")
    expect_type(sess$output_shapes, "list")
    expect_type(sess$input_types, "integer")
    expect_type(sess$output_types, "integer")
    # LM model: input [?, 3], output [?, 1]
    expect_equal(sess$input_shapes[[1]][2], 3L)
    expect_equal(sess$output_shapes[[1]][2], 1L)
})
