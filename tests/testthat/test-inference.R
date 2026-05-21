skip_if_no_ort <- function() {
    skip_if_not(ort_is_loaded(), "ONNX Runtime not loaded")
}

test_that("ort_session loads an ONNX model", {
    skip_if_no_ort()

    model_path <- system.file("extdata", "lm_iris.onnx", package = "nativeORT")
    skip_if(model_path == "", "Test model not found")

    sess <- ort_session(model_path)
    expect_s3_class(sess, "ort_session")
    expect_equal(sess$n_inputs, 1L)
    expect_true(sess$n_outputs >= 1L)
    expect_type(sess$input_names, "character")
    expect_type(sess$output_names, "character")
})

test_that("linear regression predictions match lm()", {
    skip_if_no_ort()

    model_path <- system.file("extdata", "lm_iris.onnx", package = "nativeORT")
    skip_if(model_path == "", "Test model not found")

    sess <- ort_session(model_path)

    # Fit equivalent R model: Petal.Width ~ Sepal.Length + Sepal.Width + Petal.Length
    r_model <- lm(Petal.Width ~ Sepal.Length + Sepal.Width + Petal.Length, data = iris)
    r_pred <- unname(predict(r_model))

    # Input matrix: columns in same order as Python training
    input_mat <- as.matrix(iris[, c("Sepal.Length", "Sepal.Width", "Petal.Length")])

    onnx_pred <- ort_infer_raw(sess, input_mat)

    expect_equal(as.numeric(onnx_pred), r_pred, tolerance = 1e-5)
})

test_that("logistic regression probabilities match glm()", {
    skip_if_no_ort()

    model_path <- system.file("extdata", "glm_iris.onnx", package = "nativeORT")
    skip_if(model_path == "", "Test model not found")

    sess <- ort_session(model_path)

    # Filter to versicolor + virginica (same as Python training)
    iris_sub <- iris[iris$Species != "setosa", ]
    # Recode: versicolor=0, virginica=1 (matches Python encoding)
    y <- as.integer(iris_sub$Species == "virginica")

    # Fit equivalent R model
    r_model <- glm(
        y ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width,
        data = iris_sub,
        family = binomial
    )
    r_probs <- unname(predict(r_model, type = "response"))

    # Input matrix: all 4 features
    input_mat <- as.matrix(iris_sub[, 1:4])

    # Use ort_run directly to get the probabilities output (second output)
    onnx_probs <- ort_run(
        session_ptr = sess$ptr,
        input_array = as.numeric(input_mat),
        input_shape = as.integer(dim(input_mat)),
        input_name = sess$input_names[1],
        output_name = sess$output_names[2] # "probabilities"
    )

    # col 1 = P(versicolor), col 2 = P(virginica)
    onnx_p1 <- onnx_probs[, 2]

    # Wider tolerance: sklearn L-BFGS vs R IRLS may differ slightly
    expect_equal(as.numeric(onnx_p1), r_probs, tolerance = 1e-3)
})
