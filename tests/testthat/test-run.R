test_that("linear regression predictions match lm()", {
    skip_if_not(ort_is_loaded(), "ONNX Runtime not loaded")

    model_path <- system.file("extdata", "lm_iris.onnx", package = "nativeORT")
    skip_if(model_path == "", "Test model not found")

    sess <- ort_session(model_path)

    # Fit equivalent R model: Petal.Width ~ Sepal.Length + Sepal.Width + Petal.Length
    r_model <- lm(Petal.Width ~ Sepal.Length + Sepal.Width + Petal.Length, data = iris)
    r_pred <- unname(predict(r_model))

    # Input matrix: columns in same order as Python training
    input_mat <- as.matrix(iris[, c("Sepal.Length", "Sepal.Width", "Petal.Length")])

    onnx_pred <- ort_run(sess, input_mat)

    expect_equal(as.numeric(onnx_pred), r_pred, tolerance = 1e-5)
})

test_that("logistic regression returns named list with correct types", {
    skip_if_not(ort_is_loaded(), "ONNX Runtime not loaded")

    model_path <- system.file("extdata", "glm_iris.onnx", package = "nativeORT")
    skip_if(model_path == "", "Test model not found")

    sess <- ort_session(model_path)

    iris_sub <- iris[iris$Species != "setosa", ]
    input_mat <- as.matrix(iris_sub[, 1:4])

    result <- ort_run(sess, input_mat)

    # Multi-output model returns a named list
    expect_type(result, "list")
    expect_named(result, c("label", "probabilities"))

    # Labels are integers (int64 output)
    expect_type(result$label, "integer")

    # Probabilities are doubles with correct shape
    expect_type(result$probabilities, "double")
    expect_equal(dim(result$probabilities), c(100L, 2L))
})

test_that("logistic regression probabilities match glm()", {
    skip_if_not(ort_is_loaded(), "ONNX Runtime not loaded")

    model_path <- system.file("extdata", "glm_iris.onnx", package = "nativeORT")
    skip_if(model_path == "", "Test model not found")

    sess <- ort_session(model_path)

    iris_sub <- iris[iris$Species != "setosa", ]
    y <- as.integer(iris_sub$Species == "virginica")

    r_model <- glm(
        y ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width,
        data = iris_sub,
        family = binomial
    )
    r_probs <- unname(predict(r_model, type = "response"))

    input_mat <- as.matrix(iris_sub[, 1:4])
    result <- ort_run(sess, input_mat)

    # col 2 = P(virginica)
    onnx_p1 <- result$probabilities[, 2]

    # Wider tolerance: sklearn L-BFGS vs R IRLS may differ slightly
    expect_equal(as.numeric(onnx_p1), r_probs, tolerance = 1e-3)
})

test_that("input validation catches wrong dimensions", {
    skip_if_not(ort_is_loaded(), "ONNX Runtime not loaded")

    model_path <- system.file("extdata", "lm_iris.onnx", package = "nativeORT")
    skip_if(model_path == "", "Test model not found")

    sess <- ort_session(model_path)

    # Wrong number of columns (4 instead of 3)
    bad_input <- matrix(1:8, nrow = 2, ncol = 4)
    expect_error(ort_run(sess, bad_input), "dimension 2 is 4 but model expects 3")

    # Wrong number of dimensions (3D instead of 2D)
    bad_input_3d <- array(1:24, dim = c(2, 3, 4))
    expect_error(ort_run(sess, bad_input_3d), "3 dimensions but model expects 2")
})
