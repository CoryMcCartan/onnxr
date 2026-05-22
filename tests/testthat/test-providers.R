# Detect which non-CPU backends should be available on this platform.
# The standard ORT builds from onnx_install() include:
#   - CoreML on macOS (arm64 and x86_64)
# Other backends (CUDA, XNNPACK, OpenVINO) require custom ORT builds.
available_backends <- function() {
    backends <- character()
    sysname <- Sys.info()[["sysname"]]

    if (sysname == "Darwin") {
        backends <- c(backends, "coreml")
    }

    backends
}

test_that("expected backends are detected for this platform", {
    skip_if_not(onnx_is_loaded(), "ONNX Runtime not loaded")

    backends <- available_backends()
    sysname <- Sys.info()[["sysname"]]

    if (sysname == "Darwin") {
        expect_true("coreml" %in% backends)
    } else {
        expect_length(backends, 0)
    }
})

# Run both example models on each available non-CPU backend
for (prov in available_backends()) {
    test_that(paste0("lm_iris predictions match with backend=", prov), {
        skip_if_not(onnx_is_loaded(), "ONNX Runtime not loaded")

        model_path <- system.file("extdata", "lm_iris.onnx", package = "onnxr")
        skip_if(model_path == "", "Test model not found")

        sess <- onnx_model(model_path, backend = prov)
        expect_equal(sess$backend, prov)

        r_model <- lm(Petal.Width ~ Sepal.Length + Sepal.Width + Petal.Length, data = iris)
        r_pred <- unname(predict(r_model))

        input_mat <- as.matrix(iris[, c("Sepal.Length", "Sepal.Width", "Petal.Length")])
        onnx_pred <- onnx_run(sess, input_mat, simplify = TRUE)

        expect_equal(as.numeric(onnx_pred), r_pred, tolerance = 1e-5)
    })

    test_that(paste0("glm_iris predictions match with backend=", prov), {
        skip_if_not(onnx_is_loaded(), "ONNX Runtime not loaded")

        model_path <- system.file("extdata", "glm_iris.onnx", package = "onnxr")
        skip_if(model_path == "", "Test model not found")

        sess <- onnx_model(model_path, backend = prov)
        expect_equal(sess$backend, prov)

        iris_sub <- iris[iris$Species != "setosa", ]
        y <- as.integer(iris_sub$Species == "virginica")

        r_model <- glm(
            y ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width,
            data = iris_sub,
            family = binomial
        )
        r_probs <- unname(predict(r_model, type = "response"))

        input_mat <- as.matrix(iris_sub[, 1:4])
        result <- onnx_run(sess, input_mat)

        expect_type(result, "list")
        expect_named(result, c("label", "probabilities"))

        onnx_p1 <- result$probabilities[, 2]
        expect_equal(as.numeric(onnx_p1), r_probs, tolerance = 1e-3)
    })
}
