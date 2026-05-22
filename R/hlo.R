#' Convert an ONNX model to a StableHLO function
#'
#' Reads an `.onnx` model file and translates its computation graph into
#' StableHLO operations using the `stablehlo` R package. The resulting
#' function can be serialized with [stablehlo::repr()] and compiled with
#' `pjrt`.
#'
#' @param path Path to an `.onnx` model file.
#' @param batch_size Integer batch size to use for dynamic dimensions.
#'   StableHLO requires static shapes, so dynamic ONNX dimensions (marked
#'   as -1) are replaced with this value. Defaults to `1L`.
#'
#' @returns A `stablehlo` `Func` object representing the model computation.
#' @export
#'
#' @examples \dontrun{
#' func <- onnx_to_hlo(system.file("extdata", "lm_iris.onnx", package = "onnxr"))
#' cat(stablehlo::repr(func))
#' }
onnx_to_hlo <- function(path, batch_size = 1L) {
    if (!requireNamespace("stablehlo", quietly = TRUE)) {
        stop("The 'stablehlo' package is required. ",
            "Install with: pak::pak('r-xla/stablehlo')")
    }

    path <- normalizePath(path, mustWork = TRUE)
    batch_size <- as.integer(batch_size)

    # Find external data files
    model_dir <- dirname(path)
    ext_data <- list.files(model_dir, pattern = "[.]onnx_data$", full.names = TRUE)

    # Parse the ONNX graph
    graph <- if (length(ext_data) > 0) {
        ort_read_graph(path, ext_data)
    } else {
        ort_read_graph(path)
    }

    # Create module and function
    stablehlo::hlo_module()
    func <- stablehlo::local_func()

    # Edge name → FuncValue mapping
    values <- list()
    # Edge name → shape mapping
    edge_shapes <- list()

    # Declare inputs
    for (inp in graph$inputs) {
        shape <- as.integer(inp$shape)
        shape[shape < 0L] <- batch_size
        values[[inp$name]] <- stablehlo::hlo_input(inp$name, inp$dtype, shape = shape)
        edge_shapes[[inp$name]] <- shape
    }

    # Load initializers as constants
    for (nm in names(graph$initializers)) {
        init <- graph$initializers[[nm]]
        if (length(init) == 0) next
        dtype <- attr(init, "dtype") %||% "f32"
        shape <- dim(init)
        if (is.null(shape)) shape <- length(init)
        values[[nm]] <- stablehlo::hlo_tensor(as.numeric(init), dtype = dtype,
            shape = as.integer(shape))
        edge_shapes[[nm]] <- as.integer(shape)
    }

    # Walk nodes topologically
    for (node in graph$nodes) {
        op <- node$op_type
        handler <- onnx_ops[[op]]
        if (is.null(handler)) {
            stop(sprintf("Unsupported ONNX op: '%s'. ", op),
                "This op has no StableHLO translation implemented yet.")
        }

        # Resolve input edges to FuncValues
        node_inputs <- lapply(node$inputs, function(nm) {
            if (nm == "") return(NULL)
            if (is.null(values[[nm]])) {
                stop(sprintf("Edge '%s' not found (required by %s node '%s').",
                    nm, op, node$name))
            }
            values[[nm]]
        })
        node_inputs <- Filter(Negate(is.null), node_inputs)

        # Resolve input shapes
        node_shapes <- lapply(node$inputs, function(nm) {
            if (nm == "") return(NULL)
            edge_shapes[[nm]]
        })
        node_shapes <- Filter(Negate(is.null), node_shapes)

        # Execute handler
        result <- handler(node_inputs, node$attrs, node_shapes)

        # Store outputs
        if (length(node$outputs) == 1) {
            values[[node$outputs[1]]] <- result
            edge_shapes[[node$outputs[1]]] <- infer_output_shape(
                op, node_shapes, node$attrs)
        } else {
            # Multi-output node (e.g., Split)
            for (i in seq_along(node$outputs)) {
                values[[node$outputs[i]]] <- result[[i]]
                edge_shapes[[node$outputs[i]]] <- infer_multi_output_shape(
                    op, node_shapes, node$attrs, i)
            }
        }
    }

    # Return graph outputs
    output_vals <- lapply(graph$outputs, function(out) {
        if (is.null(values[[out$name]])) {
            stop(sprintf("Output edge '%s' not found.", out$name))
        }
        values[[out$name]]
    })
    do.call(stablehlo::hlo_return, output_vals)
}

# Output shape inference for single-output ops
infer_output_shape <- function(op, input_shapes, attrs) {
    switch(op,
        # Binary with broadcasting
        Add = , Sub = , Mul = , Div = , Pow = , Mod = , Max = , Min = {
            if (length(input_shapes) >= 2) {
                broadcast_shape(input_shapes[[1]], input_shapes[[2]])
            } else {
                input_shapes[[1]]
            }
        },
        # Unary (shape preserved)
        Sigmoid = , Tanh = , Relu = , LeakyRelu = , Neg = , Abs = , Sqrt = ,
        Exp = , Log = , Ceil = , Floor = , Reciprocal = , Clip = ,
        BatchNormalization = {
            input_shapes[[1]]
        },
        MatMul = {
            a <- input_shapes[[1]]
            b <- input_shapes[[2]]
            c(a[-length(a)], b[length(b)])
        },
        Gemm = {
            a <- input_shapes[[1]]
            b <- input_shapes[[2]]
            transA <- attrs$transA %||% 0L
            transB <- attrs$transB %||% 0L
            m <- if (transA == 1L) a[2] else a[1]
            n <- if (transB == 1L) b[1] else b[2]
            c(m, n)
        },
        Conv = {
            infer_conv_output_shape(
                input_shapes[[1]], input_shapes[[2]],
                as.integer(attrs$strides %||% rep(1L, length(input_shapes[[1]]) - 2L)),
                as.integer(attrs$pads %||% rep(0L, 2L * (length(input_shapes[[1]]) - 2L))),
                as.integer(attrs$dilations %||% rep(1L, length(input_shapes[[1]]) - 2L)),
                as.integer(attrs$group %||% 1L))
        },
        MaxPool = , AveragePool = {
            infer_pool_output_shape(
                input_shapes[[1]], as.integer(attrs$kernel_shape),
                as.integer(attrs$strides %||% rep(1L, length(attrs$kernel_shape))),
                as.integer(attrs$pads %||% rep(0L, 2L * length(attrs$kernel_shape))))
        },
        GlobalAveragePool = {
            shape <- input_shapes[[1]]
            shape[3:length(shape)] <- 1L
            shape
        },
        Transpose = {
            perm <- attrs$perm
            if (is.null(perm)) rev(input_shapes[[1]])
            else input_shapes[[1]][as.integer(perm) + 1L]
        },
        Reshape = , Flatten = , Unsqueeze = , Squeeze = {
            # Shape changes — hard to infer without resolving constants
            # Return a placeholder; may need refinement
            input_shapes[[1]]
        },
        Concat = {
            axis <- as.integer(attrs$axis) + 1L  # 1-indexed for R
            shape <- input_shapes[[1]]
            for (i in seq_along(input_shapes)[-1]) {
                shape[axis] <- shape[axis] + input_shapes[[i]][axis]
            }
            shape
        },
        Slice = {
            # Approximate: same shape
            input_shapes[[1]]
        },
        Softmax = {
            input_shapes[[1]]
        },
        Cast = {
            input_shapes[[1]]
        },
        Tile = {
            input_shapes[[1]]  # approximate
        },
        ReduceSum = , ReduceMax = , ReduceMean = {
            input_shapes[[1]]  # with keepdims=1, shape is preserved
        },
        Gather = , GatherElements = {
            input_shapes[[1]]  # approximate
        },
        Resize = {
            input_shapes[[1]]  # approximate; actual size depends on scales/sizes input
        },
        Pad = {
            input_shapes[[1]]  # approximate
        },
        Constant = {
            integer(0)  # scalar by default
        },
        # Default
        input_shapes[[1]]
    )
}

# Shape inference for multi-output ops
infer_multi_output_shape <- function(op, input_shapes, attrs, output_index) {
    switch(op,
        Split = {
            shape <- input_shapes[[1]]
            axis <- as.integer(attrs$axis %||% 0L) + 1L
            split_sizes <- as.integer(attrs$split)
            shape[axis] <- split_sizes[output_index]
            shape
        },
        TopK = {
            # Both outputs (values, indices) have the same shape
            input_shapes[[1]]  # approximate
        },
        # Default: same as input
        input_shapes[[1]]
    )
}
