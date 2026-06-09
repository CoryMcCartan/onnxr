# Changelog

## onnxr 0.1.2

CRAN release: 2026-06-08

- Improved handling of external data files
- Added support for boolean tensors as both inputs and outputs
- Better parsing and display of models with optional inputs
- Address CRAN comments on DESCRIPTION and examples

## onnxr 0.1.0

Initial package release. Features:

- Handle ONNX models with arbitrary inputs and outputs
- Installation helper and dynamic loading of runtime library
- Vignette demonstrating end-to-end workflow for object detection image
  model
- Tests verifying agreement between ONNX models exported from sklearn
  and R’s lm() and glm() on the same data
