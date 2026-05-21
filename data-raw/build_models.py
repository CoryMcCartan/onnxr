#!/usr/bin/env python3
"""Build ONNX test models for nativeORT package.

Requirements: numpy, scikit-learn, skl2onnx, onnxruntime
    pip install numpy scikit-learn skl2onnx onnxruntime
"""

import os
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression, LogisticRegression
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "inst", "extdata")
os.makedirs(OUTDIR, exist_ok=True)

iris = load_iris()
X_all = iris.data  # [150, 4]: Sepal.Length, Sepal.Width, Petal.Length, Petal.Width

# --------------------------------------------------------------------------
# Model 1: Linear regression — predict Petal.Width from the other 3 features
# R equivalent: lm(Petal.Width ~ Sepal.Length + Sepal.Width + Petal.Length)
# --------------------------------------------------------------------------
X_lm = X_all[:, :3].astype(np.float32)  # Sepal.Length, Sepal.Width, Petal.Length
y_lm = X_all[:, 3].astype(np.float32)   # Petal.Width

lr = LinearRegression()
lr.fit(X_lm, y_lm)

onnx_lr = convert_sklearn(
    lr, "lm_iris",
    initial_types=[("X", FloatTensorType([None, 3]))],
    target_opset=17,
)
path_lm = os.path.join(OUTDIR, "lm_iris.onnx")
with open(path_lm, "wb") as f:
    f.write(onnx_lr.SerializeToString())
print(f"Wrote {path_lm}")

# --------------------------------------------------------------------------
# Model 2: Logistic regression — classify versicolor (1) vs virginica (2)
# Drop setosa; recode versicolor=0, virginica=1
# R equivalent: glm(species ~ ., family = binomial) on versicolor+virginica
# --------------------------------------------------------------------------
mask = iris.target != 0  # drop setosa
X_glm = X_all[mask].astype(np.float32)
y_glm = (iris.target[mask] - 1).astype(np.int64)  # versicolor=0, virginica=1

glm = LogisticRegression(C=1e10, solver="lbfgs", max_iter=10000)
glm.fit(X_glm, y_glm)

onnx_glm = convert_sklearn(
    glm, "glm_iris",
    initial_types=[("X", FloatTensorType([None, 4]))],
    options={type(glm): {"zipmap": False}},
    target_opset=17,
)
path_glm = os.path.join(OUTDIR, "glm_iris.onnx")
with open(path_glm, "wb") as f:
    f.write(onnx_glm.SerializeToString())
print(f"Wrote {path_glm}")

# --------------------------------------------------------------------------
# Verify models load and run in onnxruntime
# --------------------------------------------------------------------------
import onnxruntime as ort

for name, X_test in [("lm_iris.onnx", X_lm[:5]), ("glm_iris.onnx", X_glm[:5])]:
    sess = ort.InferenceSession(os.path.join(OUTDIR, name))
    inputs = {sess.get_inputs()[0].name: X_test}
    outputs = sess.run(None, inputs)
    print(f"\n{name}:")
    print(f"  inputs:  {[(i.name, i.shape, i.type) for i in sess.get_inputs()]}")
    print(f"  outputs: {[(o.name, o.shape, o.type) for o in sess.get_outputs()]}")
    for i, o in enumerate(outputs):
        print(f"  output[{i}] shape={np.array(o).shape}, first values={np.array(o).flat[:5]}")
