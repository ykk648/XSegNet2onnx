import onnx
import torch
from onnx2torch import convert
import numpy as np

# Path to ONNX model
onnx_model_path = "./xseg.onnx"
# You can pass the path to the onnx model to convert it or...
torch_model_1 = convert(onnx_model_path)

import onnxruntime as ort

# Create example data
x = torch.ones((1, 256, 256, 3))

out_torch = torch_model_1(x)

# Save the PyTorch model
torch_model_save_path = "./xseg_torch.pth"
torch.save(torch_model_1, torch_model_save_path)
print(f"PyTorch model saved to {torch_model_save_path}")

ort_sess = ort.InferenceSession(onnx_model_path)
outputs_ort = ort_sess.run(None, {"input": x.numpy()})

# Check the Onnx output against PyTorch
print(torch.max(torch.abs(torch.tensor(outputs_ort)- out_torch.detach().numpy())))
print(np.allclose(outputs_ort, out_torch.detach().numpy(), atol=1.0e-7))