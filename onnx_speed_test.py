# -- coding: utf-8 --
# @Time : 2025/4/9
# @Author : ykk648
from pathlib import Path
from apstone import ONNXModel
import onnxruntime

onnxruntime.preload_dlls()


onnx_p = str(Path("./xseg.onnx"))
input_dynamic_shape = [(1, 256, 256, 3)]

# # cpu
# ONNXModel(onnx_p, provider='cpu', debug=True, input_dynamic_shape=input_dynamic_shape).speed_test()

# # gpu
ONNXModel(onnx_p, provider='gpu', debug=True, input_dynamic_shape=input_dynamic_shape).speed_test()