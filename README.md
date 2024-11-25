# XSegNet2onnx
convert DeepFaceLab XSegNet's *.npy weights to onnx file. Inference time is optimized from 500ms to 90ms.

# env
```shell
# test on win10 py3.10.14
pip install -r requirements.txt
# remove ./saved_model/.gitkeep for stupid tf
```


# usage
## generate you own onnx file
1. put DeepFaceLab XSegNet weights to weights folder, such as `weights/XSeg_256.npy`.
3. Then run `python test_seg.py` to generate tensorflow SavedModel format checkpoint file to `saved_model` directory.
5. convert model to onnx file, `python -m tf2onnx.convert --saved-model ./saved_model/ --output xseg.onnx  --tag serve`.
6. (optinal) install onnxsim `pip install onnxsim` and run `onnxsim ./xseg.onnx ./xseg.sim.onnx`.

## use onnx file to predict
see `test_seg_onnx.py`.

# issue
Because of [Conv2d_transpose requires asymmetric padding which the CUDA EP currently does not support #11312](https://github.com/microsoft/onnxruntime/issues/11312), XSegNet OnnxRuntime Conv2d_transpose layer does not support CudaExcuation.

