# https://zhuanlan.zhihu.com/p/402074214
# https://blog.csdn.net/weixin_43134049/article/details/124752259
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import time

logger = trt.Logger(trt.Logger.VERBOSE)


def serialize_xseg(model_file="xseg.sim.onnx"):
    with trt.Builder(logger) as builder, builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    ) as network, trt.OnnxParser(network, logger) as parser:
        if not parser.parse_from_file(model_file):
            for idx in range(parser.num_errors):
                print(parser.get_error(idx))
    config = builder.create_builder_config()
    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1, 256, 256, 3), (1, 256, 256, 3),
                      (1, 256, 256, 3))
    config.add_optimization_profile(profile)
    serialized_engine = builder.build_serialized_network(network, config)
    with open("xseg.engine", "wb") as f:
        f.write(serialized_engine)


def serialize_inswapper(model_file="inswapper_128.fp16.sim.onnx"):
    with trt.Builder(logger) as builder, builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    ) as network, trt.OnnxParser(network, logger) as parser:
        if not parser.parse_from_file(model_file):
            for idx in range(parser.num_errors):
                print(parser.get_error(idx))

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    serialized_engine = builder.build_serialized_network(network, config)

    with open("inswapper_128.engine", "wb") as f:
        f.write(serialized_engine)


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    context.execute_async(batch_size=batch_size,
                           bindings=bindings,
                           stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()
    return [out.host for out in outputs]


class HostDeviceMem(object):

    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine: trt.ICudaEngine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        hostmem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(hostmem.nbytes)
        bindings.append(int(device_mem))

        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(hostmem, device_mem))
        else:
            outputs.append(HostDeviceMem(hostmem, device_mem))
    return inputs, outputs, bindings, stream


def preprocess(img):
    xseg_res = 256
    img = img.astype(np.float32) / 255.0
    h, w, c = img.shape

    if w != xseg_res:
        img = cv2.resize(img, (xseg_res, xseg_res),
                         interpolation=cv2.INTER_LANCZOS4)

    if len(img.shape) == 2:
        img = img[..., None]
    return img


def postprocess(mask: np.ndarray):
    xseg_res = 256
    mask = np.reshape(mask, (xseg_res, xseg_res, 1))
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    return mask



def test_xseg():
    img_file = "1.png"
    img = cv2.imread(img_file)
    img = preprocess(img)

    with open("xseg.engine", "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    inputs[0].host = img

    t0 = time.time()
    mask = do_inference(context, bindings, inputs, outputs, stream)[0]
    t1 = time.time()
    print("time: ", t1 - t0)
    mask = postprocess(mask)

    mask[mask == 1] = 255
    cv2.imshow("mask", mask)
    cv2.waitKey(-1)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_xseg()
