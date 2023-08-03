import onnxruntime
import cv2
import numpy as np
import time

def load_xseg_model():
    xseg_session = onnxruntime.InferenceSession(
        "./xseg.onnx", providers=["CUDAExecutionProvider"])
    return xseg_session

def infer(xseg, input_image):
        input_shape_len = len(input_image.shape)
        if input_shape_len == 3:
            input_image = input_image[None, ...]

        inputs = {"input": input_image}
        res = xseg.run(None, inputs)[0]
        result = np.clip(res, 0, 1.0)
        result[result < 0.1] = 0  #get rid of noise

        if input_shape_len == 3:
            result = result[0]
        return result

def apply_xseg(xseg, img):
    xseg_res = 256
    img = img.astype(np.float32) / 255.0
    h, w, c = img.shape

    if w != xseg_res:
        img = cv2.resize(img, (xseg_res, xseg_res),
                         interpolation=cv2.INTER_LANCZOS4)

    if len(img.shape) == 2:
        img = img[..., None]
    mask = infer(xseg, img)
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    return mask

if __name__ == "__main__":
    img_file = "1.png"
    img = cv2.imread(img_file)
    xseg = load_xseg_model()

    for i in range(10):
        t0 = time.time()
        mask = apply_xseg(xseg, img)
        t1 = time.time()
        print("time: ", t1 - t0)

    mask[mask == 1] = 255
    cv2.imshow("mask", mask)
    cv2.waitKey(-1)
    cv2.destroyAllWindows()