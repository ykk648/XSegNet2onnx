import numpy as np
import pickle
from pathlib import Path
import cv2

from xseg_lib.core.leras import nn
from xseg_lib.facelib import XSegNet


def get_xseg(model_path=r".\weights"):
    model_path = Path(model_path)
    if not model_path.exists():
        raise ValueError(f'{model_path} not found. Please ensure it exists.')
    nn.initialize_main_env()
    device_config = nn.DeviceConfig().CPU()
    nn.initialize(device_config)
    xseg = XSegNet(name='XSeg',
                   load_weights=True,
                   weights_file_root=model_path,
                   data_format=nn.data_format,
                   raise_on_no_model_files=True)
    return xseg


def apply_xseg(xseg, img):
    xseg_res = xseg.get_resolution()
    img = img.astype(np.float32) / 255.0
    h, w, c = img.shape

    if w != xseg_res:
        img = cv2.resize(img, (xseg_res, xseg_res),
                         interpolation=cv2.INTER_LANCZOS4)

    if len(img.shape) == 2:
        img = img[..., None]
    mask = xseg.extract(img)
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    return mask