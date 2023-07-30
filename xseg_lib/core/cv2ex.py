import cv2
import numpy as np
from pathlib import Path
from xseg_lib.core.interact import interact as io
from xseg_lib.core.imagelib import normalize_channels
import traceback


def cv2_imread(filename,
               flags=cv2.IMREAD_UNCHANGED,
               loader_func=None,
               verbose=True):
    """
    allows to open non-english characters path
    """
    try:
        if loader_func is not None:
            bytes = bytearray(loader_func(filename))
        else:
            with open(filename, "rb") as stream:
                bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        return cv2.imdecode(numpyarray, flags)
    except:
        if verbose:
            io.log_err(
                f"Exception occured in cv2_imread : {traceback.format_exc()}")
        return None


def cv2_imwrite(filename, img, *args):
    ret, buf = cv2.imencode(Path(filename).suffix, img, *args)
    if ret == True:
        try:
            with open(filename, "wb") as stream:
                stream.write(buf)
        except:
            pass


def cv2_resize(x, *args, **kwargs):
    h, w, c = x.shape
    x = cv2.resize(x, *args, **kwargs)

    x = normalize_channels(x, c)
    return x
