import cv2
import numpy as np
import time


from xseg_lib import get_xseg, apply_xseg


if __name__ == "__main__":
    img_file = "weights/tgt.jpg"
    img = cv2.imread(img_file)
    xseg = get_xseg()

    t0 = time.time()
    mask = apply_xseg(xseg, img)
    t1 = time.time()
    print("time: ", t1 - t0)

    mask[mask == 1] = 255
    # cv2.imshow("mask", mask)
    # cv2.waitKey(-1)
    # cv2.destroyAllWindows()
    cv2.imwrite('mask.png', mask)
