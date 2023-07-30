import numpy as np
import cv2


def draw_polygon(image, points, color, thickness=1):
    points_len = len(points)
    for i in range(0, points_len):
        p0 = tuple(points[i])
        p1 = tuple(points[(i + 1) % points_len])
        cv2.line(image, p0, p1, color, thickness=thickness)


def draw_rect(image, rect, color, thickness=1):
    l, t, r, b = rect
    draw_polygon(image, [(l, t), (r, t), (r, b), (l, b)], color, thickness)


def normalize_channels(img, target_channels):
    img_shape_len = len(img.shape)
    if img_shape_len == 2:
        h, w = img.shape
        c = 0
    elif img_shape_len == 3:
        h, w, c = img.shape
    else:
        raise ValueError("normalize: incorrect image dimensions.")

    if c == 0 and target_channels > 0:
        img = img[..., np.newaxis]
        c = 1

    if c == 1 and target_channels > 1:
        img = np.repeat(img, target_channels, -1)
        c = target_channels

    if c > target_channels:
        img = img[..., 0:target_channels]
        c = target_channels

    return img