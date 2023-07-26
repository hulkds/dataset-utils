# pylint: disable=missing-module-docstring, line-too-long

import numpy as np


def xywh2xyxy(bboxes_xywh: np.ndarray):
    """Convert bbox from (x_center y_center width height) to (top_left_x, top_left_y, bottom_right_x, bottom_right_y).

    Args:
        bboxes_xywh (np.ndarray): bounding boxes in format xywh (nx4).

    Returns:
        np.nd_array: bounding boxes in format xyxy (nx4).
    """
    bboxes_xyxy = np.copy(bboxes_xywh)
    bboxes_xyxy[:, 0] = bboxes_xywh[:, 0] - bboxes_xywh[:, 2] / 2  # top left x
    bboxes_xyxy[:, 1] = bboxes_xywh[:, 1] - bboxes_xywh[:, 3] / 2  # top left y
    bboxes_xyxy[:, 2] = bboxes_xywh[:, 0] + bboxes_xywh[:, 2] / 2  # bottom right x
    bboxes_xyxy[:, 3] = bboxes_xywh[:, 1] + bboxes_xywh[:, 3] / 2  # bottom right y

    return bboxes_xyxy


class Colors:
    """Class Colors that help to generate unique colors.

    Attributes:
        _type_: _description_
    """
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        """defaut constructor.
        """
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = (
            "FF3838",
            "FF9D97",
            "FF701F",
            "FFB21D",
            "CFD231",
            "48F90A",
            "92CC17",
            "3DDB86",
            "1A9334",
            "00D4BB",
            "2C99A8",
            "00C2FF",
            "344593",
            "6473FF",
            "0018EC",
            "8438FF",
            "520085",
            "CB38FF",
            "FF95C8",
            "FF37C7",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        """call method.

        Args:
            i (_type_): _description_
            bgr (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(hex_code):  # rgb order (PIL)
        """convert hex colors to rgb colors.

        Args:
            hex_code (str): color hex code.

        Returns:
            tuple: r, g, b values for rgb colors.
        """
        return tuple(int(hex_code[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))
