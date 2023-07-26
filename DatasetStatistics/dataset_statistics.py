# pylint: disable=missing-module-docstring, line-too-long, import-error, wrong-import-position

import contextlib
import sys
from pathlib import Path
from typing import Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from PIL import Image, ImageDraw

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw

from utils import Colors, xywh2xyxy

colors = Colors()  # create instance for 'from utils.plots import colors'

def plot_labels(labels: np.ndarray, names=Dict[int, str], save_dir=Path("")):
    """plot the labels statistic.

    Args:
        labels (np.ndarray): labels in format (class, x_center y_center width height). Shape (nx5).
        names (dict): class names. Defaults to Dict[int, str].
        save_dir (Path, optional): directory for saving plots. Defaults to Path("").
    """

    # plot dataset labels
    print(f"Plotting labels to {save_dir / 'labels.jpg'}... ")
    classes, boxes = labels[:, 0], labels[:, 1:].transpose()  # classes, boxes
    num_classes = int(classes.max() + 1)  # number of classes
    boxes_data_frame = pd.DataFrame(boxes.transpose(), columns=["x", "y", "width", "height"])

    # seaborn correlogram
    sn.pairplot(boxes_data_frame, corner=True, diag_kind="auto", kind="hist", diag_kws={"bins": 50}, plot_kws={"pmax": 0.9})
    plt.savefig(save_dir / "labels_correlogram.jpg", dpi=200)
    plt.close()

    # matplotlib labels
    matplotlib.use("svg")  # faster
    ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)[1].ravel()
    y = ax[0].hist(classes, bins=np.linspace(0, num_classes, num_classes + 1) - 0.5, rwidth=0.8)
    with contextlib.suppress(Exception):  # color histogram bars by class
        [y[2].patches[i].set_color([boxes_data_frame / 255 for boxes_data_frame in colors(i)]) for i in range(num_classes)]  # known issue #3195
    ax[0].set_ylabel("instances")
    if 0 < len(names) < 30:
        ax[0].set_xticks(range(len(names)))
        ax[0].set_xticklabels(list(names.values()), rotation=90, fontsize=10)
    else:
        ax[0].set_xlabel("classes")
    sn.histplot(boxes_data_frame, x="x", y="y", ax=ax[2], bins=50, pmax=0.9)
    sn.histplot(boxes_data_frame, x="width", y="height", ax=ax[3], bins=50, pmax=0.9)

    # rectangles
    labels[:, 1:3] = 0.5  # center
    labels[:, 1:] = xywh2xyxy(labels[:, 1:]) * 2000
    img = Image.fromarray(np.ones((2000, 2000, 3), dtype=np.uint8) * 255)
    for cls, *box in labels[:1000]:
        ImageDraw.Draw(img).rectangle(box, width=1, outline=colors(cls))  # plot
    ax[1].imshow(img)
    ax[1].axis("off")

    for a in [0, 1, 2, 3]:
        for s in ["top", "right", "left", "bottom"]:
            ax[a].spines[s].set_visible(False)

    plt.savefig(save_dir / "labels.jpg", dpi=200)
    matplotlib.use("Agg")
    plt.close()
