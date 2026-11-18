import sys
from pathlib import Path

import numpy as np
from PIL import Image

from lighting_ai.inference import infer
from lighting_ai.visualization import visualize


if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "data/normal.png"

    image = Image.open(image_path).convert("RGB")
    image = np.array(image)

    orig_norm, pcolor, predicted_img, boost, height, width = infer(image, weights_path="models/trained_model.npz")
    visualize(orig_norm, pcolor, predicted_img, boost, height, width)