import numpy as np

from lighting_ai.config import output_degree, sample_size, train_amplitude
from lighting_ai.data import extract_normal_patches
from lighting_ai.model import Network


def infer(normal, weights_path="models/trained_model.npz", batch_size=50):
    predictor = Network(lr=0.0005, ptcount=1)
    weights = np.load(weights_path)
    predictor.hidden_layer.weights = weights["w1"]
    predictor.hidden_layer.bias = weights["b1"]
    predictor.hidden_layer_2.weights = weights["w2"]
    predictor.hidden_layer_2.bias = weights["b2"]
    predictor.output_layer.weights = weights["w3"]
    predictor.output_layer.bias = weights["b3"]

    orig_norm = (np.asarray(normal) / 255.0).astype(np.float32)
    height, width = orig_norm.shape[:2]

    scaled_image, center = extract_normal_patches(normal, sample_size)
    scaled_image = np.hstack([scaled_image, scaled_image[:, center:center + 3] * 5.0,])

    preds = np.concatenate([
        predictor(scaled_image[i:i + batch_size])
        for i in range(0, len(scaled_image), batch_size)
    ])

    coeffs = preds.reshape(height * width, output_degree, 3)
    a, b, c, d, e = (coeffs[:, i] for i in range(output_degree))
    pcolor = scaled_image[:, center:center + 3]

    predicted_img = (a * pcolor**3 + b * pcolor**2 + c * pcolor + d + e * np.sqrt(np.abs(pcolor) + 0.00001))
    boost = pcolor - predicted_img

    return orig_norm, pcolor, predicted_img, boost, height, width