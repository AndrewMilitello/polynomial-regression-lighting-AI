# Polynomial Regression Lighting AI

A NumPy-based image-processing project that uses a neural network to learn a mapping between a normal image and a corresponding image with adjusted lighting. I made this in February-March of 2026.

The model analyzes local RGB image patches and predicts polynomial coefficients used to transform the center pixel. The resulting image can be adjusted interactively through a visualization interface.

## How It Works

1. **Training Data:** Corresponding normal and target-lit images are used to generate training samples.
2. **Patch Extraction:** Local RGB patches are extracted from the normal image. The corresponding center pixel from the target image serves as the expected output.
4. **Neural Network:** A custom neural network predicts five polynomial coefficients for each RGB channel.
5. **Polynomial Transformation:** The predicted coefficients are used to calculate the transformed pixel values.
6. **Inference and Visualization:** The trained model processes new images, and the curve application strength can be adjusted using interactive sliders.

For each color channel, the model uses:

$$ \hat{y} = ax^3 + bx^2 + cx + d + e\sqrt{|x|+\epsilon} $$

where \(x\) is the input pixel value and the network predicts ($$ \hat{y} $$) contains the coefficients \(a,b,c,d,e\).

## Visualization Controls

Intensity: Controls the overall strength of the lighting effect by blending the processed image with the original. At 0, the original image is displayed; increasing the value makes the lighting effect more prominent. Best near 1.0.
Blending: Controls how much of the model's predicted lighting is added to the original pixel values. Higher values increase the contribution of the predicted lighting. Best near 0.25.
Sharpness: Controls the contribution of the lighting boost, which is calculated from the difference between the original pixel values and the model's predictions. Increasing sharpness emphasizes this difference, making the lighting effect more pronounced. Best near 0.75.
## Installation

Clone the repository:

```bash
git clone https://github.com/AndrewMilitello/polynomial-regression-lighting-AI.git
cd polynomial-regression-lighting-AI
```

Create and activate a virtual environment:

**Windows:**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

Install the project:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

## Training

Place the paired training images in the `data/` directory:

- `data/normal.png` — the original input image.
- `data/target.png` — the corresponding target-lit image.

Then run:

```bash
python scripts/train.py
```

The trained model parameters are saved in `models/trained_model.npz`.

## Inference

After training, place an input image in the data directory and run:

```bash
python scripts/infer.py
```

The visualization interface provides controls for adjusting:

- **Intensity** 
- **Sharpness**
- **Blending**

## Configuration

The model architecture and training parameters can be modified in `src/lighting_ai/config.py`.

These include the hidden layer sizes, patch dimensions, number of training samples, learning rate, number of epochs, and image scaling.

## Implementation Notes

The network and its parameter updates are implemented using NumPy. Backpropagation contributions are manually calculated rather than delegated to an automatic differentiation framework.

The project is an experimental implementation. Its performance and generalization depend on the training image pair and configuration. 
