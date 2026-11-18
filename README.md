# Polynomial Regression Lighting AI

A NumPy-based image-processing project that uses a neural network to learn a mapping between a normal image and a corresponding image with adjusted lighting.

The model analyzes local RGB image patches and predicts polynomial coefficients used to transform the center pixel. The resulting image can be adjusted interactively through a visualization interface.

## How It Works

1. **Training Data:** Corresponding normal and target-lit images are used to generate training samples.
2. **Patch Extraction:** Local RGB patches are extracted from the normal image. The corresponding center pixel from the target image serves as the expected output. Input data is expected as shape (image_height, image_width, patch_size, patch_size, 3 [rgb channels])
4. **Neural Network:** A custom neural network predicts five polynomial coefficients for each RGB channel.
5. **Polynomial Transformation:** The predicted coefficients are used to calculate the transformed pixel values.
6. **Inference and Visualization:** The trained model processes new images, and the curve application strength can be adjusted using interactive sliders.

For each color channel, the model uses:

\[
\hat y = ax^3 + bx^2 + cx + d + e\sqrt{|x|+\epsilon}
\]

where \(x\) is the input pixel value and the network predicts the coefficients \(a,b,c,d,e\).

## Project Structure

```text
.
├── configs/              # Configuration files
├── data/                 # Training and inference images
├── experiments/          # Training results and plots
├── models/               # Saved model parameters
├── scripts/
│   ├── train.py          # Model training entry point
│   └── infer.py          # Inference entry point
├── src/
│   └── lighting_ai/
│       ├── config.py
│       ├── data.py
│       ├── model.py
│       ├── training.py
│       ├── inference.py
│       └── visualization.py
├── tests/                # Unit tests
├── pyproject.toml
├── requirements.txt
└── README.md
```

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

The trained model parameters are saved in `models/greenroom.npz`. A training loss plot is saved in `experiments/`.

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
