import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def visualize(orig_norm, pcolor, predicted_img, boost, height, width,
              train_amplitude=2):

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.3)
    intensity_s = Slider(fig.add_axes([0.25, 0.15, 0.65, 0.03]), 'Intensity', 0.0, 10.0, valinit=0.0)
    sharpness_s = Slider(fig.add_axes([0.25, 0.1, 0.65, 0.03]),'Sharpness', 0.0, 1.0, valinit=0.298)

    blending_s = Slider(fig.add_axes([0.25, 0.05, 0.65, 0.03]),'Blending', 0.0, 1.5, valinit=0.429)

    img_obj = ax.imshow(orig_norm)
    def update(v):
        intensity = intensity_s.val
        sharp = sharpness_s.val
        blend = blending_s.val
        curr_img = predicted_img + pcolor * blend + boost * sharp
        final_img = np.clip(
            (curr_img + (train_amplitude / 2)) / train_amplitude,
            0, 1
        ).reshape(height, width, 3)

        blended = (orig_norm * (1 - intensity)) + (final_img * intensity)
        img_obj.set_data(np.clip(blended, 0, 1))
        fig.canvas.draw_idle()

    intensity_s.on_changed(update)
    sharpness_s.on_changed(update)
    blending_s.on_changed(update)

    plt.show()