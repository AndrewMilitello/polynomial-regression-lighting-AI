POLYNOMIAL REGRESSION LIGHTING AI
Works best with:
    -INTENSITY ~ 1.0
    -SHARPNESS ~ 0.25
    - BLENDING ~ 0.75
Have fun playing around with these to create 
different image styles!
The demo shows a white room(wr) in the lighting style 
of a green room(gr)

To use your own image, go to lines 40-42 and replace the
existing file path with the one you want

During inferrence (the else block) THE MODEL WILL ONLY
USE THE FIRST IMAGE (normal) for prediction. It will not
try to match the style of the second image (lighting) 
that you uploaded, it will instead match the style of the
image it was trained on.

In order to make the model predict how one image
will look with the lighting of another image,
delete the existing ai.npz file and retrain with the
chosen normal and lighting image
