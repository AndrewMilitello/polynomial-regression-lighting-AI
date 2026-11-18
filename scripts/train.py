import numpy as np
import matplotlib.pyplot as plt
from lighting_ai.config import *
from lighting_ai.model import Network

loss_hist = []
acc_hist = []
indices = np.arange(training_size)
predictor = Network(lr=LEARNING_RATE, ptcount=bsize)

for e in range(EPOCHS):
    np.random.shuffle(indices)
    x_train, y_train = x_train[indices], y_train[indices]
    total_acc = 0
    bnum = int(np.ceil(len(x_train) / bsize))
    x_batches = np.array_split(x_train, bnum) 
    y_batches = np.array_split(y_train, bnum) 
    for x, y in zip(x_batches, y_batches):
        #forward pass
        #emphasize the center patch pixel that we're trying to predict
        mid_idx = ((sample_size**2) // 2) * 3 
        emphasized_mid_x = np.hstack([x, x[:, mid_idx : mid_idx + 3]*5.0])
        pred = predictor.Predict(emphasized_mid_x)
        #calc_error
        #prediction based on the emphasized center pixel, ignoring all the 'blurry' surrounding pixels
        predicted_x = x[:, mid_idx : mid_idx + 3] 
        #extract coefficients for polynomial regression
        coeffs = pred.reshape(x.shape[0], output_degree, -1) 
        # added e sqrt term to the equation to give model more lighting flexibility 
        a, b, c, d, e = coeffs[:,0,:], coeffs[:,1,:], coeffs[:,2,:], coeffs[:,3,:], coeffs[:,4,:]
        predicted = a * predicted_x**3 + b * predicted_x**2 + c * predicted_x + d + e * np.sqrt(np.abs(predicted_x) + 1e-6)
        
        prediction_error = y - predicted
        point_count = len(x) / 3
        a_contribution, b_contribution, c_contribution, d_contribution, e_contribution = [0, 0, 0, 0, 0]
        mean_absolute_error = np.mean(np.abs(prediction_error))
        error_threshold = 0.002  # threshold to determine whether values need to be updated
        absolute_error = np.abs(prediction_error)
        update_mask = (absolute_error > error_threshold).astype(float)
        update_mask = np.tile(update_mask, (1, output_degree))

        # find output gradients
        a_contribution = prediction_error * (predicted_x**3) * 0.1
        b_contribution = prediction_error * (predicted_x**2) * 0.1
        c_contribution = prediction_error * predicted_x
        d_contribution = prediction_error
        e_contribution = prediction_error * np.sqrt(np.abs(predicted_x) + 1e-6)
        contributions = np.concatenate([a_contribution, b_contribution, c_contribution, d_contribution, e_contribution], axis=1)
        contributions = contributions * update_mask
            
        #update output layer
        batch_size = x.shape[0]
        grad_vec_w = np.dot(predictor.hidden2_output.T, contributions) / batch_size
        grad_vec_b = np.sum(contributions, axis=0, keepdims=True) / batch_size
        predictor.output_layer.vw = 0.9 * predictor.output_layer.vw + predictor.lr * grad_vec_w
        predictor.output_layer.vb = 0.9 * predictor.output_layer.vb + predictor.lr * grad_vec_b
        predictor.output_layer.weights += predictor.output_layer.vw
        predictor.output_layer.bias += predictor.output_layer.vb
            
        #find hidden layer 2 gradients
    
        h2_contributions = np.dot(contributions, predictor.output_layer.weights.T) * np.where(predictor.hidden2_output > 0, 1, 0.01)
        h2_contributions = np.clip(h2_contributions, -1, 1.0)
    
            
        #update hidden layer 2 
        l2_grad_vec_w = np.dot(predictor.hidden_output.T, h2_contributions) / batch_size
        l2_grad_vec_b = np.sum(h2_contributions, axis=0, keepdims=True) / batch_size
        predictor.hidden_layer_2.vw = 0.9 * predictor.hidden_layer_2.vw + predictor.lr * l2_grad_vec_w
        predictor.hidden_layer_2.vb = 0.9 * predictor.hidden_layer_2.vb + predictor.lr * l2_grad_vec_b
            
        predictor.hidden_layer_2.weights += predictor.hidden_layer_2.vw
        predictor.hidden_layer_2.bias += predictor.hidden_layer_2.vb

        #find hidden layer 1 gradients and update
        h1_contributions = np.dot(h2_contributions, predictor.hidden_layer_2.weights.T) * np.where(predictor.hidden_output > 0, 1, 0.01)
        h1_contributions = np.clip(h1_contributions, -1, 1)
        l1_grad_vec_w = np.dot(emphasized_mid_x.T, h1_contributions) / batch_size
        l1_grad_vec_b = np.sum(h1_contributions, axis=0, keepdims=True) / batch_size
        predictor.hidden_layer.vw = 0.9 * predictor.hidden_layer.vw + predictor.lr * l1_grad_vec_w
        predictor.hidden_layer.vb = 0.9 * predictor.hidden_layer.vb + predictor.lr * l1_grad_vec_b
        predictor.hidden_layer.weights += predictor.hidden_layer.vw
        predictor.hidden_layer.bias += predictor.hidden_layer.vb
        predictor.lr *= 0.95
        loss_hist.append(np.mean(prediction_error**2))
        acc_hist.append((100-(mean_absolute_error*100)))
    
    np.savez("model_weights.npz", 
             w1=predictor.hidden_layer.weights, b1=predictor.hidden_layer.bias,
             w2=predictor.hidden_layer_2.weights, b2=predictor.hidden_layer_2.bias,
             w3=predictor.output_layer.weights, b3=predictor.output_layer.bias)
    print("done")
    print(f"acc mean over training {np.mean(acc_hist)}")
    print(f"acc max during training {np.max(acc_hist)}")
    print(f"loss min {np.min(loss_hist)}")
    print(f"loss mean {np.mean(loss_hist)}")


plt.plot(loss_hist)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.grid(True)
plt.savefig("experiments/loss_curve.png", dpi=150, bbox_inches="tight")
plt.show()