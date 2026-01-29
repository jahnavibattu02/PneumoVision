import numpy as np
import tensorflow as tf
import cv2

def find_last_conv_layer(model: tf.keras.Model) -> str:
    # EfficientNetB0 last conv-like layer often ends with "top_conv"
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found for Grad-CAM.")

def make_gradcam_heatmap(model: tf.keras.Model, img_array: np.ndarray, last_conv_layer_name: str):
    """
    img_array: shape (1, H, W, 3) scaled 0..1
    returns heatmap 0..1 with shape (H, W)
    """
    grad_model = tf.keras.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)
        # Binary model -> model.output is sigmoid
        loss = preds[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_heatmap_on_image(img_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45):
    """
    img_bgr: original image BGR uint8
    heatmap: 0..1 float (H,W) to be resized to img size
    """
    h, w = img_bgr.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_color, alpha, 0)
    return overlay, heatmap_resized

def threshold_mask(heatmap: np.ndarray, thresh: float = 0.55):
    mask = (heatmap >= thresh).astype(np.uint8) * 255
    return mask
