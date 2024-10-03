import numpy as np
import tensorflow as tf
from tensorflow.keras import backend

def dice_loss(true, pred, smooth):
    true_f = backend.flatten(true)
    pred_f = backend.flatten(pred)

    intersect = backend.sum(true_f * pred_f)
    dice = (2 * intersect + smooth) / (backend.sum(true_f) + backend.sum(pred_f) + smooth)

    return 1 - dice