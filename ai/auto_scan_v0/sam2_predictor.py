import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
from PIL import Image  # Make sure this is at the top if not already
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class FastSAMPredictor:
    def __init__(self, model_cfg, checkpoint, device='cpu'):
        self.device = device
        model = build_sam2(model_cfg, checkpoint, device=device)
        self.predictor = SAM2ImagePredictor(model)

    def set_image(self, image):
        # keep original BGR for overlay & color pulls
        self.original = image.copy()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(rgb)

    def segment_points(self, points):
        # points: list of (x,y) in original-image coords
        coords = np.array(points)
        labels = np.ones(len(points), dtype=int)
        masks, scores, logits = self.predictor.predict(
            point_coords=coords,
            point_labels=labels,
            multimask_output=False
        )
        mask = masks[0].astype(bool)           # boolean mask
        overlay = self.original.copy()
        overlay[~mask] = (overlay[~mask] * 0.4).astype(np.uint8)
        return overlay, mask