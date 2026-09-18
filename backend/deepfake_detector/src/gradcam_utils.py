import cv2
import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from model import XceptionDetector

class _BinaryLogitWrapper(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).unsqueeze(1)

class GradCAMExplainer:

    def __init__(self, model: XceptionDetector, device='cpu'):
        self.model = model
        self.device = device
        wrapped = _BinaryLogitWrapper(model)
        target_layers = [model.backbone.conv4]
        self.cam = GradCAM(model=wrapped, target_layers=target_layers)

    def generate(self, img_tensor, rgb_img_float):
        targets = [ClassifierOutputTarget(0)]
        grayscale_cam = self.cam(input_tensor=img_tensor, targets=targets)[0]
        visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)
        return visualization

def save_heatmap(overlay_array, out_path):
    bgr = cv2.cvtColor(overlay_array, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, bgr)
    return out_path
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = XceptionDetector(pretrained=True).to(device).eval()
    explainer = GradCAMExplainer(model, device=device)
    dummy_rgb = np.random.rand(224, 224, 3).astype(np.float32)
    dummy_tensor = torch.from_numpy(dummy_rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
    overlay = explainer.generate(dummy_tensor, dummy_rgb)
    save_heatmap(overlay, 'test_heatmap.png')
    print('Saved test_heatmap.png - smoke test passed')