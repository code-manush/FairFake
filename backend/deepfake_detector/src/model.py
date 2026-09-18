import timm
import torch
import torch.nn as nn

class XceptionDetector(nn.Module):

    def __init__(self, pretrained=True, freeze_backbone=False):
        super().__init__()
        self.backbone = timm.create_model('legacy_xception', pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(feat_dim, 1))
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(self, x):
        feats = self.backbone(x)
        return self.classifier(feats).squeeze(1)

class FrequencyAwareDetector(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()
        self.rgb_backbone = timm.create_model('legacy_xception', pretrained=pretrained, num_classes=0)
        rgb_feat_dim = self.rgb_backbone.num_features
        self.freq_branch = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        freq_feat_dim = 32
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(rgb_feat_dim + freq_feat_dim, 1))

    def forward(self, x_rgb, x_freq):
        f_rgb = self.rgb_backbone(x_rgb)
        f_freq = self.freq_branch(x_freq)
        combined = torch.cat([f_rgb, f_freq], dim=1)
        return self.classifier(combined).squeeze(1)

def compute_fft_magnitude(gray_img_tensor):
    import numpy as np
    f = np.fft.fft2(gray_img_tensor)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-08)
    return magnitude.astype(np.float32)
if __name__ == '__main__':
    model = XceptionDetector(pretrained=False)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print('Output shape:', out.shape)