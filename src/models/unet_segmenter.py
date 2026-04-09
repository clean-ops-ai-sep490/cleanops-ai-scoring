import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class UNetSegmenter(nn.Module):
    def __init__(self, encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=3):
        """
        U-Net multiclass cho bài toán segmentation sàn:
          - 0: background
          - 1: stains/water
          - 2: wet surface
        """
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
        )

    def forward(self, x):
        return self.model(x)

    def predict_mask(self, x):
        """
        Trả về:
          - pred: mask class index [B, H, W]
          - probs: xác suất softmax [B, C, H, W]
        """
        with torch.no_grad():
            x = x.unsqueeze(0) if x.dim() == 3 else x
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1)
            return pred, probs

    def predict_coverage(self, x):
        """
        Tính tỷ lệ diện tích theo từng lớp quan tâm từ prediction.
        Return:
          - stains_pct (% lớp 1)
          - wet_surface_pct (% lớp 2)
          - pred_mask (H, W)
        """
        pred, _ = self.predict_mask(x)
        pred_mask = pred.squeeze(0)

        total_pixels = float(pred_mask.numel())
        stains_pixels = float((pred_mask == 1).sum().item())
        wet_pixels = float((pred_mask == 2).sum().item())

        stains_pct = (stains_pixels / total_pixels) * 100.0
        wet_surface_pct = (wet_pixels / total_pixels) * 100.0
        return stains_pct, wet_surface_pct, pred_mask
