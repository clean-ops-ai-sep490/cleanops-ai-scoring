import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class UNetSegmenter(nn.Module):
    def __init__(self, encoder_name="resnet18", encoder_weights="imagenet", in_channels=3, classes=1):
        """
        Khởi tạo U-Net với backbone ResNet18 hoặc MobileNet (rất nhẹ để chạy trên RTX 3050).
        """
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation='sigmoid' # Phân vùng nhị phân (vết bẩn/không)
        )
        
    def forward(self, x):
        return self.model(x)

    def predict_coverage(self, x, threshold=0.5):
        """
        Dự đoán % diện tích bị dơ trên ảnh.
        Input x: Tensor (C, H, W)
        """
        with torch.no_grad():
            x = x.unsqueeze(0) if x.dim() == 3 else x
            mask_out = self.model(x)
            
            binary_mask = (mask_out > threshold).float()
            
            # Tính tỉ lệ % diện tích có dơ so với toàn mặt phẳng 
            total_pixels = mask_out.numel() / mask_out.size(0)
            dirty_pixels = binary_mask.sum().item() / binary_mask.size(0)
            
            coverage = (dirty_pixels / total_pixels) * 100.0
            
            return coverage, binary_mask.squeeze(0)
