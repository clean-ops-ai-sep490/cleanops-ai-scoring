"""
Model Architecture: CleaningQualityNet
======================================
Fine-tune ResNet50 pretrained để output 3 tham số theo spec:

    detected_stains    → regression 0-20
    dirt_coverage      → regression 0-100 (%)
    abnormal_objects   → regression 0-10

Kiến trúc:
    ResNet50 backbone (frozen ban đầu, unfreeze dần)
    → Global Average Pooling
    → Feature vector 2048-dim
    → 2 heads song song:
        regression_head  → [stains, coverage, objects]
        quality_head     → scalar score 0-100
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Tuple


class CleaningQualityNet(nn.Module):
    """
    Multi-task model cho cleaning quality scoring.
    
    Input: Tensor ảnh [B, 3, 224, 224]
    Output: Dict chứa tất cả predictions
    """
    
    def __init__(self, freeze_backbone: bool = True, dropout: float = 0.3):
        super().__init__()
        
        # Backbone: ResNet50 pretrained ImageNet
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Bỏ layer FC cuối, giữ feature extractor
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # output: [B, 2048, 1, 1]
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Unfreeze layer4 (block cuối) để fine-tune
            for param in self.backbone[-1].parameters():
                param.requires_grad = True
        
        feat_dim = 2048
        
        # Shared neck: giảm chiều trước khi tách heads
        self.neck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
        )
        
        # Head 1: Regression — detected_stains, dirt_coverage, abnormal_objects
        self.regression_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),   # [stains, coverage, objects]
            nn.Sigmoid()        # normalize 0-1, scale sau
        )
        
        # Head 2: Overall quality score 0-100
        self.quality_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()        # 0-1, nhân 100 khi inference
        )
        
        # Khởi tạo weights các heads
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.neck, self.regression_head, self.quality_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out')
                    nn.init.constant_(layer.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Backbone feature extraction
        features = self.backbone(x)            # [B, 2048, 1, 1]
        features = self.neck(features)         # [B, 256]
        
        # Multi-task outputs
        reg_out = self.regression_head(features)            # [B, 3] in [0,1]
        quality_raw = self.quality_head(features)           # [B, 1] in [0,1]
        
        return {
            "stains_norm": reg_out[:, 0],           # 0-1
            "coverage_norm": reg_out[:, 1],          # 0-1
            "objects_norm": reg_out[:, 2],           # 0-1
            "quality_norm": quality_raw.squeeze(1),  # 0-1
        }
    
    def predict(self, x: torch.Tensor) -> Dict:
        """
        Inference mode: trả về dict với giá trị đã scale về đơn vị thực.
        """
        self.eval()
        with torch.no_grad():
            raw = self.forward(x)
            
            stains = (raw["stains_norm"] * 20).round().int()
            coverage = raw["coverage_norm"] * 100
            objects = (raw["objects_norm"] * 10).round().int()
            quality_score = raw["quality_norm"] * 100
            
            return {
                "detected_stains": stains.cpu().tolist(),
                "dirt_coverage": coverage.cpu().tolist(),
                "abnormal_objects": objects.cpu().tolist(),
                "quality_score": quality_score.cpu().tolist(),
            }
    
    def unfreeze_backbone(self):
        """Gọi sau vài epoch để fine-tune toàn bộ backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("[*] Backbone đã được unfreeze hoàn toàn")
    
    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class BeforeAfterComparator(nn.Module):
    """
    So sánh 2 ảnh Before/After để tính improvement score.
    
    Kiến trúc:
        - Share cùng CleaningQualityNet cho cả 2 ảnh (Siamese-style)
        - Concat features → tính delta → ra confidence + verdict
    
    Input: before [B,3,224,224], after [B,3,224,224]
    Output: improvement_score, confidence, verdict
    """
    
    def __init__(self, base_model: CleaningQualityNet):
        super().__init__()
        self.encoder = base_model
        
        # Comparison head: nhận concat của 2 feature vectors
        self.compare_head = nn.Sequential(
            nn.Linear(512, 128),    # 256 * 2 = 512
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),       # [improvement_score, confidence]
            nn.Sigmoid()
        )
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract neck features (không qua heads)."""
        features = self.encoder.backbone(x)
        return self.encoder.neck(features)
    
    def forward(self, before: torch.Tensor, after: torch.Tensor):
        # Extract features cho cả 2
        feat_before = self.get_features(before)   # [B, 256]
        feat_after = self.get_features(after)     # [B, 256]
        
        # Concat và so sánh
        combined = torch.cat([feat_before, feat_after], dim=1)  # [B, 512]
        out = self.compare_head(combined)  # [B, 2]
        
        return {
            "improvement_score": out[:, 0] * 100,   # 0-100
            "confidence": out[:, 1] * 100,           # 0-100
        }


def build_model(mode: str = "single", freeze: bool = True) -> nn.Module:
    """
    Factory function tạo model.
    
    Args:
        mode: "single" (1 ảnh) hoặc "compare" (before/after)
        freeze: có freeze backbone không
    
    Returns:
        Model đã khởi tạo với pretrained weights
    """
    base = CleaningQualityNet(freeze_backbone=freeze)
    
    if mode == "compare":
        return BeforeAfterComparator(base)
    return base


if __name__ == "__main__":
    # Sanity check
    print("Kiểm tra model...")
    
    model = build_model(mode="single", freeze=True)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output keys: {list(out.keys())}")
    for k, v in out.items():
        print(f"  {k}: {v.shape}")
    
    print(f"\nTotal params:     {model.get_total_params():,}")
    print(f"Trainable params: {model.get_trainable_params():,}")
    
    # Test predict
    preds = model.predict(x)
    print(f"\nSample prediction (batch=2):")
    for k, v in preds.items():
        print(f"  {k}: {v}")
    
    # Test comparator
    print("\nKiểm tra BeforeAfterComparator...")
    comp = build_model(mode="compare", freeze=True)
    before = torch.randn(2, 3, 224, 224)
    after = torch.randn(2, 3, 224, 224)
    comp_out = comp(before, after)
    for k, v in comp_out.items():
        print(f"  {k}: {v}")
    
    print("\n[OK] Model OK!")
