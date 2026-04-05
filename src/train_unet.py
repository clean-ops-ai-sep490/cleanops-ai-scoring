import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from src.models.unet_segmenter import UNetSegmenter

class SimpleSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, mask_transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.mask_transform = mask_transform
        
        self.images = sorted([f.name for f in self.images_dir.glob("*.jpg")])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        # Giả định mask có định dạng png (màu đen trắng nhị phân) giống tên với bản Image gốc
        mask_name = img_name.replace(".jpg", ".png")
        
        img_path = self.images_dir / img_name
        mask_path = self.masks_dir / mask_name

        image = Image.open(img_path).convert("RGB")
        try:
            mask = Image.open(mask_path).convert("L")  # Grayscale
        except FileNotFoundError:
            # Fallback nếu không thấy mask -> Mask đen hoàn toàn (clean)
            mask = Image.new("L", image.size, color=0)

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        # Chuyển giá trị của mask về dạng nhị phân 0 hoặc 1
        mask = (mask > 0).float()
        return image, mask

def main():
    print("=" * 60)
    print("U-NET TRAINING PIPELINE (STAINS / DIRT SEGMENTATION)")
    print("=" * 60)
    
    img_dir = "data/raw/unet/images"
    mask_dir = "data/raw/unet/masks"

    if not Path(img_dir).exists() or len(list(Path(img_dir).glob("*.jpg"))) == 0:
        print("[ERROR] Bạn chưa tải THỦ CÔNG dataset Mendeley/SSGD vào thư mục:")
        print(f"        {img_dir} và {mask_dir}")
        print("        Xin vui lòng tự giải nén và chép file vào đây trước khi train.")
        return

    # Khởi tạo mô hình
    print("[*] Đang nạp mô hình U-Net (ResNet18 Backbone)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetSegmenter(encoder_name="resnet18").to(device)

    # Cấu hình data transforms (size gọn 512, đủ chạy 16 VRAM)
    img_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    dataset = SimpleSegmentationDataset(img_dir, mask_dir, transform=img_transform, mask_transform=mask_transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

    # Loss cho phân vùng nhị phân (0: Sạch / 1: Ố bẩn)
    # LƯU Ý: Vì U-Net trả sigmoid, nên tính loss dùng BCELoss thay vì BCEWithLogits
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 10
    print(f"\n[*] Bắt đầu vòng lặp Train U-Net (Epochs={epochs}, Batch=8)...")
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device) # C shape [B, 1, H, W]

            optimizer.zero_grad()
            outputs = model(images) # Nạp qua U-Net
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        print(f" -> Hoàn tất Epoch {epoch+1} | Lỗi trung bình: {epoch_loss/len(dataloader):.4f}")

    # Save model
    save_path = "models/unet_best.pth"
    Path("models").mkdir(parents=True, exist_ok=True)
    # Lưu trọng số của Unet Base Model (Không kèm wrapper)
    torch.save(model.model.state_dict(), save_path)
    print(f"\n[DONE] Đã lưu model U-Net vào: {save_path}")

if __name__ == "__main__":
    main()
