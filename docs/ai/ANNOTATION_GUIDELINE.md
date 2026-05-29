# Annotation Guideline

## Hai cấp annotation

AI scoring có hai cấp ground truth:

- Cấp 1: supervisor verdict `PASS/PENDING/FAIL`. Đây là ground truth nghiệp vụ, dùng cho benchmark verdict accuracy, false pass và false fail.
- Cấp 2: vùng bẩn/wet trên ảnh. Đây là dữ liệu phục vụ U-Net retrain và segmentation benchmark.

Không phải ảnh nào cũng cần cấp 2. Nếu mục tiêu chỉ là đánh giá verdict, supervisor review là đủ. Nếu mục tiêu là cải thiện U-Net dirty coverage, cần khoanh vùng bẩn/wet.

## Nhãn vùng

Hệ thống dùng ba class pixel:

- `0 background`: nền sạch hoặc không liên quan.
- `1 stain_or_water`: vết bẩn, vết nước, chất lỏng hoặc vùng bề mặt không sạch nói chung.
- `2 wet_surface`: bề mặt ướt cần phân biệt rõ với bẩn khô.

Nếu không chắc vùng là wet surface, ưu tiên `stain_or_water` và ghi chú.

## Rectangle policy

Rectangle là weak region annotation:

- Hợp lý cho MVP, retrain nhanh, hoặc vùng bẩn gần hình chữ nhật.
- Hợp lý khi mục tiêu là giúp model học vị trí vùng bẩn tương đối.
- Không phải pixel-perfect mask.
- Có thể làm U-Net học cả nền sạch nằm bên trong khung.

Không dùng rectangle-only dataset để claim U-Net segmentation chính xác cao.

## Polygon/mask policy

Polygon hoặc mask phù hợp khi:

- Vết bẩn/wet loang không đều.
- Vùng bẩn mảnh, cong, hoặc có nhiều lỗ nền sạch.
- Ảnh thuộc golden benchmark set.
- Cần đo `mIoU` nghiêm túc.

Nếu có thời gian, golden benchmark nên dùng polygon/mask cho ít nhất 30-50 ảnh.

## Quy tắc thao tác

- Khoanh sát vùng bẩn nhìn thấy, tránh lấy quá nhiều nền sạch.
- Không khoanh vật thể không liên quan nếu nó không ảnh hưởng cleanliness.
- Với nhiều vùng rời nhau, tạo nhiều region thay vì một khung lớn.
- Ghi chú những ảnh mơ hồ, ví dụ đang lau dở, phản chiếu ánh sáng, nền có hoa văn giống vết bẩn.
- Annotation đã approved nên khóa để phục vụ audit retrain.

