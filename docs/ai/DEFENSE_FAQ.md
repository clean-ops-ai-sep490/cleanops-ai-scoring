# Defense FAQ

## Vì sao cần retrain?

Vì ảnh thực tế thay đổi theo môi trường, ánh sáng, chất liệu sàn, loại vết bẩn và cách chụp. U-Net học vùng bẩn/wet từ dữ liệu cũ nên có thể bỏ sót pattern mới. Retrain giúp model học lại từ supervisor feedback đã được duyệt.

## Khi nào cần retrain?

Khi có bằng chứng:

- đủ nhãn vùng bẩn/wet đã approved;
- false pass tăng;
- verdict accuracy giảm;
- supervisor sửa nhiều case cùng một kiểu lỗi;
- xuất hiện environment mới khác dữ liệu ban đầu.

Không retrain chỉ vì đã đến lịch nếu không có dữ liệu hoặc lỗi mới.

## Vì sao chỉ retrain U-Net?

Vì trọng tâm lỗi cleanliness thường nằm ở dirty coverage: model có nhìn đúng vùng bẩn/wet hay không. U-Net trực tiếp sinh mask và coverage. YOLO đang được giữ ổn định để giảm biến số, giúp hội đồng thấy pipeline retrain có mục tiêu rõ ràng.

## Rectangle annotation có hợp lý không?

Có, nhưng phải gọi đúng. Rectangle là weak region annotation, phù hợp cho MVP và retrain hỗ trợ. Nó nhanh, dễ làm trên UI, và có thể tạo pseudo-mask.

Giới hạn là rectangle không bám sát biên vùng bẩn. Nếu vùng bẩn loang, cong hoặc có nhiều nền sạch bên trong, rectangle sẽ đưa nhiễu vào mask. Vì vậy rectangle không đủ để claim segmentation benchmark nghiêm túc.

## Khi nào cần polygon hoặc mask?

Khi ảnh thuộc golden benchmark, khi cần đo mIoU, hoặc khi vùng bẩn có hình dạng không đều. Polygon/mask giúp label sát biên hơn và giảm nhiễu cho U-Net.

## Làm sao biết model mới đủ ổn để production?

Không nhìn mỗi train log. Candidate phải chạy trên cùng benchmark với active model. Nếu candidate không tăng false pass, không giảm accuracy, mIoU không tệ hơn baseline và latency vẫn đạt, model mới mới được promote.

## Nếu chưa có benchmark lớn thì nói thế nào?

Nói trung thực là hệ thống đang ở mức pilot benchmark. Mục tiêu đồ án là xây dựng AI workflow có evaluation, human feedback, retrain và promotion gate; benchmark lớn hơn là hướng phát triển tiếp theo.

## Tài liệu tham khảo

- CVAT shape types: https://docs.cvat.ai/docs/annotation/manual-annotation/shapes/types-of-shapes/
- CVAT mask export: https://docs.cvat.ai/docs/annotation/manual-annotation/shapes/annotation-with-polygons/creating-mask/
- CVAT brush annotation: https://docs.cvat.ai/v2.3.0/docs/manual/advanced/annotation-with-brush-tool/
- Ultralytics segmentation overview: https://academy.ultralytics.com/courses/computer-vision-foundations/instance-segmentation

