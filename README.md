# BloodCare AI 2.0 🩸
**AI Personal Health Intelligence Platform**
*Khẩu hiệu: Không chỉ đọc kết quả xét nghiệm, mà còn dự đoán sức khỏe tương lai.*

## Mục tiêu dự án
Hệ thống sử dụng AI kết hợp Rule-based và Machine Learning để phân tích các chỉ số xét nghiệm máu, hỗ trợ phát hiện sớm nguy cơ Thiếu máu, Tiểu đường, Bệnh gan, và Bệnh thận.

## Cấu trúc hệ thống
- `data/`: Chứa dữ liệu huấn luyện (Dataset công khai).
- `models/`: Lưu trữ các mô hình học máy (XGBoost, CatBoost, LightGBM, Random Forest).
- `src/`: Mã nguồn xử lý Backend và Rule Engine.
- `templates/` & `static/`: Giao diện UI/UX và Dashboard theo dõi sức khỏe.