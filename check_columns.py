import joblib
import os

# Đường dẫn tới thư mục chứa các mô hình AI
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Tải 2 mô hình Gan và Thận
liver_model_path = os.path.join(BASE_DIR, 'models', 'rf_liver.pkl')
kidney_model_path = os.path.join(BASE_DIR, 'models', 'lightgbm_kidney.pkl')

print("=== KIỂM TRA DỮ LIỆU ĐẦU VÀO ===")

# Kiểm tra Khoa Gan
if os.path.exists(liver_model_path):
    liver_model = joblib.load(liver_model_path)
    if hasattr(liver_model, 'feature_names_in_'):
        print(f"\n[KHOA GAN] Cần {len(liver_model.feature_names_in_)} chỉ số sau:")
        print(list(liver_model.feature_names_in_))
    else:
        print("\n[KHOA GAN] Không trích xuất được tên cột.")

# Kiểm tra Khoa Thận
if os.path.exists(kidney_model_path):
    kidney_model = joblib.load(kidney_model_path)
    expected_cols = getattr(kidney_model, 'feature_name_', lambda: None)()
    if expected_cols is not None:
        print(f"\n[KHOA THẬN] Cần {len(expected_cols)} chỉ số sau:")
        print(list(expected_cols))
    elif hasattr(kidney_model, 'feature_names_in_'):
        print(f"\n[KHOA THẬN] Cần {len(kidney_model.feature_names_in_)} chỉ số sau:")
        print(list(kidney_model.feature_names_in_))
    else:
        print("\n[KHOA THẬN] Không trích xuất được tên cột.")