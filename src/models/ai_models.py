import joblib
import os

# Base directory của dự án
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Nạp các mô hình AI từ thư mục 'models/'
diabetes_model = joblib.load(os.path.join(BASE_DIR, 'models', 'xgboost_diabetes.pkl')) if os.path.exists(os.path.join(BASE_DIR, 'models', 'xgboost_diabetes.pkl')) else None
anemia_model = joblib.load(os.path.join(BASE_DIR, 'models', 'catboost_anemia.pkl')) if os.path.exists(os.path.join(BASE_DIR, 'models', 'catboost_anemia.pkl')) else None
liver_model = joblib.load(os.path.join(BASE_DIR, 'models', 'rf_liver.pkl')) if os.path.exists(os.path.join(BASE_DIR, 'models', 'rf_liver.pkl')) else None
kidney_model = joblib.load(os.path.join(BASE_DIR, 'models', 'lightgbm_kidney.pkl')) if os.path.exists(os.path.join(BASE_DIR, 'models', 'lightgbm_kidney.pkl')) else None
