import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Nhập khẩu 4 thuật toán AI cốt lõi
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier

def train_all_models():
    os.makedirs('models', exist_ok=True)
    print("BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN AI MULTI-MODELS\n" + "="*40)
    
    # 1. Bệnh Tiểu Đường (XGBoost)
    try:
        df = pd.read_csv('data/diabetes.csv').dropna()
        X = pd.get_dummies(df.iloc[:, :-1], drop_first=True)
        y = df.iloc[:, -1].astype('category').cat.codes
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = xgb.XGBClassifier(eval_metric='logloss')
        model.fit(X_train, y_train)
        
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"[XONG] Tiểu Đường (XGBoost) - Độ chính xác: {acc*100:.2f}%")
        joblib.dump(model, 'models/xgboost_diabetes.pkl')
    except Exception as e:
        print(f"[LỖI] Tiểu Đường: {e}")

    # 2. Bệnh Gan (Random Forest)
    try:
        df = pd.read_csv('data/liver.csv').dropna()
        X = pd.get_dummies(df.iloc[:, :-1], drop_first=True)
        y = df.iloc[:, -1].astype('category').cat.codes
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"[XONG] Bệnh Gan (Random Forest) - Độ chính xác: {acc*100:.2f}%")
        joblib.dump(model, 'models/rf_liver.pkl')
    except Exception as e:
        print(f"[LỖI] Bệnh Gan: {e}")

    # 3. Bệnh Thận (LightGBM)
    try:
        df = pd.read_csv('data/kidney.csv').dropna()
        X = pd.get_dummies(df.iloc[:, :-1], drop_first=True)
        y = df.iloc[:, -1].astype('category').cat.codes

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = lgb.LGBMClassifier(verbose=-1)
        model.fit(X_train, y_train)
        
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"[XONG] Bệnh Thận (LightGBM) - Độ chính xác: {acc*100:.2f}%")
        joblib.dump(model, 'models/lightgbm_kidney.pkl')
    except Exception as e:
        print(f"[LỖI] Bệnh Thận: {e}")

    # 4. Thiếu Máu (CatBoost)
    try:
        df = pd.read_csv('data/anemia.csv').dropna()
        X = pd.get_dummies(df.iloc[:, :-1], drop_first=True)
        y = df.iloc[:, -1].astype('category').cat.codes
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = CatBoostClassifier(verbose=0)
        model.fit(X_train, y_train)
        
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"[XONG] Thiếu Máu (CatBoost) - Độ chính xác: {acc*100:.2f}%")
        joblib.dump(model, 'models/catboost_anemia.pkl')
    except Exception as e:
        print(f"[LỖI] Thiếu Máu: {e}")

    print("="*40 + "\nTất cả mô hình đã được lưu vào thư mục 'models/'.")

if __name__ == '__main__':
    train_all_models()