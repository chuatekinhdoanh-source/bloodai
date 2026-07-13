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
    print("BAT DAU QUA TRINH HUAN LUYEN AI MULTI-MODELS (DU LIEU CHUAN HOA)\n" + "="*50)
    
    # 1. Bệnh Tiểu Đường (XGBoost)
    try:
        df = pd.read_csv('data/diabetes.csv').dropna()
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = xgb.XGBClassifier(eval_metric='logloss')
        model.fit(X_train, y_train)
        
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"[OK] Tieu Duong (XGBoost) - Do chinh xac: {acc*100:.2f}%")
        joblib.dump(model, 'models/xgboost_diabetes.pkl')
    except Exception as e:
        print(f"[ERR] Tieu Duong: {e}")

    # 2. Bệnh Gan (Random Forest)
    try:
        df = pd.read_csv('data/liver.csv').dropna()
        # Ánh xạ Gender chuỗi sang dạng số 1/0
        df['Gender'] = df['Gender'].astype(str).str.strip().map({'Male': 1, 'Female': 0})
        df = df.dropna()
        
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1].astype('category').cat.codes
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"[OK] Benh Gan (Random Forest) - Do chinh xac: {acc*100:.2f}%")
        joblib.dump(model, 'models/rf_liver.pkl')
    except Exception as e:
        print(f"[ERR] Benh Gan: {e}")

    # 3. Bệnh Thận (LightGBM)
    try:
        df = pd.read_csv('data/kidney.csv')
        # Loại bỏ cột ID không chứa thông tin chẩn đoán lý thuyết
        if 'id' in df.columns:
            df = df.drop(columns=['id'])
            
        # Chuẩn hóa cột số và loại bỏ ký tự ẩn tab (\t)
        numeric_cols = ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace('\t', '').str.strip(), errors='coerce')
            
        # Ánh xạ các cột phân loại nhị phân sang dạng số 1/0
        binary_maps = {
            'rbc': {'normal': 1, 'abnormal': 0},
            'pc': {'normal': 1, 'abnormal': 0},
            'pcc': {'present': 1, 'notpresent': 0},
            'ba': {'present': 1, 'notpresent': 0},
            'htn': {'yes': 1, 'no': 0},
            'dm': {'yes': 1, 'no': 0, '\tyes': 1, '\tno': 0},
            'cad': {'yes': 1, 'no': 0, '\tno': 0},
            'appet': {'good': 1, 'poor': 0},
            'pe': {'yes': 1, 'no': 0},
            'ane': {'yes': 1, 'no': 0}
        }
        for col, mapping in binary_maps.items():
            df[col] = df[col].astype(str).str.replace('\t', '').str.strip().map(mapping)
            
        # Chuẩn hóa cột nhãn phân loại
        df['classification'] = df['classification'].astype(str).str.replace('\t', '').str.strip().map({'ckd': 1, 'notckd': 0})
        
        # Bỏ các hàng bị khuyết dữ liệu
        df = df.dropna()
        
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = lgb.LGBMClassifier(verbose=-1)
        model.fit(X_train, y_train)
        
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"[OK] Benh Than (LightGBM) - Do chinh xac: {acc*100:.2f}%")
        joblib.dump(model, 'models/lightgbm_kidney.pkl')
    except Exception as e:
        print(f"[ERR] Benh Than: {e}")

    # 4. Thiếu Máu (CatBoost)
    try:
        df = pd.read_csv('data/anemia.csv').dropna()
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = CatBoostClassifier(verbose=0)
        model.fit(X_train, y_train)
        
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"[OK] Thieu Mau (CatBoost) - Do chinh xac: {acc*100:.2f}%")
        joblib.dump(model, 'models/catboost_anemia.pkl')
    except Exception as e:
        print(f"[ERR] Thieu Mau: {e}")

    print("="*50 + "\nTat ca mo hinh da duoc huan luyen va luu lai vao 'models/'.")

if __name__ == '__main__':
    train_all_models()