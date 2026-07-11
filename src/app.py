from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__, template_folder='../templates')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- TẢI CÁC MÔ HÌNH AI ĐÃ HUẤN LUYỆN ----
diabetes_model = joblib.load(os.path.join(BASE_DIR, 'models', 'xgboost_diabetes.pkl')) if os.path.exists(os.path.join(BASE_DIR, 'models', 'xgboost_diabetes.pkl')) else None
anemia_model = joblib.load(os.path.join(BASE_DIR, 'models', 'catboost_anemia.pkl')) if os.path.exists(os.path.join(BASE_DIR, 'models', 'catboost_anemia.pkl')) else None
liver_model = joblib.load(os.path.join(BASE_DIR, 'models', 'rf_liver.pkl')) if os.path.exists(os.path.join(BASE_DIR, 'models', 'rf_liver.pkl')) else None
kidney_model = joblib.load(os.path.join(BASE_DIR, 'models', 'lightgbm_kidney.pkl')) if os.path.exists(os.path.join(BASE_DIR, 'models', 'lightgbm_kidney.pkl')) else None

# ---- ROUTE CHUYỂN TRANG (MENU) ----
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/anemia')
def anemia_page():
    return render_template('anemia.html')

@app.route('/liver')
def liver_page():
    return render_template('liver.html')

@app.route('/kidney')
def kidney_page():
    return render_template('kidney.html')

# ---- CHỨC NĂNG DỰ ĐOÁN TIỂU ĐƯỜNG ----
@app.route('/predict', methods=['POST'])
def predict_diabetes():
    try:
        pregnancies = float(request.form['Pregnancies'])
        glucose = float(request.form['Glucose'])
        bp = float(request.form['BloodPressure'])
        skin = float(request.form['SkinThickness'])
        insulin = float(request.form['Insulin'])
        bmi = float(request.form['BMI'])
        dpf = float(request.form['DiabetesPedigreeFunction'])
        age = float(request.form['Age'])
        
        columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        input_data = pd.DataFrame([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]], columns=columns)
        
        prediction = diabetes_model.predict(input_data)[0]
        result = "<span style='color:red;'>CÓ NGUY CƠ TIỂU ĐƯỜNG</span>" if prediction == 1 else "<span style='color:green;'>BÌNH THƯỜNG</span>"
        return f"<div style='text-align:center; margin-top:50px;'><h2>Kết quả: {result}</h2><br><a href='/'>Quay lại Khoa Tiểu Đường</a></div>"
    except Exception as e:
        return f"<h3>Lỗi Tiểu đường: {e}</h3>"

# ---- CHỨC NĂNG DỰ ĐOÁN THIẾU MÁU ----
@app.route('/predict_anemia', methods=['POST'])
def predict_anemia():
    try:
        gender = float(request.form['Gender'])
        hemo = float(request.form['Hemoglobin'])
        mch = float(request.form['MCH'])
        mchc = float(request.form['MCHC'])
        mcv = float(request.form['MCV'])
        
        columns = ['Gender', 'Hemoglobin', 'MCH', 'MCHC', 'MCV']
        input_data = pd.DataFrame([[gender, hemo, mch, mchc, mcv]], columns=columns)
        
        prediction = anemia_model.predict(input_data)[0]
        result = "<span style='color:red;'>CÓ NGUY CƠ THIẾU MÁU</span>" if prediction == 1 else "<span style='color:green;'>BÌNH THƯỜNG</span>"
        return f"<div style='text-align:center; margin-top:50px;'><h2>Kết quả: {result}</h2><br><a href='/anemia'>Quay lại Khoa Thiếu Máu</a></div>"
    except Exception as e:
        return f"<h3>Lỗi Thiếu máu: {e}</h3>"

# ---- CHỨC NĂNG DỰ ĐOÁN BỆNH GAN ----
@app.route('/predict_liver', methods=['POST'])
def predict_liver():
    try:
        data = request.form.to_dict()
        for key in data: data[key] = float(data[key])
        input_df = pd.DataFrame([data])
        
        # Tự động căn chỉnh cột cho khớp với mô hình Random Forest
        if hasattr(liver_model, 'feature_names_in_'):
            input_df = input_df.reindex(columns=liver_model.feature_names_in_, fill_value=0)
            
        prediction = liver_model.predict(input_df)[0]
        result = "<span style='color:red;'>CÓ NGUY CƠ BỆNH GAN</span>" if prediction == 1 else "<span style='color:green;'>BÌNH THƯỜNG</span>"
        return f"<div style='text-align:center; margin-top:50px;'><h2>Kết quả: {result}</h2><br><a href='/liver'>Quay lại Khoa Gan</a></div>"
    except Exception as e:
        return f"<h3>Lỗi Bệnh Gan: {e}</h3>"

# ---- CHỨC NĂNG DỰ ĐOÁN BỆNH THẬN ----
@app.route('/predict_kidney', methods=['POST'])
def predict_kidney():
    try:
        data = request.form.to_dict()
        for key in data: data[key] = float(data[key])
        input_df = pd.DataFrame([data])
        
        # Tự động căn chỉnh cột cho khớp với mô hình LightGBM
        expected_cols = getattr(kidney_model, 'feature_name_', lambda: None)()
        if expected_cols is not None:
            input_df = input_df.reindex(columns=expected_cols, fill_value=0)
        elif hasattr(kidney_model, 'feature_names_in_'):
            input_df = input_df.reindex(columns=kidney_model.feature_names_in_, fill_value=0)

        prediction = kidney_model.predict(input_df)[0]
        result = "<span style='color:red;'>CÓ NGUY CƠ BỆNH THẬN</span>" if prediction == 1 else "<span style='color:green;'>BÌNH THƯỜNG</span>"
        return f"<div style='text-align:center; margin-top:50px;'><h2>Kết quả: {result}</h2><br><a href='/kidney'>Quay lại Khoa Thận</a></div>"
    except Exception as e:
        return f"<h3>Lỗi Bệnh Thận: {e}</h3>"

if __name__ == '__main__':
    app.run(debug=True)