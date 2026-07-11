from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__, template_folder='../templates')

# Sử dụng đường dẫn tuyệt đối để chắc chắn tìm thấy file mô hình
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'xgboost_diabetes.pkl')

if os.path.exists(MODEL_PATH):
    diabetes_model = joblib.load(MODEL_PATH)
else:
    diabetes_model = None

@app.route('/')
def home():
    # Hiển thị file index.html
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not diabetes_model:
        return "<h3>Lỗi: Chưa tìm thấy file mô hình. Hãy chắc chắn bạn đã chạy file models_train.py!</h3>"
    
    try:
        # Lấy đủ 8 dữ liệu từ form HTML
        pregnancies = float(request.form['Pregnancies'])
        glucose = float(request.form['Glucose'])
        bp = float(request.form['BloodPressure'])
        skin = float(request.form['SkinThickness'])
        insulin = float(request.form['Insulin'])
        bmi = float(request.form['BMI'])
        dpf = float(request.form['DiabetesPedigreeFunction'])
        age = float(request.form['Age'])
        
        # Đóng gói dữ liệu ĐÚNG THỨ TỰ mà AI yêu cầu
        columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        input_data = pd.DataFrame([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]], columns=columns)
        
        # AI đưa ra dự đoán
        prediction = diabetes_model.predict(input_data)[0]
        
        if prediction == 1:
            result = "<span style='color:red; font-weight:bold;'>CÓ NGUY CƠ TIỂU ĐƯỜNG</span>"
        else:
            result = "<span style='color:green; font-weight:bold;'>BÌNH THƯỜNG</span>"
            
        return f"<div style='text-align:center; margin-top:50px; font-family:sans-serif;'><h2>Kết quả phân tích: {result}</h2><br><a href='/' style='padding:10px 20px; background:#007bff; color:white; text-decoration:none; border-radius:5px;'>Quay lại trang chủ</a></div>"
        
    except Exception as e:
        return f"<h3>Đã xảy ra lỗi khi nhập dữ liệu: {e}</h3>"

if __name__ == '__main__':
    app.run(debug=True)