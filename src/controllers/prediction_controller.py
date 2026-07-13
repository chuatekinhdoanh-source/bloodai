from flask import Blueprint, render_template, request, jsonify, session
import pandas as pd
from controllers.auth_controller import login_required, api_login_required, doctor_required, api_doctor_required
from models import (
    save_prediction, get_user_by_id,
    diabetes_model, anemia_model, liver_model, kidney_model
)

prediction_bp = Blueprint('prediction', __name__)

# ---- CHỨC NĂNG DỰ ĐOÁN TIỂU ĐƯỜNG ----
@prediction_bp.route('/predict', methods=['POST'])
@login_required
@doctor_required
def predict_diabetes():
    try:
        patient_id = request.form['patient_id']
        patient = get_user_by_id(patient_id)
        patient_name = patient['fullname'] if patient else "Chưa rõ"
        
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
        if hasattr(diabetes_model, 'feature_names_in_'):
            input_data = input_data.reindex(columns=diabetes_model.feature_names_in_)
        
        prediction = int(diabetes_model.predict(input_data)[0])
        result = "CÓ NGUY CƠ TIỂU ĐƯỜNG" if prediction == 1 else "BÌNH THƯỜNG"
        
        input_dict = {
            'Số lần mang thai': int(pregnancies),
            'Glucose': glucose,
            'Huyết áp': bp,
            'Độ dày nếp da': skin,
            'Insulin': insulin,
            'BMI': bmi,
            'Hệ số tiền sử': dpf,
            'Tuổi': int(age)
        }
        save_prediction(patient_id, patient_name, session['user_id'], session['fullname'], 'diabetes', input_dict, result, prediction)
        
        return render_template('result.html', 
                               disease_name="Phân tích nguy cơ Tiểu Đường (XGBoost)",
                               result_text=result,
                               is_positive=prediction,
                               back_url="/",
                               input_data=input_dict)
    except Exception as e:
        return f"<h3>Lỗi Tiểu đường: {e}</h3>"

# ---- CHỨC NĂNG DỰ ĐOÁN THIẾU MÁU ----
@prediction_bp.route('/predict_anemia', methods=['POST'])
@login_required
@doctor_required
def predict_anemia():
    try:
        patient_id = request.form['patient_id']
        patient = get_user_by_id(patient_id)
        patient_name = patient['fullname'] if patient else "Chưa rõ"
        
        gender = float(request.form['Gender'])
        hemo = float(request.form['Hemoglobin'])
        mch = float(request.form['MCH'])
        mchc = float(request.form['MCHC'])
        mcv = float(request.form['MCV'])
        
        columns = ['Gender', 'Hemoglobin', 'MCH', 'MCHC', 'MCV']
        input_data = pd.DataFrame([[gender, hemo, mch, mchc, mcv]], columns=columns)
        if hasattr(anemia_model, 'feature_names_'):
            input_data = input_data.reindex(columns=anemia_model.feature_names_)
        elif hasattr(anemia_model, 'feature_names_in_'):
            input_data = input_data.reindex(columns=anemia_model.feature_names_in_)
        
        prediction = int(anemia_model.predict(input_data)[0])
        result = "CÓ NGUY CƠ THIẾU MÁU" if prediction == 1 else "BÌNH THƯỜNG"
        
        input_dict = {
            'Giới tính': 'Nam' if gender == 1 else 'Nữ',
            'Hemoglobin': hemo,
            'MCH': mch,
            'MCHC': mchc,
            'MCV': mcv
        }
        save_prediction(patient_id, patient_name, session['user_id'], session['fullname'], 'anemia', input_dict, result, prediction)
        
        return render_template('result.html', 
                               disease_name="Phân tích nguy cơ Thiếu Máu (CatBoost)",
                               result_text=result,
                               is_positive=prediction,
                               back_url="/anemia",
                               input_data=input_dict)
    except Exception as e:
        return f"<h3>Lỗi Thiếu máu: {e}</h3>"

# ---- CHỨC NĂNG DỰ ĐOÁN BỆNH GAN ----
@prediction_bp.route('/predict_liver', methods=['POST'])
@login_required
@doctor_required
def predict_liver():
    try:
        patient_id = request.form['patient_id']
        patient = get_user_by_id(patient_id)
        patient_name = patient['fullname'] if patient else "Chưa rõ"
        
        data = request.form.to_dict()
        # Loại bỏ patient_id trước khi đưa vào DataFrame dự đoán của mô hình AI
        del data['patient_id']
        
        for key in data: data[key] = float(data[key])
        input_df = pd.DataFrame([data])
        
        if hasattr(liver_model, 'feature_names_in_'):
            input_df = input_df.reindex(columns=liver_model.feature_names_in_, fill_value=0.0)
            
        prediction = int(liver_model.predict(input_df)[0])
        result = "CÓ NGUY CƠ BỆNH GAN" if prediction == 1 else "BÌNH THƯỜNG"
        
        input_dict = data.copy()
        if 'Gender' in input_dict:
            input_dict['Giới tính'] = 'Nam' if input_dict['Gender'] == 1 else 'Nữ'
            del input_dict['Gender']
        
        save_prediction(patient_id, patient_name, session['user_id'], session['fullname'], 'liver', input_dict, result, prediction)
        
        return render_template('result.html', 
                               disease_name="Phân tích nguy cơ Bệnh Gan (Random Forest)",
                               result_text=result,
                               is_positive=prediction,
                               back_url="/liver",
                               input_data=input_dict)
    except Exception as e:
        return f"<h3>Lỗi Bệnh Gan: {e}</h3>"

# ---- CHỨC NĂNG DỰ ĐOÁN BỆNH THẬN ----
@prediction_bp.route('/predict_kidney', methods=['POST'])
@login_required
@doctor_required
def predict_kidney():
    try:
        patient_id = request.form['patient_id']
        patient = get_user_by_id(patient_id)
        patient_name = patient['fullname'] if patient else "Chưa rõ"
        
        data = request.form.to_dict()
        # Loại bỏ patient_id trước khi đưa vào DataFrame dự đoán của mô hình AI
        del data['patient_id']
        
        for key in data: data[key] = float(data[key])
        input_df = pd.DataFrame([data])
        
        expected_cols = None
        if hasattr(kidney_model, 'feature_name_'):
            expected_cols = kidney_model.feature_name_
            if callable(expected_cols):
                expected_cols = expected_cols()
        elif hasattr(kidney_model, 'feature_names_in_'):
            expected_cols = kidney_model.feature_names_in_
            
        if expected_cols is not None:
            input_df = input_df.reindex(columns=expected_cols, fill_value=0.0)

        prediction = int(kidney_model.predict(input_df)[0])
        result = "CÓ NGUY CƠ BỆNH THẬN" if prediction == 1 else "BÌNH THƯỜNG"
        
        save_prediction(patient_id, patient_name, session['user_id'], session['fullname'], 'kidney', data, result, prediction)
        
        return render_template('result.html', 
                               disease_name="Phân tích nguy cơ Bệnh Thận (LightGBM)",
                               result_text=result,
                               is_positive=prediction,
                               back_url="/kidney",
                               input_data=data)
    except Exception as e:
        return f"<h3>Lỗi Bệnh Thận: {e}</h3>"


# ==========================================================
#                   HỆ THỐNG RESTful APIs (JSON)
# ==========================================================

# 1. API DỰ ĐOÁN TIỂU ĐƯỜNG
@prediction_bp.route('/api/predict/diabetes', methods=['POST'])
@api_login_required
@api_doctor_required
def api_predict_diabetes():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': 'Yêu cầu định dạng JSON.'}), 400
            
        patient_id = data.get('patient_id')
        if not patient_id:
            return jsonify({'status': 'error', 'message': 'Thiếu trường bắt buộc: patient_id'}), 400
            
        patient = get_user_by_id(patient_id)
        if not patient:
            return jsonify({'status': 'error', 'message': 'Không tìm thấy bệnh nhân tương ứng.'}), 404
        patient_name = patient['fullname']
        
        required_fields = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        for field in required_fields:
            if field not in data:
                return jsonify({'status': 'error', 'message': f'Thiếu trường bắt buộc: {field}'}), 400

        pregnancies = float(data['Pregnancies'])
        glucose = float(data['Glucose'])
        bp = float(data['BloodPressure'])
        skin = float(data['SkinThickness'])
        insulin = float(data['Insulin'])
        bmi = float(data['BMI'])
        dpf = float(data['DiabetesPedigreeFunction'])
        age = float(data['Age'])
        
        columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        input_data = pd.DataFrame([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]], columns=columns)
        if hasattr(diabetes_model, 'feature_names_in_'):
            input_data = input_data.reindex(columns=diabetes_model.feature_names_in_)
        
        if diabetes_model is None:
            return jsonify({'status': 'error', 'message': 'Mô hình chưa tải.'}), 500
            
        prediction = int(diabetes_model.predict(input_data)[0])
        result = "CÓ NGUY CƠ TIỂU ĐƯỜNG" if prediction == 1 else "BÌNH THƯỜNG"
        
        input_dict = {
            'Số lần mang thai': int(pregnancies),
            'Glucose': glucose,
            'Huyết áp': bp,
            'Độ dày nếp da': skin,
            'Insulin': insulin,
            'BMI': bmi,
            'Hệ số tiền sử': dpf,
            'Tuổi': int(age)
        }
        
        pred_id = save_prediction(patient_id, patient_name, session['user_id'], session['fullname'], 'diabetes', input_dict, result, prediction)
        
        return jsonify({
            'status': 'success',
            'disease_type': 'diabetes',
            'prediction': prediction,
            'result': result,
            'id': pred_id,
            'input_data': input_dict
        }), 201
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# 2. API DỰ ĐOÁN THIẾU MÁU
@prediction_bp.route('/api/predict/anemia', methods=['POST'])
@api_login_required
@api_doctor_required
def api_predict_anemia():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': 'Yêu cầu định dạng JSON.'}), 400
            
        patient_id = data.get('patient_id')
        if not patient_id:
            return jsonify({'status': 'error', 'message': 'Thiếu trường bắt buộc: patient_id'}), 400
            
        patient = get_user_by_id(patient_id)
        if not patient:
            return jsonify({'status': 'error', 'message': 'Không tìm thấy bệnh nhân tương ứng.'}), 404
        patient_name = patient['fullname']
        
        required_fields = ['Gender', 'Hemoglobin', 'MCH', 'MCHC', 'MCV']
        for field in required_fields:
            if field not in data:
                return jsonify({'status': 'error', 'message': f'Thiếu trường bắt buộc: {field}'}), 400

        gender = float(data['Gender'])
        hemo = float(data['Hemoglobin'])
        mch = float(data['MCH'])
        mchc = float(data['MCHC'])
        mcv = float(data['MCV'])
        
        columns = ['Gender', 'Hemoglobin', 'MCH', 'MCHC', 'MCV']
        input_data = pd.DataFrame([[gender, hemo, mch, mchc, mcv]], columns=columns)
        if hasattr(anemia_model, 'feature_names_'):
            input_data = input_data.reindex(columns=anemia_model.feature_names_)
        elif hasattr(anemia_model, 'feature_names_in_'):
            input_data = input_data.reindex(columns=anemia_model.feature_names_in_)
        
        if anemia_model is None:
            return jsonify({'status': 'error', 'message': 'Mô hình chưa tải.'}), 500
            
        prediction = int(anemia_model.predict(input_data)[0])
        result = "CÓ NGUY CƠ THIẾU MÁU" if prediction == 1 else "BÌNH THƯỜNG"
        
        input_dict = {
            'Giới tính': 'Nam' if gender == 1 else 'Nữ',
            'Hemoglobin': hemo,
            'MCH': mch,
            'MCHC': mchc,
            'MCV': mcv
        }
        
        pred_id = save_prediction(patient_id, patient_name, session['user_id'], session['fullname'], 'anemia', input_dict, result, prediction)
        
        return jsonify({
            'status': 'success',
            'disease_type': 'anemia',
            'prediction': prediction,
            'result': result,
            'id': pred_id,
            'input_data': input_dict
        }), 201
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# 3. API DỰ ĐOÁN BỆNH GAN
@prediction_bp.route('/api/predict/liver', methods=['POST'])
@api_login_required
@api_doctor_required
def api_predict_liver():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': 'Yêu cầu định dạng JSON.'}), 400
            
        patient_id = data.get('patient_id')
        if not patient_id:
            return jsonify({'status': 'error', 'message': 'Thiếu trường bắt buộc: patient_id'}), 400
            
        patient = get_user_by_id(patient_id)
        if not patient:
            return jsonify({'status': 'error', 'message': 'Không tìm thấy bệnh nhân tương ứng.'}), 404
        patient_name = patient['fullname']
        
        float_data = {}
        for key in data:
            if key == 'patient_id': continue
            try:
                float_data[key] = float(data[key])
            except ValueError:
                return jsonify({'status': 'error', 'message': f'Giá trị trường {key} phải là số.'}), 400
                
        input_df = pd.DataFrame([float_data])
        
        if liver_model is None:
            return jsonify({'status': 'error', 'message': 'Mô hình chưa tải.'}), 500
            
        if hasattr(liver_model, 'feature_names_in_'):
            input_df = input_df.reindex(columns=liver_model.feature_names_in_, fill_value=0.0)
            
        prediction = int(liver_model.predict(input_df)[0])
        result = "CÓ NGUY CƠ BỆNH GAN" if prediction == 1 else "BÌNH THƯỜNG"
        
        input_dict = float_data.copy()
        if 'Gender' in input_dict:
            input_dict['Giới tính'] = 'Nam' if float(input_dict['Gender']) == 1 else 'Nữ'
            del input_dict['Gender']
            
        pred_id = save_prediction(patient_id, patient_name, session['user_id'], session['fullname'], 'liver', input_dict, result, prediction)
        
        return jsonify({
            'status': 'success',
            'disease_type': 'liver',
            'prediction': prediction,
            'result': result,
            'id': pred_id,
            'input_data': input_dict
        }), 201
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# 4. API DỰ ĐOÁN BỆNH THẬN
@prediction_bp.route('/api/predict/kidney', methods=['POST'])
@api_login_required
@api_doctor_required
def api_predict_kidney():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': 'Yêu cầu định dạng JSON.'}), 400
            
        patient_id = data.get('patient_id')
        if not patient_id:
            return jsonify({'status': 'error', 'message': 'Thiếu trường bắt buộc: patient_id'}), 400
            
        patient = get_user_by_id(patient_id)
        if not patient:
            return jsonify({'status': 'error', 'message': 'Không tìm thấy bệnh nhân tương ứng.'}), 404
        patient_name = patient['fullname']
        
        float_data = {}
        for key in data:
            if key == 'patient_id': continue
            try:
                float_data[key] = float(data[key])
            except ValueError:
                return jsonify({'status': 'error', 'message': f'Giá trị trường {key} phải là số.'}), 400
                
        input_df = pd.DataFrame([float_data])
        
        if kidney_model is None:
            return jsonify({'status': 'error', 'message': 'Mô hình chưa tải.'}), 500
            
        expected_cols = None
        if hasattr(kidney_model, 'feature_name_'):
            expected_cols = kidney_model.feature_name_
            if callable(expected_cols):
                expected_cols = expected_cols()
        elif hasattr(kidney_model, 'feature_names_in_'):
            expected_cols = kidney_model.feature_names_in_
            
        if expected_cols is not None:
            input_df = input_df.reindex(columns=expected_cols, fill_value=0.0)
 
        prediction = int(kidney_model.predict(input_df)[0])
        result = "CÓ NGUY CƠ BỆNH THẬN" if prediction == 1 else "BÌNH THƯỜNG"
        
        pred_id = save_prediction(patient_id, patient_name, session['user_id'], session['fullname'], 'kidney', float_data, result, prediction)
        
        return jsonify({
            'status': 'success',
            'disease_type': 'kidney',
            'prediction': prediction,
            'result': result,
            'id': pred_id,
            'input_data': float_data
        }), 201
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
