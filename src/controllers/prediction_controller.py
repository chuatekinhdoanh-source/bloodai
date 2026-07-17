from flask import Blueprint, render_template, request, jsonify, session
import pandas as pd
from controllers.auth_controller import login_required, api_login_required, doctor_required, api_doctor_required
from models import (
    save_prediction, get_user_by_id, get_all_patients,
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
            
        # Mô hình Gan đầu ra 0 đại diện cho có bệnh (class 1/Dataset 1), đầu ra 1 đại diện cho bình thường (class 2/Dataset 2)
        raw_pred = int(liver_model.predict(input_df)[0])
        prediction = 1 if raw_pred == 0 else 0
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
            
        # Mô hình Gan đầu ra 0 đại diện cho có bệnh (class 1/Dataset 1), đầu ra 1 đại diện cho bình thường (class 2/Dataset 2)
        raw_pred = int(liver_model.predict(input_df)[0])
        prediction = 1 if raw_pred == 0 else 0
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


# ==========================================================
#             SÀNG LỌC SỨC KHỎE TỔNG QUÁT ĐA MÔ HÌNH
# ==========================================================

def _get_pred_and_prob(model, input_df, name):
    if model is None:
        return 0, 0.0
    try:
        # Chuyển đổi kiểu dữ liệu cho khớp với mô hình học máy
        pred = int(model.predict(input_df)[0])
        if hasattr(model, 'predict_proba'):
            if name == 'liver':
                # Đảo ngược dự báo: 0 đại diện cho có bệnh, 1 đại diện cho bình thường
                pred = 1 if pred == 0 else 0
                prob = float(model.predict_proba(input_df)[0][0])  # Xác suất class 0 (có bệnh)
            else:
                prob = float(model.predict_proba(input_df)[0][1])  # Xác suất class 1 (có bệnh)
        else:
            if name == 'liver':
                pred = 1 if pred == 0 else 0
                prob = 1.0 if pred == 1 else 0.0
            else:
                prob = 1.0 if pred == 1 else 0.0
        return pred, prob
    except Exception as e:
        print(f"[PREDICTION] Error predicting {name}: {e}")
        return 0, 0.0

def _safe_float(val, default):
    if val is None or str(val).strip() == '':
        return default
    try:
        return float(val)
    except ValueError:
        return default

@prediction_bp.route('/predict_screening', methods=['POST'])
@login_required
@doctor_required
def predict_screening():
    try:

        # 1. Lấy thông tin bệnh nhân nhận kết quả
        patient_id = request.form['patient_id']
        patient = get_user_by_id(patient_id)
        patient_name = patient['fullname'] if patient else "Chưa rõ"

        # 2. Thu thập chỉ số cơ bản
        age = _safe_float(request.form.get('Age'), 45.0)
        gender = _safe_float(request.form.get('Gender'), 1.0)  # 1: Nam, 0: Nữ
        glucose = _safe_float(request.form.get('Glucose'), 100.0)
        hemo = _safe_float(request.form.get('Hemoglobin'), 14.0)
        bp = _safe_float(request.form.get('BloodPressure'), 80.0)
        bmi = _safe_float(request.form.get('BMI'), 22.0)

        # 3. Thu thập chỉ số chuyên sâu
        mcv = _safe_float(request.form.get('MCV'), 90.0)
        mch = _safe_float(request.form.get('MCH'), 28.0)
        mchc = _safe_float(request.form.get('MCHC'), 33.0)

        total_bilirubin = _safe_float(request.form.get('Total_Bilirubin'), 1.0)
        direct_bilirubin = _safe_float(request.form.get('Direct_Bilirubin'), 0.3)
        alkaline_phosphotase = _safe_float(request.form.get('Alkaline_Phosphotase'), 187.0)
        alamine_aminotransferase = _safe_float(request.form.get('Alamine_Aminotransferase'), 16.0)
        aspartate_aminotransferase = _safe_float(request.form.get('Aspartate_Aminotransferase'), 18.0)
        total_proteins = _safe_float(request.form.get('Total_Protiens'), 6.8)
        albumin = _safe_float(request.form.get('Albumin'), 3.3)
        albumin_globulin_ratio = _safe_float(request.form.get('Albumin_and_Globulin_Ratio'), 0.9)

        sg = _safe_float(request.form.get('sg'), 1.020)
        al = _safe_float(request.form.get('al'), 0.0)
        su = _safe_float(request.form.get('su'), 0.0)
        bu = _safe_float(request.form.get('bu'), 36.0)
        sc = _safe_float(request.form.get('sc'), 1.2)
        sod = _safe_float(request.form.get('sod'), 135.0)
        pot = _safe_float(request.form.get('pot'), 4.5)
        rbc = _safe_float(request.form.get('rbc'), 1.0)
        pc = _safe_float(request.form.get('pc'), 1.0)
        pcc = _safe_float(request.form.get('pcc'), 0.0)
        ba = _safe_float(request.form.get('ba'), 0.0)
        pcv = _safe_float(request.form.get('pcv'), 44.0)
        wc = _safe_float(request.form.get('wc'), 7800.0)
        rc = _safe_float(request.form.get('rc'), 5.2)
        htn = _safe_float(request.form.get('htn'), 0.0)
        dm = _safe_float(request.form.get('dm'), 0.0)
        cad = _safe_float(request.form.get('cad'), 0.0)
        appet = _safe_float(request.form.get('appet'), 1.0)
        pe = _safe_float(request.form.get('pe'), 0.0)
        ane = _safe_float(request.form.get('ane'), 0.0)

        pregnancies = _safe_float(request.form.get('Pregnancies'), 0.0) if gender == 0.0 else 0.0
        skin = _safe_float(request.form.get('SkinThickness'), 20.0)
        insulin = _safe_float(request.form.get('Insulin'), 79.0)
        dpf = _safe_float(request.form.get('DiabetesPedigreeFunction'), 0.47)

        # 4. CHẠY MÔ HÌNH AI CHO TỪNG BỆNH

        # A. Tiểu Đường (XGBoost)
        db_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        db_df = pd.DataFrame([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]], columns=db_cols)
        if hasattr(diabetes_model, 'feature_names_in_'):
            db_df = db_df.reindex(columns=diabetes_model.feature_names_in_)
        pred_db, prob_db = _get_pred_and_prob(diabetes_model, db_df, 'diabetes')

        # B. Thiếu Máu (CatBoost)
        anm_cols = ['Gender', 'Hemoglobin', 'MCH', 'MCHC', 'MCV']
        anm_df = pd.DataFrame([[gender, hemo, mch, mchc, mcv]], columns=anm_cols)
        if hasattr(anemia_model, 'feature_names_'):
            anm_df = anm_df.reindex(columns=anemia_model.feature_names_)
        elif hasattr(anemia_model, 'feature_names_in_'):
            anm_df = anm_df.reindex(columns=anemia_model.feature_names_in_)
        pred_anm, prob_anm = _get_pred_and_prob(anemia_model, anm_df, 'anemia')

        # C. Bệnh Gan (Random Forest)
        liv_cols = ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase', 'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio']
        liv_df = pd.DataFrame([[age, gender, total_bilirubin, direct_bilirubin, alkaline_phosphotase, alamine_aminotransferase, aspartate_aminotransferase, total_proteins, albumin, albumin_globulin_ratio]], columns=liv_cols)
        if hasattr(liver_model, 'feature_names_in_'):
            liv_df = liv_df.reindex(columns=liver_model.feature_names_in_)
        pred_liv, prob_liv = _get_pred_and_prob(liver_model, liv_df, 'liver')

        # D. Bệnh Thận (LightGBM)
        kdn_cols = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
        kdn_df = pd.DataFrame([[age, bp, sg, al, su, rbc, pc, pcc, ba, glucose, bu, sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad, appet, pe, ane]], columns=kdn_cols)
        
        expected_cols = None
        if hasattr(kidney_model, 'feature_name_'):
            expected_cols = kidney_model.feature_name_
            if callable(expected_cols):
                expected_cols = expected_cols()
        elif hasattr(kidney_model, 'feature_names_in_'):
            expected_cols = kidney_model.feature_names_in_
            
        if expected_cols is not None:
            kdn_df = kdn_df.reindex(columns=expected_cols, fill_value=0.0)
        pred_kdn, prob_kdn = _get_pred_and_prob(kidney_model, kdn_df, 'kidney')

        # 5. LƯU LỊCH SỬ CHẨN ĐOÁN (4 BẢN GHI RIÊNG LẺ)
        db_res = "CÓ NGUY CƠ TIỂU ĐƯỜNG" if pred_db == 1 else "BÌNH THƯỜNG"
        anm_res = "CÓ NGUY CƠ THIẾU MÁU" if pred_anm == 1 else "BÌNH THƯỜNG"
        liv_res = "CÓ NGUY CƠ BỆNH GAN" if pred_liv == 1 else "BÌNH THƯỜNG"
        kdn_res = "CÓ NGUY CƠ BỆNH THẬN" if pred_kdn == 1 else "BÌNH THƯỜNG"

        db_dict = {
            'Số lần mang thai': int(pregnancies),
            'Glucose': glucose,
            'Huyết áp': bp,
            'Độ dày nếp da': skin,
            'Insulin': insulin,
            'BMI': bmi,
            'Hệ số tiền sử': dpf,
            'Tuổi': int(age)
        }
        anm_dict = {
            'Giới tính': 'Nam' if gender == 1 else 'Nữ',
            'Hemoglobin': hemo,
            'MCH': mch,
            'MCHC': mchc,
            'MCV': mcv
        }
        liv_dict = {
            'Tuổi': int(age),
            'Giới tính': 'Nam' if gender == 1 else 'Nữ',
            'Bilirubin toàn phần': total_bilirubin,
            'Bilirubin trực tiếp': direct_bilirubin,
            'Alkaline Phosphotase': alkaline_phosphotase,
            'Alamine Aminotransferase': alamine_aminotransferase,
            'Aspartate Aminotransferase': aspartate_aminotransferase,
            'Protein toàn phần': total_proteins,
            'Albumin': albumin,
            'Tỷ lệ Albumin/Globulin': albumin_globulin_ratio
        }
        kdn_dict = {
            'Tuổi': int(age),
            'Huyết áp': bp,
            'Tỷ trọng nước tiểu': sg,
            'Albumin nước tiểu': al,
            'Đường nước tiểu': su,
            'Hồng cầu': 'Bình thường' if rbc == 1 else 'Bất thường',
            'Tế bào mủ': 'Bình thường' if pc == 1 else 'Bất thường',
            'Cụm tế bào mủ': 'Có' if pcc == 1 else 'Không',
            'Vi khuẩn': 'Có' if ba == 1 else 'Không',
            'Đường huyết ngẫu nhiên': glucose,
            'Ure máu': bu,
            'Creatinine huyết thanh': sc,
            'Natri': sod,
            'Kali': pot,
            'Hemoglobin': hemo,
            'Thể tích hồng cầu (PCV)': pcv,
            'Số lượng bạch cầu': wc,
            'Số lượng hồng cầu': rc,
            'Tăng huyết áp': 'Có' if htn == 1 else 'Không',
            'Tiểu đường': 'Có' if dm == 1 else 'Không',
            'Bệnh mạch vành': 'Có' if cad == 1 else 'Không',
            'Thèm ăn': 'Ngon miệng' if appet == 1 else 'Chán ăn',
            'Phù chân': 'Có' if pe == 1 else 'Không',
            'Thiếu máu': 'Có' if ane == 1 else 'Không'
        }

        save_prediction(patient_id, patient_name, session['user_id'], session['fullname'], 'diabetes', db_dict, db_res, pred_db)
        save_prediction(patient_id, patient_name, session['user_id'], session['fullname'], 'anemia', anm_dict, anm_res, pred_anm)
        save_prediction(patient_id, patient_name, session['user_id'], session['fullname'], 'liver', liv_dict, liv_res, pred_liv)
        save_prediction(patient_id, patient_name, session['user_id'], session['fullname'], 'kidney', kdn_dict, kdn_res, pred_kdn)

        # 6. TÍNH TOÁN CẢNH BÁO TỔNG QUÁT (MÃ MÀU XANH / VÀNG / ĐỎ)
        risk_count = pred_db + pred_anm + pred_liv + pred_kdn
        
        if risk_count == 0:
            status_color = "green"
            status_text = "AN TOÀN"
            status_desc = "Hệ thống đánh giá các chỉ số sinh hóa máu của bệnh nhân hiện ở mức an toàn. Nguy cơ mắc các bệnh lý sàng lọc là rất thấp. Hãy duy trì thói quen ăn uống lành mạnh và kiểm tra định kỳ."
        elif risk_count <= 2:
            status_color = "yellow"
            status_text = "CẦN THEO DÕI SÁT SAO"
            status_desc = f"Hệ thống phát hiện dấu hiệu nguy cơ ở {risk_count} nhóm bệnh lý. Hãy chú ý theo dõi các triệu chứng lâm sàng lâm học, điều chỉnh chế độ sinh hoạt và tiến hành kiểm tra lại sau 3-6 tháng."
        else:
            status_color = "red"
            status_text = "CẦN ĐI KHÁM CHUYÊN KHOA NGAY"
            status_desc = f"Cảnh báo nguy cơ cao! Hệ thống phát hiện bất thường ở {risk_count} nhóm bệnh lý khác nhau. Khuyến nghị bệnh nhân nhanh chóng đến các cơ sở y tế gần nhất để thực hiện các chẩn đoán chuyên sâu và nhận phác đồ điều trị."

        # Trích xuất dữ liệu trả về cho Report
        diseases = [
            {
                'name': 'Tiểu Đường',
                'model_name': 'XGBoost (Classifier)',
                'pred': pred_db,
                'prob': round(prob_db * 100, 1),
                'result_text': db_res
            },
            {
                'name': 'Thiếu Máu',
                'model_name': 'CatBoost (Classifier)',
                'pred': pred_anm,
                'prob': round(prob_anm * 100, 1),
                'result_text': anm_res
            },
            {
                'name': 'Bệnh Gan',
                'model_name': 'Random Forest (Ensemble)',
                'pred': pred_liv,
                'prob': round(prob_liv * 100, 1),
                'result_text': liv_res
            },
            {
                'name': 'Bệnh Thận',
                'model_name': 'LightGBM (Boosting)',
                'pred': pred_kdn,
                'prob': round(prob_kdn * 100, 1),
                'result_text': kdn_res
            }
        ]

        input_summary = {
            'Tuổi (Age)': int(age),
            'Giới tính': 'Nam' if gender == 1 else 'Nữ',
            'Glucose (Đường huyết)': f"{glucose} mg/dL",
            'Hemoglobin (Huyết sắc tố)': f"{hemo} g/dL",
            'Huyết áp tâm trương': f"{bp} mmHg",
            'Chỉ số khối cơ thể (BMI)': bmi
        }

        # 8. TỰ ĐỘNG HÓA BỆNH ÁN ĐIỆN TỬ (AUTO-NOTE GENERATOR)
        gender_text = "Nam" if gender == 1 else "Nữ"
        summary_intro = f"Bệnh nhân {gender_text.lower()}, {int(age)} tuổi."
        
        findings = []
        abnormal_inputs = []
        recommendations = []
        
        # Check models
        risks = []
        if pred_db == 1:
            risks.append("Tiểu đường")
            recommendations.append("Đo HbA1c, OGTT")
        if pred_anm == 1:
            risks.append("Thiếu máu")
            recommendations.append("Định lượng Sắt, Ferritin, CBC")
        if pred_liv == 1:
            risks.append("Bệnh lý gan")
            recommendations.append("Siêu âm ổ bụng, HBsAg, Anti-HCV")
        if pred_kdn == 1:
            risks.append("Bệnh lý thận")
            recommendations.append("Siêu âm hệ tiết niệu, đo eGFR, tổng phân tích nước tiểu")
            
        if risks:
            findings.append(f"Hệ thống AI phát hiện nguy cơ cao mắc các bệnh: {', '.join(risks)}.")
        else:
            findings.append("Các mô hình AI chưa phát hiện nguy cơ bệnh lý sàng lọc nguy hiểm.")
            
        # Check abnormal inputs
        if glucose > 126:
            abnormal_inputs.append(f"Glucose tăng cao ({glucose} mg/dL)")
        elif glucose < 70:
            abnormal_inputs.append(f"Glucose giảm ({glucose} mg/dL)")
            
        if hemo < 12.0:
            abnormal_inputs.append(f"Hemoglobin giảm ({hemo} g/dL)")
            
        if bp >= 90:
            abnormal_inputs.append(f"Huyết áp tâm trương cao ({bp} mmHg)")
            
        if bmi >= 25.0:
            abnormal_inputs.append(f"Thể trạng thừa cân/béo phì (BMI {bmi})")
        elif bmi < 18.5:
            abnormal_inputs.append(f"Thể trạng thiếu cân (BMI {bmi})")
            
        if alamine_aminotransferase > 56:
            abnormal_inputs.append(f"Men gan ALT tăng cao ({alamine_aminotransferase} U/L)")
        if aspartate_aminotransferase > 40:
            abnormal_inputs.append(f"Men gan AST tăng ({aspartate_aminotransferase} U/L)")
            
        if sc > 1.2:
            abnormal_inputs.append(f"Creatinine huyết thanh tăng ({sc} mg/dL)")
            
        if abnormal_inputs:
            findings.append(f"Ghi nhận bất thường cận lâm sàng: {', '.join(abnormal_inputs)}.")
            
        # Compile note
        recommendation_text = ""
        if recommendations:
            recommendation_text = f" Đề nghị chỉ định cận lâm sàng thêm: {', '.join(recommendations)}."
        else:
            recommendation_text = " Đề nghị duy trì khám định kỳ theo dõi."
            
        auto_note = f"{summary_intro} { ' '.join(findings) }{recommendation_text} Yêu cầu bệnh nhân điều chỉnh chế độ ăn uống, sinh hoạt lành mạnh theo hướng dẫn y khoa."

        return render_template('screening_result.html',
                               patient_name=patient_name,
                               status_color=status_color,
                               status_text=status_text,
                               status_desc=status_desc,
                               diseases=diseases,
                               input_summary=input_summary,
                               auto_note=auto_note)
    except Exception as e:
        return f"<h3>Lỗi Sàng lọc tổng quát: {e}</h3>"


@prediction_bp.route('/digitize_pdf', methods=['POST'])
@login_required
@doctor_required
def digitize_pdf():
    try:
        if 'pdf_file' not in request.files:
            return jsonify({'status': 'error', 'message': 'Không tìm thấy file PDF được tải lên.'}), 400
            
        file = request.files['pdf_file']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'Tên file trống.'}), 400
            
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'status': 'error', 'message': 'Định dạng file không phải PDF.'}), 400
            
        import PyPDF2
        import io
        import re
        
        pdf_stream = io.BytesIO(file.read())
        reader = PyPDF2.PdfReader(pdf_stream)
        
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
            
        # Parse text using regex
        extracted = {}
        
        # Mapping patterns (supporting accented/unaccented variations and intermediate comments via non-greedy wildcard)
        patterns = {
            'Age': [r'(?:Tuổi|Tuoi|Age).*?(\d+)'],
            'Gender': [r'(?:Giới\s+tính|Gioi\s+tinh|Gender|Sex).*?(Nam|Nữ|Nu|Male|Female|1|0)'],
            'Glucose': [r'(?:Glucose|Đường\s+huyết|Duong\s+huyet|bgr).*?(\d+(?:\.\d+)?)'],
            'Hemoglobin': [r'(?:Hemoglobin|Huyết\s+sắc\s+tố|Huyet\s+sac\s+to|Hgb|hemo).*?(\d+(?:\.\d+)?)'],
            'BloodPressure': [
                r'(?:Huyết\s+áp|Huyet\s+ap|Blood\s+Pressure|HA|bp).*?(\d+)\s*/\s*(\d+)',
                r'(?:Huyết\s+áp\s+tâm\s+trương|Huyet\s+ap\s+tam\s+truong|Diastolic\s+Blood\s+Pressure|Diastolic\s+BP|bp).*?(\d+(?:\.\d+)?)'
            ],
            'BMI': [r'(?:BMI|Chỉ\s+số\s+khối\s+cơ\s+thể|Chi\s+so\s+khoi\s+co\s+the).*?(\d+(?:\.\d+)?)'],
            'MCV': [r'(?:MCV).*?(\d+(?:\.\d+)?)'],
            'MCH': [r'(?:MCH).*?(\d+(?:\.\d+)?)'],
            'MCHC': [r'(?:MCHC).*?(\d+(?:\.\d+)?)'],
            'Total_Bilirubin': [r'(?:Bilirubin\s+toàn\s+phần|Bilirubin\s+toan\s+phan|Total\s+Bilirubin|TBil).*?(\d+(?:\.\d+)?)'],
            'Direct_Bilirubin': [r'(?:Bilirubin\s+trực\s+tiếp|Bilirubin\s+truc\s+tiep|Direct\s+Bilirubin|DBil).*?(\d+(?:\.\d+)?)'],
            'Alkaline_Phosphotase': [r'(?:Alkaline\s+Phosphotase|ALP).*?(\d+(?:\.\d+)?)'],
            'Alamine_Aminotransferase': [r'(?:Alamine\s+Aminotransferase|ALT|SGPT|Alanine\s+Aminotransferase).*?(\d+(?:\.\d+)?)'],
            'Aspartate_Aminotransferase': [r'(?:Aspartate\s+Aminotransferase|AST|SGOT).*?(\d+(?:\.\d+)?)'],
            'Total_Protiens': [r'(?:Protein\s+toàn\s+phần|Protein\s+toan\s+phan|Total\s+Protein|TP|Total\s+Protiens).*?(\d+(?:\.\d+)?)'],
            'Albumin': [r'(?:Albumin|ALB).*?(\d+(?:\.\d+)?)'],
            'Albumin_and_Globulin_Ratio': [r'(?:Tỷ\s+lệ\s+A/G|Ty\s+le\s+A/G|Albumin/Globulin|A/G\s+Ratio|AG\s+Ratio).*?(\d+(?:\.\d+)?)'],
            'sg': [r'(?:Tỷ\s+trọng\s+nước\s+tiểu|Ty\s+trong\s+nuoc\s+tieu|Tỷ\s+trọng|Ty\s+trong|Specific\s+Gravity|sg).*?(1\.\d{3})'],
            'al': [r'(?:Albumin\s+nước\s+tiểu|Albumin\s+nuoc\s+tieu|Urine\s+Albumin|al).*?(\d+)'],
            'su': [r'(?:Đường\s+nước\s+tiểu|Duong\s+nuoc\s+tieu|Urine\s+Sugar|su).*?(\d+)'],
            'bu': [r'(?:Ure\s+máu|Ure\s+mau|Ure|Blood\s+Urea|bu).*?(\d+(?:\.\d+)?)'],
            'sc': [r'(?:Creatinine|sc).*?(\d+(?:\.\d+)?)'],
            'sod': [r'(?:Natri|Sodium|sod).*?(\d+(?:\.\d+)?)'],
            'pot': [r'(?:Kali|Potassium|pot).*?(\d+(?:\.\d+)?)'],
            'pcv': [r'(?:Thể\s+tích\s+hồng\s+cầu|The\s+tich\s+hong\s+cau|PCV|HCT).*?(\d+(?:\.\d+)?)'],
            'wc': [r'(?:Bạch\s+cầu|Bach\s+cau|WBC).*?(\d+(?:\.\d+)?)'],
            'rc': [r'(?:Hồng\s+cầu|Hong\s+cau|RBC).*?(\d+(?:\.\d+)?)'],
            'htn': [r'(?:Tăng\s+huyết\s+áp|Tang\s+huyet\s+ap|Hypertension|htn).*?(Có|Không|Co|Khong|Yes|No|1|0)'],
            'dm': [r'(?:Đái\s+tháo\s+đường|Dai\s+thao\s+duong|Tiểu\s+đường|Tieu\s+duong|Diabetes|dm).*?(Có|Không|Co|Khong|Yes|No|1|0)'],
            'cad': [r'(?:Bệnh\s+mạch\s+vành|Benh\s+mach\s+vanh|Mạch\s+vành|Mach\s+vanh|CAD).*?(Có|Không|Co|Khong|Yes|No|1|0)'],
            'pe': [r'(?:Phù\s+chân|Phu\s+chan|Pedal\s+Edema|pe).*?(Có|Không|Co|Khong|Yes|No|1|0)'],
            'ane': [r'(?:Thiếu\s+máu|Thieu\s+mau|Anemia|ane).*?(Có|Không|Co|Khong|Yes|No|1|0)'],
            'appet': [r'(?:Thèm\s+ăn|Them\s+an|Ăn\s+ngon|An\s+ngon|Appetite|appet).*?(Ngon|Kém|Kem|Chán|Chan|Good|Poor)'],
            'Pregnancies': [r'(?:Số\s+lần\s+mang\s+thai|So\s+lan\s+mang\s+thai|Mang\s+thai|Pregnancies).*?(\d+)'],
            'SkinThickness': [r'(?:Độ\s+dày\s+nếp\s+da|Do\s+day\s+nep\s+da|Skin\s+Thickness).*?(\d+(?:\.\d+)?)'],
            'Insulin': [r'(?:Insulin).*?(\d+(?:\.\d+)?)'],
            'DiabetesPedigreeFunction': [r'(?:Diabetes\s+Pedigree|DPF).*?(\d+(?:\.\d+)?)']
        }
        
        for key, regexes in patterns.items():
            for regex in regexes:
                match = re.search(regex, text, re.IGNORECASE)
                if match:
                    val = match.group(1).strip()
                    # Custom processing for different types
                    if key == 'BloodPressure' and len(match.groups()) > 1:
                        # Composite systolic/diastolic -> get diastolic (second number)
                        val2 = match.group(2)
                        if val2:
                            extracted[key] = float(val2)
                        else:
                            extracted[key] = float(val)
                    elif key == 'Gender':
                        val_lower = val.lower()
                        if val_lower in ['nam', 'male', '1']:
                            extracted[key] = 1
                        else:
                            extracted[key] = 0
                    elif key in ['htn', 'dm', 'cad', 'pe', 'ane']:
                        val_lower = val.lower()
                        if val_lower in ['có', 'yes', '1']:
                            extracted[key] = 1
                        else:
                            extracted[key] = 0
                    elif key == 'appet':
                        val_lower = val.lower()
                        if val_lower in ['ngon', 'good', 'ngon miệng']:
                            extracted[key] = 1
                        else:
                            extracted[key] = 0
                    else:
                        # standard float/int
                        try:
                            if '.' in val:
                                extracted[key] = float(val)
                            else:
                                extracted[key] = int(val)
                        except ValueError:
                            pass
                    break  # Stop search at first match

        # Patient matching logic
        matched_patient_id = None
        try:
            import unicodedata
            def normalize_text(val):
                if not val:
                    return ""
                val = unicodedata.normalize('NFKD', val)
                val = "".join([c for c in val if not unicodedata.combining(c)])
                val = val.replace('đ', 'd').replace('Đ', 'D')
                val = val.lower()
                val = re.sub(r'[^a-z0-9\s]', '', val)
                return re.sub(r'\s+', ' ', val).strip()

            # 1. Extract name candidates using regexes in PDF text
            name_candidates = []
            name_regex = r'(?:họ\s*(?:và\s*)?tên|ho\s*(?:va\s*)?ten|bệnh\s*nhân|benh\s*nhan|patient\s*name|patient|name|tên|ten|họ\s*tên|ho\s*ten)\s*[:\-]?\s*([A-Za-zÀ-ỹđĐ\s]+)'
            for m in re.finditer(name_regex, text, re.IGNORECASE):
                cand = m.group(1).strip()
                cand = re.split(r'[\r\n,;.]', cand)[0].strip()
                for noise in ['tuổi', 'tuoi', 'age', 'giới', 'gioi', 'gender', 'sex', 'địa chỉ', 'dia chi', 'address', 'số điện thoại', 'sđt', 'phone']:
                    if noise in cand.lower():
                        idx = cand.lower().find(noise)
                        cand = cand[:idx].strip()
                if cand and 3 <= len(cand) <= 50:
                    name_candidates.append(cand)

            patients = get_all_patients()
            
            # Match 1: Exact matches against extracted name candidates
            if name_candidates:
                norm_candidates = [normalize_text(c) for c in name_candidates]
                for patient in patients:
                    norm_fullname = normalize_text(patient.get('fullname', ''))
                    norm_username = normalize_text(patient.get('username', ''))
                    for norm_cand in norm_candidates:
                        if norm_cand and (norm_cand == norm_fullname or norm_cand == norm_username):
                            matched_patient_id = patient['id']
                            break
                    if matched_patient_id:
                        break

            # Match 2: Substring matching in full normalized text
            if not matched_patient_id:
                norm_text = normalize_text(text)
                for patient in patients:
                    norm_fullname = normalize_text(patient.get('fullname', ''))
                    norm_username = normalize_text(patient.get('username', ''))
                    if norm_fullname and len(norm_fullname) >= 5:
                        pattern = r'\b' + re.escape(norm_fullname) + r'\b'
                        if re.search(pattern, norm_text):
                            matched_patient_id = patient['id']
                            break
                    if norm_username and len(norm_username) >= 4:
                        pattern = r'\b' + re.escape(norm_username) + r'\b'
                        if re.search(pattern, norm_text):
                            matched_patient_id = patient['id']
                            break
        except Exception as match_err:
            print(f"[DIGITIZE] Error matching patient: {match_err}")
                    
        return jsonify({
            'status': 'success',
            'extracted_data': extracted,
            'matched_patient_id': matched_patient_id,
            'text_preview': text[:200] + '...' if len(text) > 200 else text
        }), 200
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Lỗi số hóa bệnh án: {str(e)}'}), 500


