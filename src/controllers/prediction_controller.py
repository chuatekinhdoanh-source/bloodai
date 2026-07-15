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
        age = float(request.form.get('Age', 45))
        gender = float(request.form.get('Gender', 1))  # 1: Nam, 0: Nữ
        glucose = float(request.form.get('Glucose', 100.0))
        hemo = float(request.form.get('Hemoglobin', 14.0))
        bp = float(request.form.get('BloodPressure', 80.0))
        bmi = float(request.form.get('BMI', 22.0))

        # 3. Thu thập chỉ số chuyên sâu
        mcv = float(request.form.get('MCV', 90.0))
        mch = float(request.form.get('MCH', 28.0))
        mchc = float(request.form.get('MCHC', 33.0))

        total_bilirubin = float(request.form.get('Total_Bilirubin', 1.0))
        direct_bilirubin = float(request.form.get('Direct_Bilirubin', 0.3))
        alkaline_phosphotase = float(request.form.get('Alkaline_Phosphotase', 187.0))
        alamine_aminotransferase = float(request.form.get('Alamine_Aminotransferase', 16.0))
        aspartate_aminotransferase = float(request.form.get('Aspartate_Aminotransferase', 18.0))
        total_proteins = float(request.form.get('Total_Protiens', 6.8))
        albumin = float(request.form.get('Albumin', 3.3))
        albumin_globulin_ratio = float(request.form.get('Albumin_and_Globulin_Ratio', 0.9))

        sg = float(request.form.get('sg', 1.020))
        al = float(request.form.get('al', 0.0))
        su = float(request.form.get('su', 0.0))
        bu = float(request.form.get('bu', 36.0))
        sc = float(request.form.get('sc', 1.2))
        sod = float(request.form.get('sod', 135.0))
        pot = float(request.form.get('pot', 4.5))
        rbc = float(request.form.get('rbc', 1.0))
        pc = float(request.form.get('pc', 1.0))
        pcc = float(request.form.get('pcc', 0.0))
        ba = float(request.form.get('ba', 0.0))
        pcv = float(request.form.get('pcv', 44.0))
        wc = float(request.form.get('wc', 7800.0))
        rc = float(request.form.get('rc', 5.2))
        htn = float(request.form.get('htn', 0.0))
        dm = float(request.form.get('dm', 0.0))
        cad = float(request.form.get('cad', 0.0))
        appet = float(request.form.get('appet', 1.0))
        pe = float(request.form.get('pe', 0.0))
        ane = float(request.form.get('ane', 0.0))

        pregnancies = float(request.form.get('Pregnancies', 0)) if gender == 0 else 0.0
        skin = float(request.form.get('SkinThickness', 20.0))
        insulin = float(request.form.get('Insulin', 79.0))
        dpf = float(request.form.get('DiabetesPedigreeFunction', 0.47))

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

        return render_template('screening_result.html',
                               patient_name=patient_name,
                               status_color=status_color,
                               status_text=status_text,
                               status_desc=status_desc,
                               diseases=diseases,
                               input_summary=input_summary)
    except Exception as e:
        return f"<h3>Lỗi Sàng lọc tổng quát: {e}</h3>"

