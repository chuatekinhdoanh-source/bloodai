from datetime import datetime
from bson.objectid import ObjectId
from models.db import db

def save_prediction(patient_id, patient_name, doctor_id, doctor_name, disease_type, input_dict, result_text, is_positive, pdf_file=None):
    """Lưu kết quả dự đoán mới vào MongoDB, liên kết với patient_id và doctor_id."""
    if db is None:
        print("[DATABASE] Error: Database connection is not initialized.")
        return None
    try:
        predictions_col = db['predictions']
        doc = {
            'patient_id': str(patient_id),
            'patient_name': patient_name,
            'doctor_id': str(doctor_id),
            'doctor_name': doctor_name,
            'user_id': str(patient_id), # Để tương thích ngược với các hàm cũ
            'disease_type': disease_type,
            'input_data': input_dict,
            'result': result_text,
            'is_positive': int(is_positive),
            'pdf_file': pdf_file,
            'created_at': datetime.now()
        }
        res = predictions_col.insert_one(doc)
        print(f"[DATABASE] Saved {disease_type} prediction successfully. Patient: {patient_name}, Doctor: {doctor_name}")
        return str(res.inserted_id)
    except Exception as e:
        print(f"[DATABASE] Error saving prediction: {e}")
        return None

def get_predictions(user_id, role, disease_filter=None, limit=100):
    """Lấy lịch sử chẩn đoán dựa trên vai trò (Bác sĩ xem ca đã khám, Bệnh nhân xem ca của mình)."""
    if db is None:
        return []
    try:
        predictions_col = db['predictions']
        
        # Phân quyền lọc dữ liệu lịch sử
        if role == 'doctor':
            query = {'doctor_id': str(user_id)}
        else:
            query = {'patient_id': str(user_id)}
            
        if disease_filter:
            query['disease_type'] = disease_filter
            
        cursor = predictions_col.find(query).sort('created_at', -1).limit(limit)
        results = []
        for doc in cursor:
            pred_id = str(doc['_id'])
            created_str = ""
            if 'created_at' in doc and doc['created_at']:
                created_str = doc['created_at'].strftime('%Y-%m-%d %H:%M:%S')
                
            results.append({
                'id': pred_id,
                'disease_type': doc.get('disease_type', ''),
                'input_data': doc.get('input_data', {}),
                'result': doc.get('result', ''),
                'is_positive': doc.get('is_positive', 0),
                'created_at': created_str,
                'patient_id': doc.get('patient_id', ''),
                'patient_name': doc.get('patient_name', 'Chưa rõ'),
                'doctor_id': doc.get('doctor_id', ''),
                'doctor_name': doc.get('doctor_name', 'Chưa rõ'),
                'pdf_file': doc.get('pdf_file', None)
            })
        return results
    except Exception as e:
        print(f"[DATABASE] Error fetching predictions: {e}")
        return []

def delete_prediction(prediction_id):
    """Xóa một bản ghi khỏi MongoDB dựa trên ObjectId."""
    if db is None:
        return
    try:
        predictions_col = db['predictions']
        predictions_col.delete_one({'_id': ObjectId(prediction_id)})
        print(f"[DATABASE] Deleted prediction {prediction_id} successfully.")
    except Exception as e:
        print(f"[DATABASE] Error deleting prediction: {e}")


# ---- ÁNH XẠ CHỈ SỐ LÂM SÀNG TỪ DATABASE SANG TÊN FORM ----
VIETNAMESE_TO_TECHNICAL_MAP = {
    'Số lần mang thai': 'Pregnancies',
    'Glucose': 'Glucose',
    'Huyết áp': 'BloodPressure',
    'Độ dày nếp da': 'SkinThickness',
    'Insulin': 'Insulin',
    'BMI': 'BMI',
    'Hệ số tiền sử': 'DiabetesPedigreeFunction',
    'Tuổi': 'Age',
    
    'Giới tính': 'Gender',
    'Hemoglobin': 'Hemoglobin',
    'MCH': 'MCH',
    'MCHC': 'MCHC',
    'MCV': 'MCV',
    
    'Bilirubin toàn phần': 'Total_Bilirubin',
    'Bilirubin trực tiếp': 'Direct_Bilirubin',
    'Alkaline Phosphotase': 'Alkaline_Phosphotase',
    'Alamine Aminotransferase': 'Alamine_Aminotransferase',
    'Aspartate Aminotransferase': 'Aspartate_Aminotransferase',
    'Protein toàn phần': 'Total_Protiens',
    'Albumin': 'Albumin',
    'Tỷ lệ Albumin/Globulin': 'Albumin_and_Globulin_Ratio',
    
    'Tỷ trọng nước tiểu': 'sg',
    'Albumin nước tiểu': 'al',
    'Đường nước tiểu': 'su',
    'Hồng cầu': 'rbc',
    'Tế bào mủ': 'pc',
    'Cụm tế bào mủ': 'pcc',
    'Vi khuẩn': 'ba',
    'Đường huyết ngẫu nhiên': 'bgr',
    'Ure máu': 'bu',
    'Creatinine huyết thanh': 'sc',
    'Natri': 'sod',
    'Kali': 'pot',
    'Thể tích hồng cầu (PCV)': 'pcv',
    'Số lượng bạch cầu': 'wc',
    'Số lượng hồng cầu': 'rc',
    'Tăng huyết áp': 'htn',
    'Tiểu đường': 'dm',
    'Bệnh mạch vành': 'cad',
    'Thèm ăn': 'appet',
    'Phù chân': 'pe',
    'Thiếu máu': 'ane'
}

def map_vietnamese_to_technical(vn_data):
    """Ánh xạ dữ liệu từ tiếng Việt lưu ở DB sang các tên biến kỹ thuật phục vụ pre-fill form."""
    tech_data = {}
    for vn_key, val in vn_data.items():
        tech_key = VIETNAMESE_TO_TECHNICAL_MAP.get(vn_key)
        if not tech_key:
            continue
            
        # Chuyển đổi ngược các giá trị phân loại dạng chuỗi thành số
        if val == 'Nam':
            mapped_val = 1.0
        elif val == 'Nữ':
            mapped_val = 0.0
        elif val == 'Bình thường':
            mapped_val = 1.0
        elif val == 'Bất thường':
            mapped_val = 0.0
        elif val == 'Có':
            mapped_val = 1.0
        elif val == 'Không':
            mapped_val = 0.0
        elif val == 'Ngon miệng':
            mapped_val = 1.0
        elif val == 'Chán ăn':
            mapped_val = 0.0
        else:
            mapped_val = val
            
        tech_data[tech_key] = mapped_val
        
    # Tạo các bí danh (aliases) cho biểu mẫu Bệnh Thận (Kidney)
    if 'Age' in tech_data:
        tech_data['age'] = tech_data['Age']
    if 'BloodPressure' in tech_data:
        tech_data['bp'] = tech_data['BloodPressure']
    if 'Glucose' in tech_data:
        tech_data['bgr'] = tech_data['Glucose']
    if 'Hemoglobin' in tech_data:
        tech_data['hemo'] = tech_data['Hemoglobin']
        
    return tech_data

def get_latest_patient_clinical_data(patient_id):
    """Truy vấn các bản ghi dự đoán mới nhất của từng loại bệnh của bệnh nhân để tái tạo bộ chỉ số cận lâm sàng đầy đủ."""
    if db is None:
        return {}
    try:
        predictions_col = db['predictions']
        merged_data = {}
        # Lấy bản ghi mới nhất của từng loại bệnh (nếu có)
        for disease_type in ['diabetes', 'anemia', 'liver', 'kidney']:
            doc = predictions_col.find_one(
                {'patient_id': str(patient_id), 'disease_type': disease_type},
                sort=[('created_at', -1)]
            )
            if doc and 'input_data' in doc:
                merged_data.update(doc['input_data'])
        return merged_data
    except Exception as e:
        print(f"[DATABASE] Error getting latest clinical data for patient {patient_id}: {e}")
    return {}
