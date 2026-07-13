from datetime import datetime
from bson.objectid import ObjectId
from models.db import db

def save_prediction(patient_id, patient_name, doctor_id, doctor_name, disease_type, input_dict, result_text, is_positive):
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
                'doctor_name': doc.get('doctor_name', 'Chưa rõ')
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
