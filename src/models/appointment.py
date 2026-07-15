from datetime import datetime
from bson.objectid import ObjectId
from models.db import db

def create_appointment(patient_id, patient_name, doctor_id, doctor_name, appointment_date, appointment_time, reason):
    """Tạo một yêu cầu lịch hẹn khám bệnh mới."""
    if db is None:
        return None
    try:
        appointments_col = db['appointments']
        doc = {
            'patient_id': str(patient_id),
            'patient_name': patient_name,
            'doctor_id': str(doctor_id),
            'doctor_name': doctor_name,
            'appointment_date': appointment_date, # Định dạng YYYY-MM-DD
            'appointment_time': appointment_time, # Định dạng HH:MM
            'reason': reason.strip(),
            'status': 'pending', # pending, confirmed, cancelled, completed
            'created_at': datetime.now()
        }
        res = appointments_col.insert_one(doc)
        print(f"[APPOINTMENT] Created new appointment {res.inserted_id} for Patient: {patient_name} with Doctor: {doctor_name}")
        return str(res.inserted_id)
    except Exception as e:
        print(f"[APPOINTMENT] Error creating appointment: {e}")
        return None

def get_appointments(user_id, role):
    """Lấy danh sách các lịch hẹn (Bác sĩ xem ca đặt khám mình phụ trách, Bệnh nhân xem ca mình đặt)."""
    if db is None:
        return []
    try:
        appointments_col = db['appointments']
        if role == 'doctor':
            query = {'doctor_id': str(user_id)}
        else:
            query = {'patient_id': str(user_id)}
            
        cursor = appointments_col.find(query).sort('created_at', -1)
        results = []
        for doc in cursor:
            created_str = ""
            if 'created_at' in doc and doc['created_at']:
                created_str = doc['created_at'].strftime('%Y-%m-%d %H:%M:%S')
                
            results.append({
                'id': str(doc['_id']),
                'patient_id': doc.get('patient_id', ''),
                'patient_name': doc.get('patient_name', ''),
                'doctor_id': doc.get('doctor_id', ''),
                'doctor_name': doc.get('doctor_name', ''),
                'appointment_date': doc.get('appointment_date', ''),
                'appointment_time': doc.get('appointment_time', ''),
                'reason': doc.get('reason', ''),
                'status': doc.get('status', 'pending'),
                'created_at': created_str
            })
        return results
    except Exception as e:
        print(f"[APPOINTMENT] Error fetching appointments: {e}")
        return []

def update_appointment_status(appointment_id, status):
    """Cập nhật trạng thái lịch hẹn khám bệnh."""
    if db is None:
        return False
    try:
        appointments_col = db['appointments']
        valid_statuses = ['pending', 'confirmed', 'cancelled', 'completed']
        if status not in valid_statuses:
            return False
            
        res = appointments_col.update_one(
            {'_id': ObjectId(appointment_id)},
            {'$set': {'status': status}}
        )
        print(f"[APPOINTMENT] Updated status of appointment {appointment_id} to '{status}'.")
        return res.modified_count > 0
    except Exception as e:
        print(f"[APPOINTMENT] Error updating status: {e}")
        return False

def get_all_doctors():
    """Lấy danh sách tất cả người dùng có vai trò là bác sĩ."""
    if db is None:
        return []
    try:
        users_col = db['users']
        cursor = users_col.find({'role': 'doctor'}).sort('fullname', 1)
        doctors = []
        for doc in cursor:
            doctors.append({
                'id': str(doc['_id']),
                'fullname': doc.get('fullname', doc['username']),
                'username': doc['username']
            })
        return doctors
    except Exception as e:
        print(f"[USER] Error getting doctors list: {e}")
        return []
