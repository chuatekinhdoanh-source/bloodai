from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from models.db import db
from bson.objectid import ObjectId

def register_user(username, password, fullname, role='patient', email=None, phone=None, dob=None, require_password_change=False):
    """Đăng ký tài khoản người dùng mới kèm theo vai trò (doctor hoặc patient)."""
    if db is None:
        return None
    try:
        users_col = db['users']
        username_clean = username.strip().lower()
        # Kiểm tra xem tài khoản đã tồn tại chưa
        if users_col.find_one({'username': username_clean}):
            print(f"[USER] Register failed: Username '{username_clean}' already exists.")
            return "exists"
            
        password_hash = generate_password_hash(password)
        doc = {
            'username': username_clean,
            'password_hash': password_hash,
            'fullname': fullname.strip(),
            'role': role.strip().lower() if role in ['doctor', 'patient'] else 'patient',
            'email': email.strip() if email else '',
            'phone': phone.strip() if phone else '',
            'dob': dob.strip() if dob else '',
            'require_password_change': require_password_change,
            'created_at': datetime.now()
        }
        res = users_col.insert_one(doc)
        print(f"[USER] User '{username_clean}' ({role}) registered successfully.")
        return str(res.inserted_id)
    except Exception as e:
        print(f"[USER] Error registering user: {e}")
        return None

def authenticate_user(username, password):
    """Xác thực người dùng. Trả về thông tin chi tiết (gồm cả role) nếu thành công."""
    if db is None:
        return None
    try:
        users_col = db['users']
        username_clean = username.strip().lower()
        user = users_col.find_one({'username': username_clean})
        if not user:
            print(f"[USER] Auth failed: Username '{username_clean}' not found.")
            return None
            
        if check_password_hash(user['password_hash'], password):
            print(f"[USER] User '{username_clean}' authenticated successfully.")
            return {
                'id': str(user['_id']),
                'username': user['username'],
                'fullname': user.get('fullname', user['username']),
                'role': user.get('role', 'patient'),
                'require_password_change': user.get('require_password_change', False)
            }
        else:
            print(f"[USER] Auth failed: Incorrect password for '{username_clean}'.")
            return None
    except Exception as e:
        print(f"[USER] Error authenticating user: {e}")
        return None

def get_all_patients():
    """Lấy danh sách tất cả người dùng có vai trò là bệnh nhân."""
    if db is None:
        return []
    try:
        users_col = db['users']
        cursor = users_col.find({'role': 'patient'}).sort('fullname', 1)
        patients = []
        for doc in cursor:
            patients.append({
                'id': str(doc['_id']),
                'fullname': doc.get('fullname', doc['username']),
                'username': doc['username']
            })
        return patients
    except Exception as e:
        print(f"[USER] Error getting patients list: {e}")
        return []

def get_user_by_id(user_id):
    """Lấy thông tin người dùng dựa trên ID."""
    if db is None:
        return None
    try:
        users_col = db['users']
        user = users_col.find_one({'_id': ObjectId(user_id)})
        if user:
            return {
                'id': str(user['_id']),
                'username': user['username'],
                'fullname': user.get('fullname', user['username']),
                'role': user.get('role', 'patient'),
                'email': user.get('email', ''),
                'phone': user.get('phone', ''),
                'dob': user.get('dob', ''),
                'require_password_change': user.get('require_password_change', False)
            }
        return None
    except Exception as e:
        print(f"[USER] Error getting user by ID: {e}")
        return None
