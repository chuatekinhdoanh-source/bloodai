from flask import Blueprint, render_template, request, redirect, url_for, session, flash, jsonify
from functools import wraps
import re
import random
from models import register_user, authenticate_user

auth_bp = Blueprint('auth', __name__)

def login_required(f):
    """Decorator bảo vệ các trang HTML, yêu cầu đăng nhập."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash("Vui lòng đăng nhập để tiếp tục.", "error")
            return redirect(url_for('auth.login_page'))
        return f(*args, **kwargs)
    return decorated_function

def api_login_required(f):
    """Decorator bảo vệ các API Endpoint, yêu cầu đăng nhập."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'status': 'error', 'message': 'Yêu cầu đăng nhập để truy cập API.'}), 401
        return f(*args, **kwargs)
    return decorated_function

def doctor_required(f):
    """Decorator bảo vệ trang chẩn đoán, chỉ cho phép bác sĩ truy cập."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get('role') != 'doctor':
            flash("Quyền truy cập bị từ chối. Chỉ Bác sĩ mới được phép thực hiện chẩn đoán.", "error")
            return redirect(url_for('history.history_page'))
        return f(*args, **kwargs)
    return decorated_function

def api_doctor_required(f):
    """Decorator bảo vệ API chẩn đoán, chỉ cho phép bác sĩ gọi."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get('role') != 'doctor':
            return jsonify({'status': 'error', 'message': 'Quyền truy cập bị từ chối. Chỉ Bác sĩ mới được phép thực hiện chẩn đoán.'}), 403
        return f(*args, **kwargs)
    return decorated_function

@auth_bp.route('/register', methods=['GET', 'POST'])
def register_page():
    if 'user_id' in session:
        return redirect(url_for('home.index'))
        
    if request.method == 'POST':
        fullname = request.form['fullname']
        username = request.form['username']
        password = request.form['password']
        role = request.form.get('role', 'patient')
        
        result = register_user(username, password, fullname, role=role)
        if result == "exists":
            flash("Tên đăng nhập đã tồn tại.", "error")
        elif result:
            flash("Đăng ký tài khoản thành công! Hãy đăng nhập.", "success")
            return redirect(url_for('auth.login_page'))
        else:
            flash("Đăng ký thất bại. Vui lòng kiểm tra lại thông tin.", "error")
            
    return render_template('register.html')

@auth_bp.route('/login', methods=['GET', 'POST'])
def login_page():
    if 'user_id' in session:
        return redirect(url_for('home.index'))
        
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = authenticate_user(username, password)
        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['fullname'] = user['fullname']
            session['role'] = user['role']
            session['require_password_change'] = user.get('require_password_change', False)
            
            flash(f"Chào mừng, BS. {user['fullname']}!" if user['role'] == 'doctor' else f"Chào mừng, BN. {user['fullname']}!", "success")
            
            # Check if password change is forced
            if session.get('require_password_change'):
                flash("Đây là lần đăng nhập đầu tiên bằng mật khẩu tạm thời. Vui lòng đổi mật khẩu để tiếp tục bảo mật thông tin y tế.", "warning")
                return redirect(url_for('auth.change_temp_password_page'))
                
            # Điều hướng linh hoạt dựa trên vai trò
            if user['role'] == 'patient':
                return redirect(url_for('history.history_page'))
            return redirect(url_for('home.index'))
        else:
            flash("Tên đăng nhập hoặc mật khẩu không chính xác.", "error")
            
    return render_template('login.html')

@auth_bp.route('/change_temp_password', methods=['GET', 'POST'])
def change_temp_password_page():
    if 'user_id' not in session:
        return redirect(url_for('auth.login_page'))
        
    if not session.get('require_password_change'):
        # If they don't need a password change, redirect to history or index
        if session.get('role') == 'patient':
            return redirect(url_for('history.history_page'))
        return redirect(url_for('home.index'))
        
    if request.method == 'POST':
        new_password = request.form.get('new_password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()
        
        if len(new_password) < 6:
            flash("Mật khẩu mới phải có ít nhất 6 ký tự.", "error")
        elif new_password != confirm_password:
            flash("Xác nhận mật khẩu mới không khớp.", "error")
        else:
            try:
                # Update password hash in MongoDB
                from models.db import db
                from werkzeug.security import generate_password_hash
                from bson.objectid import ObjectId
                
                users_col = db['users']
                password_hash = generate_password_hash(new_password)
                users_col.update_one(
                    {'_id': ObjectId(session['user_id'])},
                    {'$set': {
                        'password_hash': password_hash,
                        'require_password_change': False
                    }}
                )
                
                session['require_password_change'] = False
                flash("Đổi mật khẩu thành công! Tài khoản của bạn đã được kích hoạt bảo mật.", "success")
                
                if session.get('role') == 'patient':
                    return redirect(url_for('history.history_page'))
                return redirect(url_for('home.index'))
            except Exception as e:
                flash(f"Lỗi khi đổi mật khẩu: {e}", "error")
                
    return render_template('change_temp_password.html')

@auth_bp.before_app_request
def check_forced_password_change():
    # If they are logged in and must change password, restrict access
    if 'user_id' in session and session.get('require_password_change'):
        # Allow access only to logout, change_temp_password, and static files
        allowed_endpoints = ['auth.change_temp_password_page', 'auth.logout', 'static']
        if request.endpoint and request.endpoint not in allowed_endpoints:
            # For APIs, return JSON error
            if request.path.startswith('/api/'):
                return jsonify({'status': 'error', 'message': 'Yêu cầu đổi mật khẩu tạm thời để tiếp tục.'}), 403
            flash("Vui lòng đổi mật khẩu tạm thời trước khi truy cập trang khác.", "warning")
            return redirect(url_for('auth.change_temp_password_page'))

@auth_bp.route('/logout')
def logout():
    session.clear()
    flash("Bạn đã đăng xuất tài khoản thành công.", "success")
    return redirect(url_for('auth.login_page'))


@auth_bp.route('/api/quick_add_patient', methods=['POST'])
@api_login_required
@api_doctor_required
def api_quick_add_patient():
    try:
        data = request.get_json() or {}
        fullname = data.get('fullname', '').strip()
        email = data.get('email', '').strip()
        phone = data.get('phone', '').strip()
        dob = data.get('dob', '').strip()
        
        if not fullname:
            return jsonify({'status': 'error', 'message': 'Họ và tên bệnh nhân không được để trống.'}), 400
            
        import unicodedata
        # Helper to normalize text (strip accents)
        def clean_text(val):
            if not val:
                return ""
            val = unicodedata.normalize('NFKD', val)
            val = "".join([c for c in val if not unicodedata.combining(c)])
            val = val.replace('đ', 'd').replace('Đ', 'D')
            return val

        # 1. Generate password as BệnhNhân_NgàySinh (e.g. NguyenVanA_15081995)
        # Clean name: capitalize each word, keep only letters
        cleaned_name_parts = re.sub(r'[^a-zA-Z\s]', '', clean_text(fullname)).split()
        name_part = "".join([w.capitalize() for w in cleaned_name_parts])
        # Clean DOB: digits only
        dob_digits = re.sub(r'\D', '', dob) if dob else "123456"
        generated_password = f"{name_part}_{dob_digits}"

        # 2. Generate username
        base_username = re.sub(r'[^a-zA-Z0-9]', '', clean_text(fullname)).lower()
        # Find year in DOB
        birth_year = ""
        if dob:
            match = re.search(r'\b(19\d{2}|20\d{2})\b', dob)
            if match:
                birth_year = match.group(1)
                
        if birth_year:
            username = f"{base_username}{birth_year}"
        else:
            username = f"{base_username}{random.randint(1000, 9999)}"
            
        # Register user in DB with extra fields and require_password_change=True
        result = register_user(username, generated_password, fullname, role='patient', email=email, phone=phone, dob=dob, require_password_change=True)
        
        if result == "exists":
            # If username already exists, try appending a random suffix
            username = f"{username}{random.randint(10, 99)}"
            result = register_user(username, generated_password, fullname, role='patient', email=email, phone=phone, dob=dob, require_password_change=True)
            
        if result == "exists":
            return jsonify({'status': 'error', 'message': f'Tên đăng nhập "@{username}" đã tồn tại trên hệ thống.'}), 400
        elif result:
            return jsonify({
                'status': 'success',
                'message': 'Kích hoạt tài khoản bệnh nhân mới thành công!',
                'patient': {
                    'id': result,
                    'fullname': fullname,
                    'username': username,
                    'password': generated_password
                }
            }), 200
        else:
            return jsonify({'status': 'error', 'message': 'Không thể tạo bệnh nhân. Vui lòng kiểm tra dữ liệu.'}), 500
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Lỗi hệ thống: {str(e)}'}), 500
