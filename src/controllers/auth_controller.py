from flask import Blueprint, render_template, request, redirect, url_for, session, flash, jsonify
from functools import wraps
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
            flash(f"Chào mừng, BS. {user['fullname']}!" if user['role'] == 'doctor' else f"Chào mừng, BN. {user['fullname']}!", "success")
            
            # Điều hướng linh hoạt dựa trên vai trò
            if user['role'] == 'patient':
                return redirect(url_for('history.history_page'))
            return redirect(url_for('home.index'))
        else:
            flash("Tên đăng nhập hoặc mật khẩu không chính xác.", "error")
            
    return render_template('login.html')

@auth_bp.route('/logout')
def logout():
    session.clear()
    flash("Bạn đã đăng xuất tài khoản thành công.", "success")
    return redirect(url_for('auth.login_page'))
