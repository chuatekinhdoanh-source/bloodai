from flask import Blueprint, render_template, redirect, url_for, session
from controllers.auth_controller import login_required, doctor_required
from models import get_all_patients

home_bp = Blueprint('home', __name__)

@home_bp.route('/')
@login_required
def index():
    # Bệnh nhân tự động điều hướng sang trang lịch sử chẩn đoán cá nhân
    if session.get('role') == 'patient':
        return redirect(url_for('history.history_page'))
        
    patients = get_all_patients()
    return render_template('index.html', patients=patients)

@home_bp.route('/anemia')
@login_required
@doctor_required
def anemia():
    patients = get_all_patients()
    return render_template('anemia.html', patients=patients)

@home_bp.route('/liver')
@login_required
@doctor_required
def liver():
    patients = get_all_patients()
    return render_template('liver.html', patients=patients)

@home_bp.route('/kidney')
@login_required
@doctor_required
def kidney():
    patients = get_all_patients()
    return render_template('kidney.html', patients=patients)
