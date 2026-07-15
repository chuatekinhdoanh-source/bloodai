from flask import Blueprint, render_template, request, redirect, url_for, session, flash, jsonify
from controllers.auth_controller import login_required, doctor_required
from models import (
    create_appointment, get_appointments, update_appointment_status,
    get_all_doctors, get_user_by_id
)

appointment_bp = Blueprint('appointment', __name__)

@appointment_bp.route('/appointments', methods=['GET'])
@login_required
def appointments_page():
    try:
        # Lấy danh sách lịch hẹn của tài khoản hiện tại
        appointments = get_appointments(session['user_id'], session['role'])
        
        # Nếu là Bệnh nhân, cần lấy thêm danh sách Bác sĩ để đặt lịch
        doctors = []
        if session.get('role') == 'patient':
            doctors = get_all_doctors()
            
        return render_template('appointments.html', 
                               appointments=appointments, 
                               doctors=doctors,
                               active_page='appointments')
    except Exception as e:
        return f"<h3>Lỗi tải lịch hẹn: {e}</h3>"

@appointment_bp.route('/appointments/book', methods=['POST'])
@login_required
def book_appointment():
    try:
        if session.get('role') != 'patient':
            flash("Chỉ Bệnh nhân mới được phép đặt lịch hẹn khám.", "error")
            return redirect(url_for('appointment.appointments_page'))
            
        doctor_id = request.form['doctor_id']
        appointment_date = request.form['appointment_date']
        appointment_time = request.form['appointment_time']
        reason = request.form['reason']
        
        # Tra cứu tên Bác sĩ
        doctor = get_user_by_id(doctor_id)
        if not doctor or doctor['role'] != 'doctor':
            flash("Bác sĩ được chọn không hợp lệ.", "error")
            return redirect(url_for('appointment.appointments_page'))
            
        res = create_appointment(
            patient_id=session['user_id'],
            patient_name=session['fullname'],
            doctor_id=doctor_id,
            doctor_name=doctor['fullname'],
            appointment_date=appointment_date,
            appointment_time=appointment_time,
            reason=reason
        )
        if res:
            flash("Đăng ký lịch hẹn khám thành công! Vui lòng chờ bác sĩ xác nhận.", "success")
        else:
            flash("Đăng ký lịch hẹn thất bại. Vui lòng thử lại.", "error")
            
        return redirect(url_for('appointment.appointments_page'))
    except Exception as e:
        return f"<h3>Lỗi khi đặt lịch hẹn: {e}</h3>"

@appointment_bp.route('/appointments/status/<string:appointment_id>', methods=['POST'])
@login_required
@doctor_required
def update_status(appointment_id):
    try:
        status = request.form['status']
        res = update_appointment_status(appointment_id, status)
        
        status_map = {
            'confirmed': 'Đã xác nhận',
            'cancelled': 'Đã hủy',
            'completed': 'Đã hoàn thành'
        }
        status_text = status_map.get(status, status)
        
        if res:
            flash(f"Đã cập nhật trạng thái lịch hẹn thành '{status_text}'.", "success")
        else:
            flash("Cập nhật trạng thái thất bại.", "error")
            
        return redirect(url_for('appointment.appointments_page'))
    except Exception as e:
        return f"<h3>Lỗi khi cập nhật lịch hẹn: {e}</h3>"
