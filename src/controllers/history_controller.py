from flask import Blueprint, render_template, request, jsonify, redirect, url_for, session, flash
from controllers.auth_controller import login_required, api_login_required, doctor_required, api_doctor_required
from models import get_predictions, delete_prediction
from datetime import datetime

history_bp = Blueprint('history', __name__)

def group_predictions_into_sessions(predictions):
    """Gom nhóm các bản ghi chẩn đoán rời rạc thành các phiên khám tổng thể."""
    sessions = []
    
    for pred in predictions:
        pred_session_id = pred.get('session_id')
        pred_patient_id = pred.get('patient_id')
        
        # Parse thời gian để so sánh khoảng cách
        pred_time = None
        if pred.get('created_at'):
            try:
                pred_time = datetime.strptime(pred['created_at'], '%Y-%m-%d %H:%M:%S')
            except Exception:
                pred_time = None
                
        matched_session = None
        for sess in sessions:
            # 1. So khớp theo session_id nếu có
            if pred_session_id and sess['session_id'] == pred_session_id:
                matched_session = sess
                break
            # 2. Dự phòng: khớp theo bệnh nhân + thời gian gần nhau (trong vòng 5 phút) cho dữ liệu cũ
            if not pred_session_id and not sess['session_id']:
                if sess['patient_id'] == pred_patient_id and pred_time and sess['time_obj']:
                    diff = abs((sess['time_obj'] - pred_time).total_seconds())
                    if diff <= 300: # 5 phút
                        matched_session = sess
                        break
                        
        if matched_session:
            matched_session['predictions'].append(pred)
            if not matched_session['session_id'] and pred_session_id:
                matched_session['session_id'] = pred_session_id
        else:
            sessions.append({
                'session_id': pred_session_id,
                'patient_id': pred_patient_id,
                'patient_name': pred.get('patient_name', 'Chưa rõ'),
                'doctor_id': pred.get('doctor_id', ''),
                'doctor_name': pred.get('doctor_name', 'Chưa rõ'),
                'created_at': pred.get('created_at', ''),
                'time_obj': pred_time,
                'predictions': [pred]
            })
            
    # Xây dựng các huy hiệu hiển thị (disease_badges) cho mỗi phiên
    for sess in sessions:
        badges = []
        # Loại trùng lặp loại bệnh lý hiển thị
        seen_types = set()
        for p in sess['predictions']:
            dt = p['disease_type']
            if dt in seen_types:
                continue
            seen_types.add(dt)
            
            if dt == 'screening':
                badges.append({
                    'type': 'screening',
                    'name': 'Sàng Lọc Tổng Quát',
                    'class': 'badge-screening',
                    'is_positive': p['is_positive']
                })
            elif dt == 'diabetes':
                badges.append({
                    'type': 'diabetes',
                    'name': 'Tiểu Đường',
                    'class': 'badge-diabetes',
                    'is_positive': p['is_positive']
                })
            elif dt == 'anemia':
                badges.append({
                    'type': 'anemia',
                    'name': 'Thiếu Máu',
                    'class': 'badge-anemia',
                    'is_positive': p['is_positive']
                })
            elif dt == 'liver':
                badges.append({
                    'type': 'liver',
                    'name': 'Bệnh Gan',
                    'class': 'badge-liver',
                    'is_positive': p['is_positive']
                })
            elif dt == 'kidney':
                badges.append({
                    'type': 'kidney',
                    'name': 'Bệnh Thận',
                    'class': 'badge-kidney',
                    'is_positive': p['is_positive']
                })
        sess['disease_badges'] = badges
        
        # Dọn dẹp time_obj để tránh lỗi JSON serializable
        if 'time_obj' in sess:
            del sess['time_obj']
            
    return sessions

@history_bp.route('/history')
@login_required
def history_page():
    try:
        disease_filter = request.args.get('filter')
        
        # Lấy tất cả chẩn đoán để tính toán thống kê và gom nhóm đầy đủ
        all_preds = get_predictions(session['user_id'], session['role'])
        sessions = group_predictions_into_sessions(all_preds)
        
        # Tính toán thống kê theo số lượng phiên (session)
        total = len(sessions)
        risk = sum(1 for s in sessions if any(p.get('is_positive') == 1 for p in s['predictions'] if p.get('disease_type') != 'screening'))
        normal = total - risk
        
        stats = {
            'total': total,
            'normal': normal,
            'risk': risk
        }
        
        # Lọc danh sách hiển thị theo loại bệnh lý nếu được chọn
        if disease_filter:
            filtered_sessions = []
            for s in sessions:
                if any(p.get('disease_type') == disease_filter for p in s['predictions']):
                    filtered_sessions.append(s)
            sessions = filtered_sessions
            
        return render_template('history.html', 
                               predictions=sessions, # truyền danh sách phiên khám gom nhóm
                               stats=stats, 
                               current_filter=disease_filter)
    except Exception as e:
        return f"<h3>Lỗi lịch sử: {e}</h3>"

@history_bp.route('/delete_session', methods=['POST'])
@login_required
@doctor_required
def delete_session_route():
    try:
        session_id = request.form.get('session_id')
        patient_id = request.form.get('patient_id')
        created_at = request.form.get('created_at')
        
        from models.db import db
        predictions_col = db['predictions']
        
        if session_id and session_id != 'None' and session_id != '':
            predictions_col.delete_many({'session_id': session_id})
            flash("Đã xóa toàn bộ phiên chẩn đoán thành công.", "success")
        elif patient_id and created_at:
            try:
                target_time = datetime.strptime(created_at, '%Y-%m-%d %H:%M:%S')
                candidates = list(predictions_col.find({'patient_id': patient_id}))
                to_delete = []
                for doc in candidates:
                    doc_time = doc.get('created_at')
                    if doc_time:
                        diff = abs((doc_time - target_time).total_seconds())
                        if diff <= 300: # 5 phút
                            to_delete.append(doc['_id'])
                if to_delete:
                    predictions_col.delete_many({'_id': {'$in': to_delete}})
                flash("Đã xóa toàn bộ phiên chẩn đoán thành công.", "success")
            except Exception as parse_err:
                flash(f"Lỗi phân tích thời gian khi xóa: {parse_err}", "error")
        else:
            flash("Thông tin xóa không hợp lệ.", "error")
            
        return redirect(url_for('history.history_page'))
    except Exception as e:
        return f"<h3>Lỗi khi xóa phiên chẩn đoán: {e}</h3>"

# ==========================================================
#                   HỆ THỐNG RESTful APIs (JSON)
# ==========================================================

@history_bp.route('/api/history', methods=['GET'])
@api_login_required
def api_get_history():
    try:
        disease_filter = request.args.get('filter')
        limit = request.args.get('limit', default=100, type=int)
        
        predictions = get_predictions(session['user_id'], session['role'], limit=limit)
        sessions = group_predictions_into_sessions(predictions)
        
        if disease_filter:
            sessions = [s for s in sessions if any(p.get('disease_type') == disease_filter for p in s['predictions'])]
            
        return jsonify({
            'status': 'success',
            'count': len(sessions),
            'data': sessions
        }), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@history_bp.route('/api/history/session', methods=['DELETE'])
@api_login_required
@api_doctor_required
def api_delete_session():
    try:
        data = request.get_json(silent=True) or request.form
        session_id = data.get('session_id')
        patient_id = data.get('patient_id')
        created_at = data.get('created_at')
        
        from models.db import db
        predictions_col = db['predictions']
        
        if session_id:
            predictions_col.delete_many({'session_id': session_id})
        elif patient_id and created_at:
            target_time = datetime.strptime(created_at, '%Y-%m-%d %H:%M:%S')
            candidates = list(predictions_col.find({'patient_id': patient_id}))
            to_delete = []
            for doc in candidates:
                doc_time = doc.get('created_at')
                if doc_time:
                    diff = abs((doc_time - target_time).total_seconds())
                    if diff <= 300:
                        to_delete.append(doc['_id'])
            if to_delete:
                predictions_col.delete_many({'_id': {'$in': to_delete}})
        else:
            return jsonify({'status': 'error', 'message': 'Thiếu tham số xóa.'}), 400
            
        return jsonify({
            'status': 'success',
            'message': 'Đã xóa toàn bộ phiên chẩn đoán thành công.'
        }), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
