from flask import Blueprint, render_template, request, jsonify, redirect, url_for, session, flash
from controllers.auth_controller import login_required, api_login_required, doctor_required, api_doctor_required
from models import get_predictions, delete_prediction

history_bp = Blueprint('history', __name__)

@history_bp.route('/history')
@login_required
def history_page():
    try:
        disease_filter = request.args.get('filter')
        predictions = get_predictions(session['user_id'], session['role'], disease_filter=disease_filter)
        
        # Thống kê tổng số lượng (chỉ của tài khoản đang đăng nhập theo vai trò)
        all_preds = get_predictions(session['user_id'], session['role'])
        total = len(all_preds)
        risk = sum(1 for p in all_preds if p['is_positive'] == 1)
        normal = total - risk
        
        stats = {
            'total': total,
            'normal': normal,
            'risk': risk
        }
        
        return render_template('history.html', 
                               predictions=predictions, 
                               stats=stats, 
                               current_filter=disease_filter)
    except Exception as e:
        return f"<h3>Lỗi lịch sử: {e}</h3>"

@history_bp.route('/delete/<string:prediction_id>', methods=['POST'])
@login_required
@doctor_required
def delete_prediction_route(prediction_id):
    try:
        delete_prediction(prediction_id)
        return redirect(url_for('history.history_page'))
    except Exception as e:
        return f"<h3>Lỗi khi xóa bản ghi: {e}</h3>"

# ==========================================================
#                   HỆ THỐNG RESTful APIs (JSON)
# ==========================================================

@history_bp.route('/api/history', methods=['GET'])
@api_login_required
def api_get_history():
    try:
        disease_filter = request.args.get('filter')
        limit = request.args.get('limit', default=100, type=int)
        
        predictions = get_predictions(session['user_id'], session['role'], disease_filter=disease_filter, limit=limit)
        return jsonify({
            'status': 'success',
            'count': len(predictions),
            'data': predictions
        }), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@history_bp.route('/api/history/<string:prediction_id>', methods=['DELETE'])
@api_login_required
@api_doctor_required
def api_delete_prediction(prediction_id):
    try:
        delete_prediction(prediction_id)
        return jsonify({
            'status': 'success',
            'message': f'Đã xóa bản ghi chẩn đoán {prediction_id} thành công.'
        }), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
