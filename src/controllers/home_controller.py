from flask import Blueprint, render_template, redirect, url_for, session, request
from controllers.auth_controller import login_required, doctor_required
from models import get_all_patients

home_bp = Blueprint('home', __name__)

def _get_patient_and_screening_data(patient_id):
    if not patient_id:
        return None, None
        
    from models import get_user_by_id, get_latest_patient_clinical_data, map_vietnamese_to_technical
    patient = get_user_by_id(patient_id)
    if not patient:
        return None, None
        
    # Check if there is data in session for this patient
    session_data = session.get('screening_data')
    if session_data and session_data.get('patient_id') == patient_id:
        screening_data = session_data
    else:
        # Load from database history
        db_data = get_latest_patient_clinical_data(patient_id)
        if db_data:
            screening_data = map_vietnamese_to_technical(db_data)
            screening_data['patient_id'] = patient_id
        else:
            screening_data = {'patient_id': patient_id}
            
    return patient, screening_data

@home_bp.route('/')
@login_required
def index():
    # Bệnh nhân tự động điều hướng sang trang lịch sử chẩn đoán cá nhân
    if session.get('role') == 'patient':
        return redirect(url_for('history.history_page'))
    return redirect(url_for('home.screening'))

@home_bp.route('/diabetes')
@login_required
@doctor_required
def diabetes():
    patients = get_all_patients()
    patient_id = request.args.get('patient_id')
    patient, screening_data = _get_patient_and_screening_data(patient_id)
    
    if not screening_data and request.args.get('from_screening') == '1':
        screening_data = session.get('screening_data')
        if screening_data and 'patient_id' in screening_data:
            from models import get_user_by_id
            patient = get_user_by_id(screening_data['patient_id'])
            
    next_disease = request.args.get('next_disease', '')
    from_pathway = request.args.get('from_pathway', '')
    session_id = request.args.get('session_id', '')
    return render_template('index.html', patients=patients, patient=patient, screening_data=screening_data, next_disease=next_disease, from_pathway=from_pathway, session_id=session_id)

@home_bp.route('/anemia')
@login_required
@doctor_required
def anemia():
    patients = get_all_patients()
    patient_id = request.args.get('patient_id')
    patient, screening_data = _get_patient_and_screening_data(patient_id)
    
    if not screening_data and request.args.get('from_screening') == '1':
        screening_data = session.get('screening_data')
        if screening_data and 'patient_id' in screening_data:
            from models import get_user_by_id
            patient = get_user_by_id(screening_data['patient_id'])
            
    next_disease = request.args.get('next_disease', '')
    from_pathway = request.args.get('from_pathway', '')
    session_id = request.args.get('session_id', '')
    return render_template('anemia.html', patients=patients, patient=patient, screening_data=screening_data, next_disease=next_disease, from_pathway=from_pathway, session_id=session_id)

@home_bp.route('/liver')
@login_required
@doctor_required
def liver():
    patients = get_all_patients()
    patient_id = request.args.get('patient_id')
    patient, screening_data = _get_patient_and_screening_data(patient_id)
    
    if not screening_data and request.args.get('from_screening') == '1':
        screening_data = session.get('screening_data')
        if screening_data and 'patient_id' in screening_data:
            from models import get_user_by_id
            patient = get_user_by_id(screening_data['patient_id'])
            
    next_disease = request.args.get('next_disease', '')
    from_pathway = request.args.get('from_pathway', '')
    session_id = request.args.get('session_id', '')
    return render_template('liver.html', patients=patients, patient=patient, screening_data=screening_data, next_disease=next_disease, from_pathway=from_pathway, session_id=session_id)

@home_bp.route('/kidney')
@login_required
@doctor_required
def kidney():
    patients = get_all_patients()
    patient_id = request.args.get('patient_id')
    patient, screening_data = _get_patient_and_screening_data(patient_id)
    
    if not screening_data and request.args.get('from_screening') == '1':
        screening_data = session.get('screening_data')
        if screening_data and 'patient_id' in screening_data:
            from models import get_user_by_id
            patient = get_user_by_id(screening_data['patient_id'])
            
    next_disease = request.args.get('next_disease', '')
    from_pathway = request.args.get('from_pathway', '')
    session_id = request.args.get('session_id', '')
    return render_template('kidney.html', patients=patients, patient=patient, screening_data=screening_data, next_disease=next_disease, from_pathway=from_pathway, session_id=session_id)

@home_bp.route('/screening')
@login_required
@doctor_required
def screening():
    patients = get_all_patients()
    return render_template('screening.html', patients=patients)

