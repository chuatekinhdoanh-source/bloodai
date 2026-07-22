from models.db import init_db
from models.user import register_user, authenticate_user, get_all_patients, get_user_by_id
from models.prediction import save_prediction, get_predictions, delete_prediction, get_latest_patient_clinical_data, map_vietnamese_to_technical
from models.ai_models import diabetes_model, anemia_model, liver_model, kidney_model
from models.appointment import create_appointment, get_appointments, update_appointment_status, get_all_doctors
