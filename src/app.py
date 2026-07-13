from flask import Flask
import os
import sys

# Đảm bảo import được các module trong thư mục src/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import init_db
from controllers import auth_bp, home_bp, prediction_bp, history_bp

app = Flask(__name__, template_folder='../templates')
app.secret_key = os.environ.get('SECRET_KEY', 'bloodcare_secret_session_key_2026')

# ---- KHỞI TẠO CƠ SỞ DỮ LIỆU ----
init_db()

# ---- ĐĂNG KÝ CÁC BLUEPRINTS ----
app.register_blueprint(auth_bp)
app.register_blueprint(home_bp)
app.register_blueprint(prediction_bp)
app.register_blueprint(history_bp)

if __name__ == '__main__':
    app.run(debug=True)