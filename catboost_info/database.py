from pymongo import MongoClient

# Thiết lập kết nối đến MongoDB cục bộ
client = MongoClient('mongodb://localhost:27017/')

# Truy cập vào database bloodcare_db
db = client['bloodcare_db']

# Tạo hoặc chọn một collection (ví dụ: 'patient_data')
collection = db['patient_data']

print("Kết nối thành công tới bloodcare_db!")