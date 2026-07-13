import pymongo
import os

MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
DB_NAME = 'bloodcare_db'

client = None
db = None

try:
    client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
    db = client[DB_NAME]
except Exception as e:
    print(f"[DATABASE] Connection error: {e}")

def init_db():
    """Kiểm tra kết nối tới MongoDB và thiết lập indexes."""
    try:
        if client is not None:
            client.admin.command('ping')
            print(f"[DATABASE] Database initialized successfully at MongoDB: {MONGO_URI}")
            print(f"[DATABASE] Database: '{DB_NAME}'")
            
            # Đảm bảo username là duy nhất
            if db is not None:
                db['users'].create_index("username", unique=True)
                print("[DATABASE] Unique index on 'username' ensured.")
    except Exception as e:
        print(f"[DATABASE] WARNING: Could not connect to MongoDB server at {MONGO_URI}. Error: {e}")
        print("[DATABASE] Please make sure MongoDB service is running.")
