from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "<h1>Chào mừng đến với hệ thống BloodCare AI 2.0</h1><p>Hệ thống AI phân tích xét nghiệm máu đang được xây dựng...</p>"

if __name__ == '__main__':
    app.run(debug=True)