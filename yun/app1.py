from flask import Flask, request, jsonify, render_template
import os

app = Flask(__name__)

# 创建一个保存录音的目录
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # 创建 uploads 文件夹（如果不存在的话）

@app.route('/')
def home():
    return render_template('index11.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'audio' not in request.files:
        return jsonify({'error': '没有文件部分'}), 400

    file = request.files['audio']

    if file.filename == '':
        return jsonify({'error': '选择一个文件'}), 400

    # 保存音频文件
    audio_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(audio_path)

    # 这里可以添加对音频文件的进一步处理（例如，识别文本等）

    # 返回相应的文本
    return jsonify({'text': '111'}), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)