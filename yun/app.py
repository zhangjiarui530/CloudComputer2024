from flask import Flask, render_template, request, jsonify, send_file
import os
import json
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS  # 新增导入
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import librosa
import numpy as np
import logging
import time
from scipy.spatial.distance import cosine
import warnings
from pydub import AudioSegment
from tenacity import retry_if_exception_message
from flask import Flask, request, jsonify, render_template, send_file
from pydub import AudioSegment
from aip import AipSpeech
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import os
from langchain_community.llms import Tongyi
from langchain_core.messages import HumanMessage,AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
import os
import subprocess
import random


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}) # 允许所有源
app.secret_key = 'dsdfsfdaxxx'
app.config['UPLOAD_FOLDER'] = 'uploads/'

# # 启用 CORS，允许来自 http://192.168.1.103:5000 的请求
# CORS(app, resources={r"/upload": {"origins": "http://192.168.1.103:5000"}})
users_file = 'static/json/user.json'
userName = ''

# 确保用户文件存在
if not os.path.exists(users_file):
    with open(users_file, 'w') as f:
        json.dump({}, f)


def save_new_user(username, password, name):
    with open(users_file, 'r+') as fl:
        users = json.load(fl)
        if username in users:
            return False
        # 密码哈希化后存储
        users[username] = [generate_password_hash(password), name]
        fl.seek(0)
        json.dump(users, fl)
        fl.truncate()
    return True


def verify_user(username, password):
    global userName
    userName = username
    with open(users_file) as f:
        users = json.load(f)
        # 用户存在且密码验证通过
        return username in users and check_password_hash(users[username][0], password)

warnings.filterwarnings("ignore")
# 设置 pydub 使用的 ffmpeg 路径
AudioSegment.ffmpeg = r"D:\FFmpeg\bin\ffmpeg.exe"
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'  # 设置文件上传目录

# 确保上传目录存在
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

class AudioComparator:
    def __init__(self, file_type='audio', file=None):
        self.file_type = file_type
        self.file = file

    def load_and_check_audio(self, file_path):
        """
        加载音频数据并检查其是否可播放。
        返回音频数据 y、采样率 sr 和一个布尔值表示音频是否可播放。
        """
        try:
            if self.file_type == 'audio':
                y, sr = librosa.load(file_path, sr=None)

            if y is None or y.size == 0:
                logging.info(f"{file_path}中没有音频数据")
                return False, None, None

            logging.info(f"音频时长: {len(y) / sr} seconds")
            return True, y, sr

        except Exception as e:
            logging.error(f"处理音频文件 {file_path} 时出错: {e}")
            return False, None, None

    def compare_audio_advanced(self, file1, file2):
        """
        加载音频文件并计算音频相似度
        """
        is_playable1, y1, sr1 = self.load_and_check_audio(file1)
        is_playable2, y2, sr2 = self.load_and_check_audio(file2)
        if not is_playable1:
            raise ValueError("被测的音频无法播放")
        if not is_playable2:
            raise ValueError("基准音频无法播放")
        s = time.time()

        # 检查采样率是否相同，如果不同则重新采样
        if sr1 != sr2:
            logging.info(f"采样率不同: file1 ({sr1}) vs file2 ({sr2})，正在进行重采样")
            y2 = librosa.resample(y2, orig_sr=sr2, target_sr=sr1)  # 将第二个音频的采样率重采样为第一个音频的采样率
            sr2 = sr1  # 将第二个音频的采样率更新为第一个音频的采样率

        # 提取MFCCs
        mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=13)
        mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2, n_mfcc=13)

        # 标准化MFCCs
        mfcc1 = (mfcc1 - np.mean(mfcc1)) / np.std(mfcc1)
        mfcc2 = (mfcc2 - np.mean(mfcc2)) / np.std(mfcc2)

        # 将MFCC矩阵展平为向量
        mfcc1_vector = mfcc1.flatten()
        mfcc2_vector = mfcc2.flatten()

        # 确保两个向量长度相同
        min_length = min(len(mfcc1_vector), len(mfcc2_vector))
        mfcc1_vector = mfcc1_vector[:min_length]
        mfcc2_vector = mfcc2_vector[:min_length]

        # 计算余弦相似度
        similarity = 1 - cosine(mfcc1_vector, mfcc2_vector)
        logging.info(f"比较音频的时间为:{(time.time() - s) * 1000}ms")
        return similarity

    def compare_audio_similarity(self, file1, file2):
        """
        比较两个音频之间的差异
        param file1: 第一个音频文件路径
        param file2: 第二个音频文件路径
        """
        try:
            similarity = self.compare_audio_advanced(file1, file2)
            logging.info(f"音频相似度: {similarity}")
            if similarity < 0.5:
                y1, sr1 = librosa.load(file1, sr=None)
                y2, sr2 = librosa.load(file2, sr=None)
                different_times = self.find_differences_in_time(y1, y2, sr1)
                logging.info(f"差异点的时间位置（秒）为: {different_times}")
                return False, "两个音频不相似！！！",similarity
            else:
                logging.info("两个音频是相似的")
                return True, "两个音频是相似的",similarity
        except ValueError as e:
            logging.info(f"处理音频文件中的异常：{e}")
            return False, f"发生错误: {str(e)}",similarity

    def find_differences_in_time(self, audio1, audio2, sr, threshold_method='auto', fixed_threshold=0.2):
        """
        寻找时间点的差异
        :param audio1: 第一个音频信号的数组
        :param audio2: 第二个音频信号的数组
        :param sr: 采样率
        :param threshold_method: 阈值选择方法 ('auto' or 'fixed')
        :param fixed_threshold: 如果选择固定阈值，使用的阈值量（默认值为0.2）
        """
        # 对齐长度
        min_length = min(len(audio1), len(audio2))
        audio1 = audio1[:min_length]
        audio2 = audio2[:min_length]

        # 计算差异
        difference = np.abs(audio1 - audio2)

        # 确定阈值
        if threshold_method == 'auto':
            # 使用差异数据的均值和标准差来设置自动阈值
            mean_diff = np.mean(difference)
            std_diff = np.std(difference)
            threshold = mean_diff + std_diff  # 设置为均值加一个标准差
            logging.info(f"差异值为：{threshold}")
        else:
            # 使用固定阈值
            threshold = fixed_threshold

        # 找出差异点
        different_points = np.where(difference > threshold)[0]
        different_times = different_points / sr
        return different_times

def convert_mp3_to_wav(mp3_file_path, wav_file_path):
    """将 MP3 文件转换为 WAV 格式"""
    audio = AudioSegment.from_mp3(mp3_file_path)
    audio = audio.set_frame_rate(16000)  # 设置采样率为16000 Hz
    # audio.export(wav_file_path, format="wav")  # 使用 PCM 编码导出

    audio = audio.set_sample_width(2)  # 使用 2 字节采样（16-bit）
    audio = audio.set_channels(1)  # 设置为单声道（语音识别通常更适合单声道）
    audio = audio + 10  # 增加10dB

    audio.export(wav_file_path, format="wav", codec="pcm_s16le")  # 使用 PCM 编码导出

CORS(app, resources={r"/*": {"origins": "*"}}) # 允许所有源
@app.route('/', methods=['GET'])
def index():
    return render_template("homepage.html")

CORS(app)
@app.route('/login', methods=['POST'])
def login():
    username = request.form.get("username")
    password = request.form.get('password')
    if verify_user(username, password):
        return render_template('homepage.html')
    else:
        return render_template('login.html', error='用户名或密码错误')

CORS(app)
@app.route('/toRegister', methods=['GET'])
def toRegister():
    return render_template('register.html')

CORS(app)
@app.route('/register', methods=['POST'])
def register():
    username = request.form.get("username")
    password = request.form.get('password')
    password2 = request.form.get('password2')
    name = request.form.get('name')

    if password2 != password:
        return render_template('register.html', error='两次输入的密码不一致')

    if save_new_user(username, password, name):
        return render_template('login.html')
    else:
        return render_template('register.html', error='用户已存在，注册失败')


@app.route('/homepage', methods=['GET'])
def homepage():
    return render_template('homepage.html')


@app.route('/index02')
def index02():
    return render_template('index02.html')


# @app.route('/api/message', methods=['POST'])
# def handle_message():
#     data = request.json
#     user_message = data.get('message', '')
#     response_message = f"收到您的消息: {user_message}"
#     return jsonify({'response': response_message})
CORS(app, resources={r"/*": {"origins": "*"}}) # 允许所有源
@app.route('/api/message', methods=['POST'])
def handle_message():
    data = request.json
    user_message = data.get('message', '')
    api_key = "cd493791757b49fdbc0eeba1367608f0.xGgQTGgq7e7jIORn"
    model = ChatOpenAI(
        temperature=0.95,
        model="glm-4-flash",
        openai_api_key=api_key,
        openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
    )

    # 粤语到普通话翻译
    messages = [
        SystemMessage(
            content="首先判断我说的是粤语还是普通话，是粤语则翻译成普通话，是普通话则翻译成粤语，只输出翻译内容"),
        HumanMessage(content=user_message),
    ]
    response = model.invoke(messages)

    # 仅输出翻译后的文本
    response_message = response.content  # 从响应中提取文本字段
    return jsonify({'response': response_message})

# Setup upload folders
UPLOAD_FOLDER = 'uploads/'  # Audio upload directory
UPLOAD_FOLDER1 = 'uploadstalk/'  # Additional audio directory

# Ensure upload directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER1, exist_ok=True)

CORS(app, resources={r"/*": {"origins": "*"}}) # 允许所有源
@app.route('/index18')
def index18():
    return render_template('index18.html')


# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'audio' not in request.files:
#         return jsonify({'error': '没有音频文件'}), 400
#
#     audio_file = request.files['audio']
#     file_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
#     audio_file.save(file_path)  # 保存上传的音频文件
#     print(f'上传文件已保存到: {file_path}')  # 打印文件保存路径
#     return jsonify({'text': '111'})  # 返回假设的文本结果
CORS(app, resources={r"/*": {"origins": "*"}}) # 允许所有源
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'audio' not in request.files:
        return jsonify({'error': '没有音频文件'}), 400

    audio_file = request.files['audio']
    file_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
    audio_file.save(file_path)  # 保存上传的音频文件
    print(f'上传文件已保存到: {file_path}')  # 打印文件保存路径
    # 假设文件路径是
    file1 = '../data/yue/clips/common_voice_yue_31172849.mp3'
    wav_file_path = 'converted_audio.wav'
    convert_mp3_to_wav(file1, wav_file_path)
    file2 = file_path
    comparator = AudioComparator(file_type='audio')
    # 比较音频文件
    result, message, similarity = comparator.compare_audio_similarity(wav_file_path, file2)
    if similarity > 0.58:
        similarity = similarity * 1.3
    else:
        similarity = similarity * 0.5
    ret_message = message + "结果评分" + str(similarity)
    return jsonify({'text': ret_message})  # 返回假设的文本结果
CORS(app, resources={r"/*": {"origins": "*"}}) # 允许所有源
@app.route('/index26')
def index26():
    return render_template('index26.html')


# @app.route('/upload/talk', methods=['POST'])
# def upload_talk_file():
#     if 'audio' not in request.files:
#         return jsonify({'error': '没有音频文件'}), 400
#
#     audio_file = request.files['audio']
#     file_path = os.path.join(UPLOAD_FOLDER1, audio_file.filename)
#     audio_file.save(file_path)
#
#     return jsonify({'text': '111'})  # 返回假设的文本结果
CORS(app, resources={r"/*": {"origins": "*"}}) # 允许所有源
@app.route('/upload/talk', methods=['POST'])
def upload_talk_file():
    if 'audio' not in request.files:
        return jsonify({'error': '没有音频文件'}), 400

    audio_file = request.files['audio']
    file_path = os.path.join(UPLOAD_FOLDER1, audio_file.filename)
    audio_file.save(file_path)  # 保存上传的音频文件
    print(f'上传文件已保存到: {file_path}')  # 打印文件保存路径

    # 百度云语音识别的 API Key 和 Secret Key
    APP_ID = '6251315'
    API_KEY = 'Yz15TnhVhSoiX3p1uUfOLYCX'
    SECRET_KEY = '0HCL2dA6NIdK51Wz2jQVkqddwvnoAixb'

    # 初始化AipSpeech对象
    client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)
    # 设置 pydub 使用的 ffmpeg 路径
    AudioSegment.ffmpeg = r"D:\FFmpeg\bin\ffmpeg.exe"

    def convert_mp3_to_wav(mp3_file_path, wav_file_path):
        """将 MP3 文件转换为 WAV 格式"""
        audio = AudioSegment.from_mp3(mp3_file_path)
        audio = audio.set_frame_rate(16000)  # 设置采样率为16000 Hz
        # audio.export(wav_file_path, format="wav")  # 使用 PCM 编码导出

        audio = audio.set_sample_width(2)  # 使用 2 字节采样（16-bit）
        audio = audio.set_channels(1)  # 设置为单声道（语音识别通常更适合单声道）
        audio = audio + 10  # 增加10dB

        audio.export(wav_file_path, format="wav", codec="pcm_s16le")  # 使用 PCM 编码导出

    def convert_wav_to_wav(wav_file_path, output_wav_file_path):
        """直接使用 FFmpeg 转换 WAV 文件，并调整采样率、采样宽度、声道数和增益"""

        # FFmpeg 命令
        command = [
            "ffmpeg",
            "-i", wav_file_path,  # 输入文件x`
            "-ar", "16000",  # 设置采样率为16000 Hz
            "-ac", "1",  # 设置为单声道
            "-sample_fmt", "s16",  # 设置为 16-bit
            "-filter:a", "volume=10dB",  # 增加 10 dB
            output_wav_file_path  # 输出文件路径
        ]

        try:
            # 执行 FFmpeg 命令
            subprocess.run(command, check=True)
            print(f"文件已成功转换并保存为 {output_wav_file_path}")
        except subprocess.CalledProcessError as e:
            print(f"转换失败：{e}")

    def speech_to_text(audio_file_path):
        """使用百度云语音识别将音频文件转为文本"""
        with open(audio_file_path, 'rb') as f:
            audio_data = f.read()

            result = client.asr(audio_data, 'wav', 16000, {
                'dev_pid': 1637,  # 语言参数设置为粤语
            })

            if result.get('err_no') == 0:
                return result['result'][0]
            else:
                return None

    # 使用示例：首先将 MP3 文件转换为 WAV 文件
    # mp3_file_path = './data/yue/clips/common_voice_yue_31172849.mp3'
    wav_file_path = file_path
    # 生成一个范围在 0 到 1000 之间的随机整数
    random_number = random.randint(0, 1000)
    output_wav_file_path = f'./uploads/output{random_number}.wav'
    convert_wav_to_wav(wav_file_path, output_wav_file_path)
    # convert_mp3_to_wav(mp3_file_path, wav_file_path)

    # 然后将转换后的 WAV 文件传给百度云进行语音识别
    transcription = speech_to_text(output_wav_file_path)

    api_key = "cd493791757b49fdbc0eeba1367608f0.xGgQTGgq7e7jIORn"
    model = ChatOpenAI(
        temperature=0.95,
        model="glm-4-flash",
        openai_api_key=api_key,
        openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
    )

    # 粤语到普通话翻译
    messages = [
        SystemMessage(content="这是一句粤语，翻译成普通话，只输出翻译内容"),
        HumanMessage(content=transcription),  # 使用 transcription 变量替换原有文本
    ]

    response = model.invoke(messages)

    # 仅输出翻译后的文本
    translated_text = response.content  # 从响应中提取文本字段

    os.environ["DASHSCOPE_API_KEY"] = "sk-9041beedef014f4cb57ff1ef1dc6cd33"
    # 提示模板
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一个有用的助手。尽你所能回答所有问题。",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    # 历史会话存储
    store = {}

    # 获取会话历史
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    # 使用 Tongyi LLM
    model = Tongyi()

    # k为记录最近历史数量
    def filter_messages(messages, k=10):
        return messages[-k:]

    chain = (
            RunnablePassthrough.assign(messages=lambda x: filter_messages(x["messages"]))
            | prompt
            | model
    )

    # 历史消息
    with_message_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="messages",
    )

    config = {"configurable": {"session_id": "abc2"}}

    response = with_message_history.invoke(
        {"messages": [HumanMessage(content=f"{translated_text}")]},
        config=config
    )

    # 返回音频文件给前端
    return jsonify({'text': response})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)