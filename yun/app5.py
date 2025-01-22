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
UPLOAD_FOLDER = 'uploadstalk'  # 设置文件上传目录

# 确保上传目录存在
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def home():
    return render_template('index26.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'audio' not in request.files:
        return jsonify({'error': '没有音频文件'}), 400

    audio_file = request.files['audio']
    file_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
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
            "-i", wav_file_path,  # 输入文件
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
    app.run(host='127.0.0.1', port=5000, debug=True)