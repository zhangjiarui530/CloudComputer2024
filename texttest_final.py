from pydub import AudioSegment
from aip import AipSpeech
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import librosa
import numpy as np
from scipy.spatial.distance import cosine


# # 百度云语音识别的 API Key 和 Secret Key
# APP_ID = '6251315'
# API_KEY = 'Yz15TnhVhSoiX3p1uUfOLYCX'
# SECRET_KEY = '0HCL2dA6NIdK51Wz2jQVkqddwvnoAixb'
#
# # 初始化AipSpeech对象
# client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)
# # 设置 pydub 使用的 ffmpeg 路径
# AudioSegment.ffmpeg = r"D:\FFmpeg\bin\ffmpeg.exe"
#
#
# def convert_mp3_to_wav(mp3_file_path, wav_file_path):
#     """将 MP3 文件转换为 WAV 格式"""
#     audio = AudioSegment.from_mp3(mp3_file_path)
#     audio = audio.set_frame_rate(16000)  # 设置采样率为16000 Hz
#     # audio.export(wav_file_path, format="wav")  # 使用 PCM 编码导出
#
#     audio = audio.set_sample_width(2)  # 使用 2 字节采样（16-bit）
#     audio = audio.set_channels(1)  # 设置为单声道（语音识别通常更适合单声道）
#     audio = audio + 10  # 增加10dB
#
#     audio.export(wav_file_path, format="wav", codec="pcm_s16le")  # 使用 PCM 编码导出
#
#
# def speech_to_text(audio_file_path):
#     """使用百度云语音识别将音频文件转为文本"""
#     with open(audio_file_path, 'rb') as f:
#         audio_data = f.read()
#
#         result = client.asr(audio_data, 'wav', 16000, {
#             'dev_pid': 1637,  # 语言参数设置为粤语
#         })
#
#         if result.get('err_no') == 0:
#             return result['result'][0]
#         else:
#             return None
#
#
# # 使用示例：首先将 MP3 文件转换为 WAV 文件
# mp3_file_path = './data/yue/clips/common_voice_yue_31172849.mp3'
# wav_file_path = 'converted_audio.wav'
# convert_mp3_to_wav(mp3_file_path, wav_file_path)
#
# # 然后将转换后的 WAV 文件传给百度云进行语音识别
# transcription = speech_to_text(wav_file_path)
#
#
# api_key = "cd493791757b49fdbc0eeba1367608f0.xGgQTGgq7e7jIORn"
# model=ChatOpenAI(
# temperature=0.95,
# model="glm-4-flash",
# openai_api_key=api_key,
# openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
# )
#
# # 粤语到普通话翻译
# messages = [
#     SystemMessage(content="这是一句粤语，翻译成普通话，只输出翻译内容"),
#     HumanMessage(content=transcription),  # 使用 transcription 变量替换原有文本
# ]
#
# response = model.invoke(messages)
#
# # 仅输出翻译后的文本
# translated_text = response.content  # 从响应中提取文本字段


# 1. 加载MP3文件并提取MFCC特征
def extract_mfcc(file_path):
    # 加载音频文件
    y, sr = librosa.load(file_path)

    # 提取MFCC特征，默认提取13维MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # 将MFCC特征平均化（每一帧的特征平均）
    mfcc_mean = np.mean(mfcc, axis=1)

    return mfcc_mean


# 2. 计算两段音频的余弦相似度
def calculate_similarity(file1, file2):
    # 提取MFCC特征
    mfcc1 = extract_mfcc(file1)
    mfcc2 = extract_mfcc(file2)

    # 计算余弦相似度
    similarity = 1 - cosine(mfcc1, mfcc2)

    return similarity

def convert_mp3_to_wav(mp3_file_path, wav_file_path):
    """将 MP3 文件转换为 WAV 格式"""
    audio = AudioSegment.from_mp3(mp3_file_path)
    audio = audio.set_frame_rate(16000)  # 设置采样率为16000 Hz
    # audio.export(wav_file_path, format="wav")  # 使用 PCM 编码导出

    audio = audio.set_sample_width(2)  # 使用 2 字节采样（16-bit）
    audio = audio.set_channels(1)  # 设置为单声道（语音识别通常更适合单声道）
    audio = audio + 10  # 增加10dB

    audio.export(wav_file_path, format="wav", codec="pcm_s16le")  # 使用 PCM 编码导出

# 示例使用
file1 = './data/yue/clips/common_voice_yue_31172849.mp3'  # 替换为你的第一个音频文件路径
wav_file_path_1 = 'converted_audio.wav1'
convert_mp3_to_wav(file1, wav_file_path_1)
file2 = './data/yue/clips/test.mp3'  # 替换为你的第二个音频文件路径
wav_file_path_2 = 'converted_audio.wav2'
convert_mp3_to_wav(file1, wav_file_path_2)

similarity_score = calculate_similarity(wav_file_path_1, wav_file_path_2)

print(f"音频匹配度（相似度）：{similarity_score:.4f}")