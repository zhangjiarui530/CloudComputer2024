from flask import Flask, request, jsonify,render_template
import os
import librosa
import numpy as np
import logging
import time
from scipy.spatial.distance import cosine
import warnings
from pydub import AudioSegment


from tenacity import retry_if_exception_message

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


@app.route('/')
def home():
    return render_template('index18.html')

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


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)