import sys
import os
import wave
import threading
import time
import pyaudio
import torch
import whisper
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, 
                            QVBoxLayout, QHBoxLayout, QComboBox, QTextEdit, 
                            QWidget, QGroupBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
import torch
# 假设TextExtractor已在models模块中定义，包含predict(text)方法
from models.TextExtractor import TextExtractor

# 实际录音和转录实现
class RecorderTranscriber:
    def __init__(self, save_folder="./audio_files", model_name="base"):
        self.save_folder = save_folder
        self.model_name = model_name
        os.makedirs(save_folder, exist_ok=True)
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.recording = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.whisper_model = whisper.load_model(model_name, device=self.device)
    
    def start_recording(self, output_file):
        self.output_file = output_file
        self.frames = []
        self.recording = True
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=44100,
            input=True,
            frames_per_buffer=1024
        )
        threading.Thread(target=self._record, daemon=True).start()
    
    def _record(self):
        while self.recording:
            data = self.stream.read(1024)
            self.frames.append(data)
    
    def stop_recording(self):
        if self.recording:
            self.recording = False
            self.stream.stop_stream()
            self.stream.close()
            with wave.open(self.output_file, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(44100)
                wf.writeframes(b''.join(self.frames))
    
    def transcribe(self, audio_file):
        result = self.whisper_model.transcribe(audio_file, language="zh")
        return result["text"]
    
    def record_and_transcribe(self, output_file):
        try:
            self.start_recording(output_file)
            return {"status": "recording", "output_file": output_file}
        except Exception as e:
            return {"error": f"录音失败: {str(e)}"}

# 录音线程类（管理时间更新和异步处理）
class RecordingThread(QThread):
    update_time = pyqtSignal(int)
    recording_finished = pyqtSignal(dict)
    
    def __init__(self, recorder, output_file):
        super().__init__()
        self.recorder = recorder
        self.output_file = output_file
        self.is_recording = False
        self.start_time = time.time()
    
    def run(self):
        self.is_recording = True
        self.recorder.start_recording(self.output_file)
        
        # 更新录音时间
        while self.is_recording:
            elapsed = int(time.time() - self.start_time)
            self.update_time.emit(elapsed)
            time.sleep(1)
    
    def stop_recording(self):
        self.is_recording = False
        self.recorder.stop_recording()
        transcription = self.recorder.transcribe(self.output_file)
        self.recording_finished.emit({"transcription": transcription})

class EmotionAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_emotion_analyzer()
        self.recorder = RecorderTranscriber()
        self.recording_thread = None
        self.init_ui()
    
    def init_emotion_analyzer(self):
        try:
            # 初始化情感分析器
            self.emotion_analyzer = TextExtractor(type="load",data_path = r"data\ch-simsv2s\meta.csv")
            self.emotion_available = True
            print("情感分析器初始化成功")
        except Exception as e:
            self.emotion_analyzer = None
            self.emotion_available = False
            print(f"情感分析器初始化失败: {str(e)}")
    
    def init_ui(self):
        self.setWindowTitle("情感分析器")
        self.setGeometry(100, 100, 800, 600)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # 标题
        title = QLabel("文本情感分析", font=QFont("黑体", 20, QFont.Bold), alignment=Qt.AlignCenter)
        layout.addWidget(title)
        
        # 控制组
        control_group = QGroupBox("录音控制")
        control_layout = QVBoxLayout(control_group)
        
        # 模型选择
        model_layout = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems(["base", "small", "medium", "large"])
        self.model_combo.currentTextChanged.connect(self.change_model)
        model_layout.addWidget(QLabel("选择识别模型:"))
        model_layout.addWidget(self.model_combo)
        control_layout.addLayout(model_layout)
        
        # 录音按钮
        self.record_btn = QPushButton("开始录音", clicked=self.toggle_recording)
        self.record_btn.setMinimumSize(120, 40)
        self.record_btn.setStyleSheet("background-color: #4CAF50; color: white; font-size: 16px;")
        control_layout.addWidget(self.record_btn)
        
        # 时间显示
        self.time_label = QLabel("就绪", alignment=Qt.AlignCenter, font=QFont("Arial", 14))
        control_layout.addWidget(self.time_label)
        layout.addWidget(control_group)
        
        # 结果组
        result_group = QGroupBox("分析结果")
        result_layout = QVBoxLayout(result_group)
        
        # 转录文本
        self.transcription_area = QTextEdit(readOnly=True)
        self.transcription_area.setPlaceholderText("录音转录结果将显示在此处")
        result_layout.addWidget(QLabel("转录文本:"))
        result_layout.addWidget(self.transcription_area)
        
        # 情感分析
        self.emotion_area = QTextEdit(readOnly=True)
        self.emotion_area.setPlaceholderText("情感分析结果将显示在此处")
        result_layout.addWidget(QLabel("情感分析:"))
        result_layout.addWidget(self.emotion_area)
        
        # 情感可视化
        self.emotion_bar = QLabel()
        self.emotion_bar.setAlignment(Qt.AlignCenter)
        self.emotion_bar.setMinimumHeight(30)
        self.emotion_bar.setStyleSheet("background-color: #CCCCCC; border-radius: 5px;")
        result_layout.addWidget(QLabel("情感可视化:"))
        result_layout.addWidget(self.emotion_bar)
        layout.addWidget(result_group)
        
        # 状态日志
        self.log_area = QTextEdit(readOnly=True)
        self.log_area.setPlaceholderText("系统日志将显示在此处")
        self.log_area.setMaximumHeight(100)
        layout.addWidget(QLabel("系统日志:"))
        layout.addWidget(self.log_area)
        
        self.statusBar().showMessage("准备就绪")
        self.log_message("应用已启动")
        if not self.emotion_available:
            self.log_message("警告: 情感分析器初始化失败，请检查模型")
    
    def log_message(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.log_area.append(f"[{timestamp}] {message}")
    
    # 添加缺失的change_model方法
    def change_model(self, model_name):
        self.recorder.model_name = model_name
        self.statusBar().showMessage(f"已切换到 {model_name} 模型")
        self.log_message(f"模型已切换到 {model_name}")
    
    def toggle_recording(self):
        if not self.recording_thread or not self.recording_thread.isRunning():
            # 开始录音
            output_file = os.path.join(self.recorder.save_folder, 
                                      f"recording_{time.strftime('%Y%m%d_%H%M%S')}.wav")
            self.recording_thread = RecordingThread(self.recorder, output_file)
            self.recording_thread.update_time.connect(self.update_time)
            self.recording_thread.recording_finished.connect(self.process_result)
            self.recording_thread.start()
            
            self.record_btn.setText("停止录音")
            self.record_btn.setStyleSheet("background-color: #FF4444; color: white;")
            self.time_label.setText("录音中: 0秒")
            self.statusBar().showMessage("正在录音...")
            self.log_message("开始录音")
        else:
            # 停止录音
            self.recording_thread.stop_recording()
            self.recording_thread.wait()  # 等待线程结束
            self.record_btn.setText("开始录音")
            self.record_btn.setStyleSheet("background-color: #4CAF50; color: white;")
            self.time_label.setText("就绪")
            self.statusBar().showMessage("录音已停止，正在处理...")
            self.log_message("录音已停止，等待处理结果")
    
    def update_time(self, seconds):
        self.time_label.setText(f"录音中: {seconds}秒")
    
    def process_result(self, result):
        try:
            transcription = result["transcription"]
            self.transcription_area.setText(transcription)
            self.log_message(f"转录完成: {transcription[:50]}...")
            
            if not self.emotion_available:
                self.emotion_area.setText("情感分析器未初始化，请检查系统日志")
                self.emotion_bar.setText("")
                self.statusBar().showMessage("情感分析不可用")
                return
                
            # 调用情感分析模型
            self.log_message("开始情感分析...")
            emotion_score = self.emotion_analyzer.predict(transcription)
            
            # 检查返回结果格式
            if not isinstance(emotion_score, (int, float)):
                self.emotion_area.setText(f"情感分析返回格式错误: {type(emotion_score).__name__}")
                self.emotion_bar.setText("")
                self.log_message(f"错误: 情感分析返回格式错误: 期望数值，得到 {type(emotion_score).__name__}")
                self.statusBar().showMessage("情感分析格式错误")
                return
                
            # 验证分数范围
            if not -1.0 <= emotion_score <= 1.0:
                self.emotion_area.setText(f"情感分数超出范围: {emotion_score}")
                self.emotion_bar.setText("")
                self.log_message(f"错误: 情感分数超出范围: {emotion_score}")
                self.statusBar().showMessage("情感分数异常")
                return
                
            # 将分数映射到情感类别
            emotion_category = self.get_emotion_category(emotion_score)
            
            # 显示情感分析结果
            emotion_text = f"情感分数: {emotion_score:.1f}\n情感类别: {emotion_category}"
            self.emotion_area.setText(emotion_text)
            
            # 更新情感可视化条
            self.update_emotion_bar(emotion_score)
            
            self.log_message(f"情感分析完成: 分数={emotion_score:.1f}, 类别={emotion_category}")
            self.statusBar().showMessage("分析完成")
        except Exception as e:
            self.emotion_area.setText(f"情感分析错误: {str(e)}")
            self.emotion_bar.setText("")
            self.log_message(f"错误: 情感分析失败: {str(e)}")
            self.statusBar().showMessage(f"分析失败: {str(e)}", 5000)
    
    def get_emotion_category(self, score):
        # 将-1.0到1.0的分数映射到情感类别
        if score >= 0.8:
            return "正"
        elif score >= 0.2:
            return "弱正"
        elif score == 0.0:
            return "中性"
        elif score <= -0.2:
            return "弱负"
        else:
            return "负"
    
    def update_emotion_bar(self, score):
        # 将分数(-1.0到1.0)映射到0-100的百分比
        percentage = int((score + 1.0) * 50)
        
        # 根据分数选择颜色 (绿色到红色)
        if score >= 0:
            # 积极情绪 - 绿色渐变
            r = 100 - int(score * 100)
            g = 255
            b = 100
        else:
            # 消极情绪 - 红色渐变
            r = 255
            g = 100 + int(score * 100)
            b = 100
        
        color = f"rgb({r}, {g}, {b})"
        
        # 创建一个简单的进度条
        bar_html = f"""
        <div style="display: flex; height: 100%; border-radius: 5px;">
            <div style="width: {percentage}%; background-color: {color}; border-radius: 5px;"></div>
            <div style="width: {100-percentage}%; background-color: #EEEEEE; border-radius: 5px;"></div>
        </div>
        """
        
        self.emotion_bar.setText(bar_html)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EmotionAnalysisApp()
    window.show()
    sys.exit(app.exec_())