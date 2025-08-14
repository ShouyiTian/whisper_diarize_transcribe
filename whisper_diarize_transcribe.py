import os
import subprocess
import datetime
import torch
import logging
import json
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
from pydub import AudioSegment
from pyannote.audio import Pipeline
import whisper
from tqdm import tqdm

# ==== 配置类 ====
@dataclass
class Config:
    """配置类，集中管理所有配置项"""
    video_file: str = "meeting.mp4"
    audio_file: str = "meeting.wav"
    chunks_dir: str = "chunks"
    text_output: str = "transcription.txt"
    srt_output: str = "transcription.srt"
    language: str = "ja"
    huggingface_token: str = "<huggingfaceid>"
    whisper_model_size: str = "large"
    use_gpu: bool = torch.cuda.is_available()
    sample_rate: int = 16000
    channels: int = 1
    min_segment_duration: float = 0.5
    log_level: str = "INFO"
    log_file: str = "whisper_diarize.log"
    
    def __post_init__(self):
        """初始化后验证配置"""
        if not self.huggingface_token or self.huggingface_token == "YOUR_TOKEN_HERE":
            raise ValueError("请设置有效的 HuggingFace token")
    
    @classmethod
    def from_file(cls, config_file: str = "config.json") -> "Config":
        """从配置文件加载配置"""
        if os.path.exists(config_file):
            with open(config_file, "r", encoding="utf-8") as f:
                config_data = json.load(f)
            return cls(**config_data)
        return cls()

# ==== 日志配置 ====
def setup_logging(config: Config):
    """设置日志配置"""
    log_level = getattr(logging, config.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# ==== 错误处理装饰器 ====
def handle_errors(func):
    """错误处理装饰器"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"执行 {func.__name__} 时发生错误: {str(e)}")
            raise
    return wrapper

# ==== 音频处理类 ====
class AudioProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    @handle_errors
    def extract_audio(self) -> bool:
        """提取音频文件"""
        if os.path.exists(self.config.audio_file):
            self.logger.info(f"音频文件已存在: {self.config.audio_file}")
            return True
            
        if not os.path.exists(self.config.video_file):
            raise FileNotFoundError(f"视频文件不存在: {self.config.video_file}")
            
        self.logger.info(f"🎼 正在提取音频: {self.config.audio_file}")
        
        cmd = [
            "ffmpeg", "-y", "-i", self.config.video_file,
            "-ar", str(self.config.sample_rate),
            "-ac", str(self.config.channels),
            self.config.audio_file
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg 执行失败: {result.stderr}")
            
        self.logger.info("✅ 音频提取完成")
        return True

# ==== 说话人分离类 ====
class SpeakerDiarizer:
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.pipeline = None
        
    @handle_errors
    def load_model(self):
        """加载说话人分离模型"""
        self.logger.info("👥 正在加载 pyannote 模型...")
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization", 
            use_auth_token=self.config.huggingface_token
        )
        
        if self.config.use_gpu:
            self.pipeline.to(torch.device("cuda"))
            self.logger.info("使用 GPU 加速")
        else:
            self.logger.info("使用 CPU 处理")
            
    @handle_errors
    def diarize(self) -> List[Tuple]:
        """执行说话人分离"""
        if not self.pipeline:
            self.load_model()
            
        self.logger.info("正在进行说话人分离...")
        diarization = self.pipeline(self.config.audio_file)
        segments = list(diarization.itertracks(yield_label=True))
        
        num_speakers = len(set(s[2] for s in segments))
        self.logger.info(f"✅ 识别到 {num_speakers} 位说话人，共 {len(segments)} 段")
        
        return segments

# ==== 音频分割类 ====
class AudioSegmenter:
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    @handle_errors
    def segment_audio(self, segments: List[Tuple]) -> List[Tuple]:
        """分割音频段落"""
        self.logger.info("✂️ 正在分割音频段落...")
        
        audio = AudioSegment.from_wav(self.config.audio_file)
        os.makedirs(self.config.chunks_dir, exist_ok=True)
        chunk_files = []
        
        for i, (segment, _, speaker) in enumerate(tqdm(segments, desc="分割音频")):
            start_ms = int(segment.start * 1000)
            end_ms = int(segment.end * 1000)
            chunk = audio[start_ms:end_ms]
            
            # 过滤过短的音频段
            if len(chunk) < self.config.min_segment_duration * 1000:
                continue
                
            filename = f"{self.config.chunks_dir}/{i:03d}_{speaker}.wav"
            chunk.export(filename, format="wav")
            chunk_files.append((i, segment, speaker, filename))
            
        self.logger.info(f"✅ 分割完成，共 {len(chunk_files)} 个有效音频段")
        return chunk_files

# ==== Whisper 转写类 ====
class WhisperTranscriber:
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        
    @handle_errors
    def load_model(self):
        """加载 Whisper 模型"""
        self.logger.info(f"🧠 正在加载 Whisper {self.config.whisper_model_size} 模型...")
        device = "cuda" if self.config.use_gpu else "cpu"
        self.model = whisper.load_model(self.config.whisper_model_size, device=device)
        
    @handle_errors
    def transcribe_chunks(self, chunk_files: List[Tuple]) -> List[Tuple]:
        """转写音频段落"""
        if not self.model:
            self.load_model()
            
        self.logger.info("📝 正在转写音频段落...")
        results = []
        
        for i, segment, speaker, chunk_file in tqdm(chunk_files, desc="转写音频"):
            try:
                result = self.model.transcribe(
                    chunk_file, 
                    language=self.config.language,
                    task="transcribe"
                )
                text = result["text"].strip()
                
                # 过滤空文本
                if text:
                    results.append((i, segment, speaker, text))
                    
            except Exception as e:
                self.logger.warning(f"转写第 {i+1} 段失败: {str(e)}")
                continue
                
        self.logger.info(f"✅ 转写完成，共 {len(results)} 段有效文本")
        return results

# ==== 输出格式化类 ====
class OutputFormatter:
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    @handle_errors
    def format_time(self, seconds: float) -> str:
        """格式化时间"""
        return str(datetime.timedelta(seconds=seconds)).split(".")[0]
        
    @handle_errors
    def format_srt_time(self, seconds: float) -> str:
        """格式化 SRT 时间格式"""
        td = datetime.timedelta(seconds=seconds)
        return str(td)[:11].replace(".", ",").rjust(12, "0")
        
    @handle_errors
    def save_outputs(self, results: List[Tuple]):
        """保存输出文件"""
        self.logger.info("📤 正在保存输出文件...")
        
        # 保存文本文件
        with open(self.config.text_output, "w", encoding="utf-8") as txt_file:
            for i, segment, speaker, text in results:
                start = self.format_time(segment.start)
                end = self.format_time(segment.end)
                txt_file.write(f"[{start} --> {end}] {speaker}: {text}\n")
                
        # 保存 SRT 文件
        with open(self.config.srt_output, "w", encoding="utf-8") as srt_file:
            for idx, (i, segment, speaker, text) in enumerate(results):
                srt_file.write(f"{idx+1}\n")
                srt_file.write(f"{self.format_srt_time(segment.start)} --> {self.format_srt_time(segment.end)}\n")
                srt_file.write(f"{speaker}: {text}\n\n")
                
        # 保存统计信息
        stats = {
            "total_segments": len(results),
            "speakers": list(set(r[2] for r in results)),
            "total_duration": sum(r[1].end - r[1].start for r in results),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        with open("transcription_stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
            
        self.logger.info(f"✅ 输出完成:\n- 文本文件: {self.config.text_output}\n- 字幕文件: {self.config.srt_output}\n- 统计文件: transcription_stats.json")

# ==== 主处理类 ====
class WhisperDiarizationProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.audio_processor = AudioProcessor(config)
        self.diarizer = SpeakerDiarizer(config)
        self.segmenter = AudioSegmenter(config)
        self.transcriber = WhisperTranscriber(config)
        self.formatter = OutputFormatter(config)
        
    @handle_errors
    def process(self):
        """执行完整的处理流程"""
        self.logger.info("🚀 开始处理音频转写和说话人分离...")
        
        # Step 1: 提取音频
        self.audio_processor.extract_audio()
        
        # Step 2: 说话人分离
        segments = self.diarizer.diarize()
        
        # Step 3: 分割音频
        chunk_files = self.segmenter.segment_audio(segments)
        
        # Step 4: 转写音频
        results = self.transcriber.transcribe_chunks(chunk_files)
        
        # Step 5: 保存输出
        self.formatter.save_outputs(results)
        
        self.logger.info("🎉 处理完成！")

# ==== 主函数 ====
def main():
    """主函数"""
    try:
        # 从配置文件加载配置
        config = Config.from_file()
        
        # 设置日志
        logger = setup_logging(config)
        
        # 创建处理器并执行
        processor = WhisperDiarizationProcessor(config)
        processor.process()
        
    except Exception as e:
        print(f"处理失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()
