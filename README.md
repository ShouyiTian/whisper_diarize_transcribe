# Whisper 说话人分离转写工具

这是一个基于 Whisper 和 pyannote.audio 的音频转写和说话人分离工具，可以将视频文件转换为带说话人标识的文字记录。

## 功能特点

-  **视频转音频**：自动从视频文件提取音频
-  **说话人分离**：识别不同的说话人并标记时间段
-  **智能转写**：使用 Whisper 进行高精度语音转文字
-  **多格式输出**：生成文本文件和 SRT 字幕文件
-  **统计信息**：提供详细的处理统计报告
-  **灵活配置**：支持配置文件自定义参数
-  **进度显示**：实时显示处理进度
-  **错误处理**：完善的错误处理和日志记录

## 安装依赖

### 1. 安装 Python 依赖
```bash
pip install -r requirements.txt
```

### 2. 安装 FFmpeg
- **Windows**: 下载并安装 [FFmpeg](https://ffmpeg.org/download.html)
- **macOS**: `brew install ffmpeg`
- **Ubuntu/Debian**: `sudo apt install ffmpeg`

### 3. 获取 HuggingFace Token
1. 访问 [HuggingFace](https://huggingface.co/)
2. 注册并登录账户
3. 访问 [pyannote/speaker-diarization](https://huggingface.co/pyannote/speaker-diarization)
4. 申请访问权限
5. 在个人设置中生成 Access Token
6. 将 Token 添加到 `config.json` 文件中

## 使用方法

### 1. 准备文件
将视频文件重命名为 `meeting.mp4` 或修改 `config.json` 中的 `video_file` 参数。

### 2. 配置参数
编辑 `config.json` 文件，根据需要调整参数：

```json
{
  "video_file": "meeting.mp4",
  "audio_file": "meeting.wav",
  "language": "ja",
  "huggingface_token": "your_token_here",
  "whisper_model_size": "large"
}
```

### 3. 运行程序
```bash
python whisper_diarize_transcribe.py
```

## 配置参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `video_file` | 输入视频文件路径 | `meeting.mp4` |
| `audio_file` | 输出音频文件路径 | `meeting.wav` |
| `chunks_dir` | 音频片段目录 | `chunks` |
| `text_output` | 文本输出文件 | `transcription.txt` |
| `srt_output` | SRT字幕文件 | `transcription.srt` |
| `language` | 转写语言代码 | `ja` |
| `whisper_model_size` | Whisper模型大小 | `large` |
| `sample_rate` | 音频采样率 | `16000` |
| `channels` | 音频声道数 | `1` |
| `min_segment_duration` | 最小音频段时长(秒) | `0.5` |

## 输出文件

### 1. 文本文件 (`transcription.txt`)
```
[0:00:00 --> 0:00:05] SPEAKER_00: こんにちは、今日の会議を始めましょう。
[0:00:05 --> 0:00:10] SPEAKER_01: はい、よろしくお願いします。
```

### 2. SRT字幕文件 (`transcription.srt`)
```
1
00:00:00,000 --> 00:00:05,000
SPEAKER_00: こんにちは、今日の会議を始めましょう。

2
00:00:05,000 --> 00:00:10,000
SPEAKER_01: はい、よろしくお願いします。
```

### 3. 统计文件 (`transcription_stats.json`)
```json
{
  "total_segments": 25,
  "speakers": ["SPEAKER_00", "SPEAKER_01"],
  "total_duration": 1800.5,
  "timestamp": "2024-01-15T10:30:00"
}
```

## 性能优化

### GPU 加速
- 程序会自动检测 GPU 并启用加速
- 确保安装了 CUDA 版本的 PyTorch

### 内存优化
- 自动过滤过短的音频段
- 分批处理大文件
- 及时释放内存

### 模型选择
- `tiny`: 最快，精度较低
- `base`: 平衡速度和精度
- `small`: 较好的精度
- `medium`: 高精度
- `large`: 最高精度（推荐）

## 故障排除

### 常见问题

1. **FFmpeg 未找到**
   - 确保 FFmpeg 已正确安装并添加到系统 PATH

2. **HuggingFace Token 错误**
   - 检查 Token 是否有效
   - 确认已获得 pyannote/speaker-diarization 的访问权限

3. **内存不足**
   - 使用较小的 Whisper 模型
   - 减少 `min_segment_duration` 值

4. **GPU 内存不足**
   - 使用 CPU 模式：设置 `use_gpu: false`
   - 或使用较小的模型

### 日志文件
程序运行时会生成 `whisper_diarize.log` 文件，包含详细的处理日志。

## 技术架构

```
视频文件 → 音频提取 → 说话人分离 → 音频分割 → Whisper转写 → 格式化输出
    ↓           ↓           ↓           ↓           ↓           ↓
  FFmpeg    pyannote    pydub      Whisper    文本/SRT    统计报告
```

