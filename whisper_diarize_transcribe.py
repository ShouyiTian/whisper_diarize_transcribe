import os
import subprocess
import datetime
import torch
from pydub import AudioSegment
from pyannote.audio import Pipeline
import whisper

# ==== 配置项 ====
VIDEO_FILE = "meeting.mp4"
AUDIO_FILE = "meeting.wav"
CHUNKS_DIR = "chunks"
TEXT_OUTPUT = "transcription.txt"
SRT_OUTPUT = "transcription.srt"
LANGUAGE = "ja"
HUGGINGFACE_TOKEN = "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"  # <<< 替换为你的 HuggingFace token
USE_GPU = torch.cuda.is_available()

# ==== Step 1: 提取音频 ====
if not os.path.exists(AUDIO_FILE):
    print(f"🎼 Step 1: 提取音频 {AUDIO_FILE} ...")
    subprocess.run(["ffmpeg", "-y", "-i", VIDEO_FILE, "-ar", "16000", "-ac", "1", AUDIO_FILE])
    print("✅ 音频提取完成")

# ==== Step 2: 说话人分离 ====
print("👥 Step 2: 加载 pyannote 模型并进行说话人分离 ...")
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HUGGINGFACE_TOKEN)
if USE_GPU:
    pipeline.to(torch.device("cuda"))

diarization = pipeline(AUDIO_FILE)
segments = list(diarization.itertracks(yield_label=True))
num_speakers = len(set(s[2] for s in segments))
print(f"✅ 识别到 {num_speakers} 位说话人，共 {len(segments)} 段")

# ==== Step 3: 分割音频段 ====
print("✂️ Step 3: 分割音频段落 ...")
audio = AudioSegment.from_wav(AUDIO_FILE)
os.makedirs(CHUNKS_DIR, exist_ok=True)
chunk_files = []

for i, (segment, _, speaker) in enumerate(segments):
    start_ms = int(segment.start * 1000)
    end_ms = int(segment.end * 1000)
    chunk = audio[start_ms:end_ms]
    filename = f"{CHUNKS_DIR}/{i:03d}_{speaker}.wav"
    chunk.export(filename, format="wav")
    chunk_files.append((i, segment, speaker, filename))
print("✅ 分割完成")

# ==== Step 4: Whisper 转写 ====
print("🧠 Step 4: 加载 Whisper 模型 ...")
whisper_model = whisper.load_model("large", device="cuda" if USE_GPU else "cpu")

results = []
print("📝 Step 5: 转写每段音频 ...")
for i, segment, speaker, chunk_file in chunk_files:
    print(f"  ⏳ 正在转写第 {i+1}/{len(chunk_files)} 段：{chunk_file}")
    result = whisper_model.transcribe(chunk_file, language=LANGUAGE)
    results.append((i, segment, speaker, result["text"].strip()))

# ==== Step 6: 输出文本与 SRT ====
print("📤 Step 6: 输出转写文本与字幕 ...")

with open(TEXT_OUTPUT, "w", encoding="utf-8") as txt_file, open(SRT_OUTPUT, "w", encoding="utf-8") as srt_file:
    for idx, (i, segment, speaker, text) in enumerate(results):
        start = str(datetime.timedelta(seconds=segment.start)).split(".")[0]
        end = str(datetime.timedelta(seconds=segment.end)).split(".")[0]

        # 文本文件输出
        txt_file.write(f"[{start} --> {end}] {speaker}: {text}\n")

        # SRT 文件输出
        def format_srt_time(t):
            td = datetime.timedelta(seconds=t)
            return str(td)[:11].replace(".", ",").rjust(12, "0")

        srt_file.write(f"{idx+1}\n")
        srt_file.write(f"{format_srt_time(segment.start)} --> {format_srt_time(segment.end)}\n")
        srt_file.write(f"{speaker}: {text}\n\n")

print(f"\n✅ 所有内容保存完毕：\n- 文本文件: {TEXT_OUTPUT}\n- 字幕文件: {SRT_OUTPUT}")
