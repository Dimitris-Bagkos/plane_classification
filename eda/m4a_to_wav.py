from pydub import AudioSegment
from pydub.utils import which
import os

AudioSegment.converter = which("ffmpeg") or r"C:\Users\Dimitris\Downloads\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"
AudioSegment.ffprobe   = which("ffprobe") or r"C:\Users\Dimitris\Downloads\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin\ffprobe.exe"


input_dir = "../data/Data 18 June 2025/Data 18 June 2025"
output_dir = "../data/converted_wav"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith(".m4a"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.replace(".m4a", ".wav"))
        try:
            audio = AudioSegment.from_file(input_path, format="m4a")
            audio.export(output_path, format="wav")
            print(f"✅ Converted {filename}")
        except Exception as e:
            print(f"❌ Failed to convert {filename}: {e}")

