import contextlib
import os
import subprocess
import time
import whisper
from transformers import pipeline
import logging
from transformers import logging as hf_logging
import warnings
warnings.filterwarnings("ignore")


def format_timestamp(sec: float) -> str:
    """Chuyển đổi số giây thành định dạng SRT: HH:MM:SS,mmm"""
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int((sec - int(sec)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def translate_text(text: str, src_lang: str, tgt_lang: str = "vi"):
    """
    model ko thể dịch chích xác đc các ngôn ngữ khác ngoài tiếng anh sang thẳng tiếng việt
    dòng 47 là đường dẫn file video 
    dòng 53 là ddg dẫn tyệt đối của ffmpeg (phải sửa lại đg dẫn này)
    """
    # nếu là tiếng việt thì bỏ quá
    if src_lang == "vi":
        return text 
    # dịch tiếng anh sang tiensg việt
    elif src_lang == "en":
        translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-vi")
        return translator(text)[0]["translation_text"]
    # dịch các tiếng khác sang tiếng anh rồi sang việt
    else:
        try:
            translator_src_en = pipeline("translation", model=f"Helsinki-NLP/opus-mt-{src_lang}-en")
            translated_en = translator_src_en(text)[0]["translation_text"]
        except Exception as e:
            return f"[Lỗi: Không thể dịch từ {src_lang} sang tiếng Anh. Chi tiết: {e}]"
        
        # Dịch từ tiếng Anh sang tiếng Việt
        translator_en_vi = pipeline("translation", model="Helsinki-NLP/opus-mt-en-vi")
        translated_vi = translator_en_vi(translated_en)[0]["translation_text"]
        return translated_vi

def main():
    start_time = time.time()
    print("🔄 Đang xử lý video…")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Đường dẫn tệp
    video_goc = os.path.join("video", "Download.mp4")
    file_phu_de = "temp_subtitle.srt"
    video_da_xu_ly = os.path.join("video_co_vietsub", "video_vietsub.mp4")
    # đường dẫn tuyệt đối model của ffmpeg
    Dg_dan_ffmpeg = r"D:\VS Code\Vietsub\ffmpeg-2025-05-05-git-f4e72eb5a3-full_build\bin\ffmpeg.exe"

    # Tải mô hình Whisper
    print("📝 Đang tải mô hình Whisper…")
    whisper_model = whisper.load_model("medium")  # Dùng mô hình medium để nhận diện tốt hơn (có 4 model khác nhau)
    hf_logging.set_verbosity_error()
    logging.getLogger("transformers").setLevel(logging.ERROR)
    
    # Phiên âm
    print("🎤 Đang phiên âm âm thanh…")
    transcription = whisper_model.transcribe(video_goc)
    detected_lang = transcription["language"]
    print(f"🌐 Ngôn ngữ nhận diện: {detected_lang}")

    segments = transcription["segments"]
    # xử lý lần 2 sau khi chỉ còn tiếng anh hoặc tiếng
    # Dịch văn bản nếu không phải tiếng Việt
    if detected_lang != "vi":
        print("🌏 Đang dịch các đoạn văn bản…")
        for segment in segments:
            try:
                segment["text"] = translate_text(segment["text"], detected_lang)
            except Exception as e:
                print(f"❌ Lỗi khi dịch: {segment['text']}. Chi tiết lỗi: {e}")
                segment["text"] = "[Lỗi dịch thuật]"
    else:
        print("✅ Phát hiện tiếng Việt, không cần dịch.")

    # Ghi file SRT
    print(f"📝 Đang ghi phụ đề vào {file_phu_de}…")
    with open(file_phu_de, "w", encoding="utf-8") as f:
        for idx, segment in enumerate(segments, start=1):
            f.write(f"{idx}\n")
            f.write(f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n")
            f.write(f"{segment['text']}\n\n")

    # Tạo thư mục đầu ra
    os.makedirs(os.path.dirname(video_da_xu_ly), exist_ok=True)

    # Thêm phụ đề vào video bằng FFmpeg
    ffmpeg_cmd = [
        Dg_dan_ffmpeg,
        "-i", video_goc,
        "-vf", f"subtitles={file_phu_de}",
        "-c:a", "copy",
        "-y",
        video_da_xu_ly
    ]
    print("🚀 Đang chạy FFmpeg:", " ".join(ffmpeg_cmd))
    subprocess.run(ffmpeg_cmd, check=True)

    # Xóa file SRT tạm thời
    if os.path.isfile(video_da_xu_ly):
        os.remove(file_phu_de)
        print("🗑️ Đã xóa file phụ đề tạm thời.")
    else:
        print("⚠️ Giữ lại file SRT để kiểm tra.")

    # Thời gian thực thi
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"🎉 Hoàn thành trong {int(minutes)} phút {int(seconds)} giây")
    print(f"📁 Video đầu ra: {video_da_xu_ly}")

if __name__ == "__main__":
    main()