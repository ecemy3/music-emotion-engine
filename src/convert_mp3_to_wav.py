"""
DEAM MP3 dosyalarını WAV formatına dönüştürme scripti
"""

import os
import librosa
import soundfile as sf

# Kaynak ve hedef klasörler
SRC = "data/deam/audio_mp3"
DST = "data/deam/audio"

# Hedef klasörü oluştur
os.makedirs(DST, exist_ok=True)

print(f"Kaynak klasör: {SRC}")
print(f"Hedef klasör: {DST}")
print("-" * 50)

# MP3 dosyalarını WAV'a dönüştür
mp3_files = [f for f in os.listdir(SRC) if f.endswith(".mp3")]

if not mp3_files:
    print("⚠️  Kaynak klasörde MP3 dosyası bulunamadı!")
else:
    print(f"{len(mp3_files)} adet MP3 dosyası bulundu.\n")
    
    success_count = 0
    fail_count = 0
    
    for i, f in enumerate(mp3_files, 1):
        try:
            # MP3 dosyasını yükle
            audio, sr = librosa.load(os.path.join(SRC, f), sr=None, mono=False)
            
            # WAV olarak kaydet
            output_path = os.path.join(DST, f.replace(".mp3", ".wav"))
            sf.write(output_path, audio.T if audio.ndim > 1 else audio, sr)
            
            print(f"[{i}/{len(mp3_files)}] ✓ {f} → {f.replace('.mp3', '.wav')}")
            success_count += 1
        
        except Exception as e:
            print(f"[{i}/{len(mp3_files)}] ✗ {f} - Hata: {e}")
            fail_count += 1

print("-" * 50)
print(f"Dönüştürme tamamlandı!")
print(f"Başarılı: {success_count} | Başarısız: {fail_count}")
