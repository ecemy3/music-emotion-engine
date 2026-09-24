"""
DEAM Annotation CSV dosyalarını birleştirme scripti
İki CSV dosyasını birleştirir ve standart formata dönüştürür
"""

import pandas as pd

print("=" * 60)
print("DEAM Annotation Dosyalarını Birleştirme")
print("=" * 60)

# CSV dosyalarını oku
print("\n1. CSV dosyaları okunuyor...")
try:
    df1 = pd.read_csv("static_annotations_averaged_songs_1_2000.csv")
    print(f"   ✓ 1. dosya okundu: {len(df1)} satır")
except FileNotFoundError:
    print("   ✗ Hata: 'static_annotations_averaged_songs_1_2000.csv' bulunamadı!")
    print("   Lütfen dosyayı proje kök dizinine koyun.")
    exit(1)

try:
    df2 = pd.read_csv("static_annotations_averaged_songs_2000_2058.csv")
    print(f"   ✓ 2. dosya okundu: {len(df2)} satır")
except FileNotFoundError:
    print("   ✗ Hata: 'static_annotations_averaged_songs_2000_2058.csv' bulunamadı!")
    print("   Lütfen dosyayı proje kök dizinine koyun.")
    exit(1)

# Birleştir
print("\n2. Dosyalar birleştiriliyor...")
df = pd.concat([df1, df2])
print(f"   ✓ Toplam: {len(df)} satır")

# Kolon adlarını düzenle
print("\n3. Kolon adları düzenleniyor...")
# Önce boşlukları temizle
df.columns = [c.strip().lower() for c in df.columns]
df = df.rename(columns={"valence_mean": "valence", "arousal_mean": "arousal"})
print(f"   ✓ Kolonlar: {list(df.columns)}")

# Sadece gerekli kolonları seç
print("\n4. Gerekli kolonlar seçiliyor...")
df = df[["song_id", "valence", "arousal"]]
print(f"   ✓ Seçilen kolonlar: {list(df.columns)}")

# Kaydet
print("\n5. Dosya kaydediliyor...")
output_path = "data/deam/annotations.csv"
df.to_csv(output_path, index=False)
print(f"   ✓ Kaydedildi: {output_path}")

# Özet bilgi
print("\n" + "=" * 60)
print("Birleştirme Tamamlandı!")
print("=" * 60)
print(f"Toplam kayıt sayısı: {len(df)}")
print(f"\nİlk 5 kayıt:")
print(df.head())
print(f"\nİstatistikler:")
print(df.describe())
