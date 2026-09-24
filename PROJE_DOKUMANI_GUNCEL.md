# Bitirme Projesi - Güncel Proje Dokümanı

Hazırlama Tarihi: 09.06.2026

## 1) Proje Özeti

Bu proje, müzik parçalarının insanlarda oluşturduğu duygusal karşılığı Valence-Arousal uzayında analiz eden, derin öğrenme modelleri ve üretken yapay zeka bileşenlerini tek Streamlit arayüzünde birleştiren bir duygu analizi sistemidir.

Projenin güncel kapsamı dört ana eksenden oluşur:

1. DEAM veri seti üzerinde eğitilmiş müzik duygu tahmin modelleri
2. Anket cevaplarından insan duygu algısının Valence-Arousal skorlarına dönüştürülmesi
3. Gemini 2.5 Flash ile ses tabanlı yapay zeka duygu yorumunun alınması
4. Anket tabanlı duygu profilinden FLUX ile soyut duygu görseli oluşturulması

Sistem, bir şarkı için insan algısı, klasik derin öğrenme modeli tahmini, ensemble model tahmini, Gemini yorumu ve soyut görsel temsili aynı analitik akışta incelemeyi hedefler.

## 2) Temel Amaçlar

- Müzik parçalarının Valence ve Arousal skorlarını tahmin etmek
- İnsan anket cevaplarını sayısal duygu uzayına dönüştürmek
- Farklı model ailelerinin performansını karşılaştırmak
- İnsan, model ve Gemini sonuçlarını aynı ölçekte değerlendirmek
- Şarkıların anket verilerinden açıklanabilir bir Emotion-to-Art analizi üretmek
- Bu analizden FLUX.1-schnell ile soyut sanat görseli oluşturmak
- Geçmiş analizleri DynamoDB Local üzerinde saklayarak tekrar incelenebilir hale getirmek

## 3) Güncel Sistem Mimarisi

Sistem beş ana katmandan oluşur:

### 3.1 Veri Katmanı

- DEAM annotation dosyaları
- Ses dosyaları
- Google Forms anket CSV çıktısı
- Sabit train/validation/test split dosyası
- Ground truth ve model tahmin dosyaları

### 3.2 Model Katmanı

- CNN
- Optimized CNN
- CRNN
- Transformer
- CNN + Transformer
- Ensemble modeller

Model checkpoint dosyaları `models/` altında tutulur. Uygulama, uygun checkpointleri otomatik keşfeder ve kullanıcıya seçilebilir model listesi olarak sunar.

### 3.3 Analiz Katmanı

- Anketten Valence-Arousal hesaplama
- Demografik analiz
- Duygu değişimi analizi
- İnsan-model fark analizi
- İnsan-Gemini fark analizi
- Üçlü karşılaştırma analizi
- Emotion-to-Art analizi

### 3.4 Üretken Yapay Zeka Katmanı

- Gemini 2.5 Flash: Ses dosyası üzerinden müzik duygu analizi
- Hugging Face FLUX.1-schnell: Soyut duygu görseli üretimi

### 3.5 Uygulama ve Kayıt Katmanı

- Streamlit ana uygulama: `app.py`
- DynamoDB Local ile analiz geçmişi
- Görsel cache sistemi: `cache/images/`
- API kullanım ve hata yönetimi

## 4) Dizin Yapısı ve Dosya Rolleri

### Ana Dosyalar

- `app.py`: Streamlit arayüzü, model yükleme, karşılaştırma, Gemini ve FLUX akışları
- `PROJE_DOKUMANI.md`: Önceki kapsamlı proje dokümanı
- `PROJE_DOKUMANI_GUNCEL.md`: Güncel proje dokümanı
- `requirements.txt`: Python bağımlılıkları
- `docker-compose.yml`: DynamoDB Local çalıştırma yapılandırması

### `src/` Dosyaları

- `survey_utils.py`: Anket satırından Valence-Arousal hesaplama
- `advanced_analytics.py`: Demografik, duygu değişimi ve insan-AI analizleri
- `emotion_to_art_engine.py`: Anket verisini açıklanabilir sanat yönlendirmesine dönüştürür
- `flux_image_generator.py`: Hugging Face FLUX görsel üretimi ve PNG cache yönetimi
- `image_generator.py`: Google GenAI tabanlı görsel üretim yardımcıları
- `api_safety_guard.py`: Görsel üretimi için kullanım limiti, cache ve güvenlik yardımcıları
- `db_utils.py`: DynamoDB Local kayıt, okuma ve silme işlemleri
- `ensemble_three_models.py`: Ensemble model deneyleri ve karşılaştırmaları
- `train_deam_cnn_va.py`: Baseline CNN eğitimi
- `train_cnn_experiments.py`: CNN deneyleri
- `train_crnn_experiments.py`: CRNN, Transformer ve hibrit model deneyleri
- `merge_deam_annotations.py`: DEAM annotation birleştirme
- `convert_mp3_to_wav.py`: Ses formatı dönüştürme

### Veri ve Artefakt Klasörleri

- `data/`: DEAM ses ve annotation verileri
- `configs/`: Model deney konfigürasyonları
- `models/`: Eğitilmiş checkpoint dosyaları
- `results/`: Tahminler, metrik tabloları ve grafik çıktıları
- `logs/`: Eğitim geçmişleri ve API kullanım kayıtları
- `cache/images/`: Üretilen görsellerin prompt hash tabanlı cache dosyaları
- `config/api_limits.json`: API kullanım limitleri

## 5) Veri Kaynakları

### 5.1 DEAM Veri Seti

DEAM veri seti müzik duygu analizi için kullanılan temel eğitim veri setidir.

- Annotation satır sayısı: 1802
- Skorlar: Valence ve Arousal
- Ölçek: DEAM formatında sürekli duygu değerleri
- Birleşmiş annotation dosyası: `data/deam/annotations.csv`

Kullanılan kaynak dosyalar:

- `static_annotations_averaged_songs_1_2000.csv`
- `static_annotations_averaged_songs_2000_2058.csv`

### 5.2 Anket Verisi

Kullanılan CSV:

- `Duygu Durumu Analiz Anketi (Yanıtlar) - Form Yanıtları 1 (3).csv`

Anket verisi, katılımcıların belirli şarkıları dinledikten sonra verdikleri duygu cevaplarını içerir. Uygulama şarkı kolonunu dinamik olarak bulur ve “dinlediniz” içeren kolonlara öncelik verir. “Amaç/amac” içeren kolonlar şarkı kolonu seçimi sırasında cezalandırılır.

Öne çıkan şarkılar:

- Barber: Adagio for Strings
- Married Life - Michael Giacchino (Instrumental Version)

## 6) Ses Ön İşleme

Model tahmini için kullanılan temel akış:

1. Ses dosyası okunur
2. Mono formata dönüştürülür
3. Gerekirse yeniden örneklenir
4. Sabit süreye pad/trim uygulanır
5. Mel-Spectrogram çıkarılır
6. Amplitude-to-dB dönüşümü yapılır
7. Normalizasyon uygulanır
8. Tensor model girişine dönüştürülür

Eğitim scriptlerinde kullanılan temel parametreler:

- Sample rate: 22050 Hz
- Süre: 30 saniye
- FFT: 2048
- Hop length: 512
- Mel bin: 128

Streamlit uygulamasındaki anlık tahmin akışında:

- Sample rate: 16000 Hz
- Süre: 30 saniye
- FFT: 1024
- Hop length: 512
- Mel bin: 128

## 7) Anketten Valence-Arousal Hesabı

Anket cevapları `survey_row_to_va` fonksiyonu ile 1-9 aralığında Valence ve Arousal skorlarına dönüştürülür.

Temel yaklaşım:

1. Likert cevapları 1-5 aralığından 0-1 aralığına normalize edilir
2. Pozitif duygu göstergeleri gruplanır
3. Negatif duygu göstergeleri gruplanır
4. Yüksek ve düşük enerji göstergeleri ayrıştırılır
5. Valence pozitif-negatif farkından hesaplanır
6. Arousal enerji yoğunluğundan hesaplanır
7. Sonuçlar 1-9 Valence-Arousal ölçeğine dönüştürülür

Kullanılan temel duygu kolonları:

- Huzurlu ve sakin
- Neşeli ve enerjik
- Nostaljik
- Üzgün veya melankolik
- Hayranlık ve şaşkınlık
- Gergin veya huzursuz
- Güçlü, motive olmuş

## 8) Model Aileleri

### 8.1 CNN

Baseline CNN modeli Mel-Spectrogram girdisi üzerinde konvolüsyon blokları kullanır. Son katmanda Valence ve Arousal için iki boyutlu regresyon çıktısı üretir.

### 8.2 Optimized CNN

Daha derin CNN mimarisi kullanır. Kanal yapısı 32-64-128-256 olarak genişletilmiştir. Güncel metriklerde en iyi tekil model sonuçlarından birini verir.

### 8.3 CRNN

CNN özellik çıkarıcıdan sonra LSTM tabanlı temporal modelleme uygular. `crnn_v2`, daha güçlü CNN bloğu ve BiLSTM yapısıyla öne çıkar.

### 8.4 Transformer

Mel-Spectrogram girdisini patch embedding mantığıyla işler ve Transformer Encoder ile zamansal/uzamsal ilişkileri modellemeye çalışır.

### 8.5 CNN + Transformer

CNN ile lokal özellik çıkarımı, Transformer ile daha geniş bağlam modelleme yaklaşımını birleştirir.

### 8.6 Ensemble

Güncel projede ensemble yaklaşımı da eklenmiştir. Temel bileşenler:

- `cnn_optimized`
- `cnn_baseline`
- `crnn_v2`

Uygulamadaki manuel ağırlıklı ensemble:

- `cnn_optimized`: 0.30
- `cnn_baseline`: 0.30
- `crnn_v2`: 0.40

Ensemble sonuçları tekil modellere göre daha düşük ortalama RMSE üretmiştir.

## 9) Güncel Model Performansı

Ana karşılaştırma dosyaları:

- `results/model_comparison.csv`
- `results/ensemble_comparison.csv`
- `results/model_comparison_with_stacking.csv`

Öne çıkan sonuçlar:

| Model | RMSE V | RMSE A | Pearson V | Pearson A | Avg RMSE | Avg Pearson |
|---|---:|---:|---:|---:|---:|---:|
| cnn_optimized | 0.8448 | 0.8174 | 0.7095 | 0.7789 | 0.8311 | 0.7442 |
| cnn_baseline | 0.8404 | 0.8332 | 0.7152 | 0.7757 | 0.8368 | 0.7455 |
| crnn_v2 | 0.8500 | 0.8363 | 0.7160 | 0.7678 | 0.8431 | 0.7419 |
| ensemble_weighted | 0.8239 | 0.8033 | 0.7317 | 0.7911 | 0.8136 | 0.7614 |
| ensemble_simple_average | 0.8242 | 0.8052 | 0.7309 | 0.7926 | 0.8147 | 0.7617 |
| stacking_ensemble_cv_ridge_alpha_1.0 | 0.8112 | 0.8163 | 0.7234 | 0.7771 | 0.8137 | 0.7503 |

Güncel gözlem:

- En iyi avg RMSE tekil modellerde `cnn_optimized` iken ensemble modeller daha iyi avg RMSE üretmiştir.
- `ensemble_weighted`, avg RMSE açısından güçlü sonuç vermiştir.
- Stacking ensemble, Valence RMSE ve CCC açısından başarılıdır ancak uygulama akışında manuel ağırlıklı ensemble daha doğrudan kullanılmaktadır.

## 10) Streamlit Arayüzü

Ana uygulama `app.py` dosyasında bulunur ve `python -m streamlit run app.py` komutu ile çalıştırılır.

### Sidebar

- Model seçimi
- Hugging Face API Key girişi
- Anket CSV yükleme
- Şarkı filtresi
- Seçili model metrikleri

### Sekmeler

Güncel uygulama sekmeleri:

1. Model Karşılaştırma
2. Genel Model Performansı
3. Demografik Analiz
4. Duygu Değişimi
5. İnsan vs Gemini AI
6. İnsan vs Seçili Model vs Gemini
7. Şarkının Soyut Duygu Görselleştirmesi
8. Geçmiş Analizler

## 11) Model Karşılaştırma Akışı

Bu sekmede kullanıcı bir şarkı seçer ve isteğe bağlı olarak ses dosyası yükler.

Akış:

1. CSV içinden seçilen şarkıya ait anket satırları filtrelenir
2. İnsan ortalama Valence-Arousal skoru hesaplanır
3. Seçili model ses dosyası üzerinden tahmin üretir
4. İnsan-model farkları hesaplanır
5. Bar chart ve Valence-Arousal uzayı görselleştirilir
6. DynamoDB Local aktifse analiz geçmişe kaydedilir

## 12) Gemini Entegrasyonu

Gemini sekmeleri müzik dosyasını Gemini 2.5 Flash modeline göndererek ses tabanlı duygu analizi üretir.

Beklenen Gemini çıktısı:

- Emotion
- Valence
- Arousal
- Confidence
- Explanation

Uygulama, Gemini yanıtını JSON olarak ayrıştırır ve Valence-Arousal değerlerini 1-9 aralığında sınırlar. İnsan ortalamasıyla delta metrikleri hesaplanır ve grafiklerle gösterilir.

Not:

- Uygulamada `google.generativeai` kullanımı devam etmektedir.
- `requirements.txt` içinde hem `google-genai` hem de `google-generativeai` yer almaktadır.
- Uzun vadede Gemini tarafının tek ve güncel `google-genai` istemcisine taşınması önerilir.

## 13) Soyut Duygu Görselleştirmesi

Güncel projedeki en yeni özellik, anket verilerinden açıklanabilir sanat yönlendirmesi çıkarıp FLUX ile soyut görsel üretmesidir.

Bu özellik `Şarkının Soyut Duygu Görselleştirmesi` sekmesinde yer alır.

### 13.1 Kullanılan Dosyalar

- `src/emotion_to_art_engine.py`
- `src/flux_image_generator.py`
- `cache/images/`

### 13.2 Emotion-to-Art Engine

`emotion_to_art_engine.py`, seçilen şarkıya ait anket cevaplarını analiz ederek deterministik ve açıklanabilir bir sanat profili üretir.

Üretilen bilgiler:

- Katılımcı sayısı
- Baskın duygu dağılımı
- Ortalama Valence
- Ortalama Arousal
- Valence kategorisi
- Arousal kategorisi
- Atmosfer
- Duygu yoğunluğu
- Standart sapma
- Varyans
- Entropy
- Duygu çeşitliliği
- Renk paleti
- Kompozisyon
- Işıklandırma
- Sanat/fırça stili
- Sanatsal sembolizm
- FLUX için üretim promptu
- Profil imzası
- Akademik açıklama

### 13.3 Duygu Dağılımı

Sistem anket kolonlarından ve metin cevaplarından hedef duygu skorları çıkarır.

Hedef duygu sınıfları:

- Huzurlu
- Nostaljik
- Mutlu
- Üzgün
- Motivasyon
- Heyecan
- Romantik
- Yalnız
- Özgür
- Melankolik
- Kaygılı

Bu duygu dağılımı yüzdesel olarak hesaplanır ve en baskın ilk beş duygu arayüzde gösterilir.

### 13.4 Valence-Arousal'dan Atmosfer Seçimi

Emotion-to-Art Engine, Valence ve Arousal ortalamasını 0-1 aralığına dönüştürür.

Valence kategorileri:

- Çok Negatif
- Negatif
- Nötr
- Pozitif
- Çok Pozitif

Arousal kategorileri:

- Çok Sakin
- Sakin
- Orta Enerji
- Yüksek Enerji

Bu kategorilerden atmosfer seçilir:

- Positive + Low Arousal: Peaceful
- Positive + Mid/High Arousal: Energetic
- Negative + Low/Mid Arousal: Melancholic
- Negative + High Arousal: Chaotic
- Neutral + Low Arousal: Minimal
- Diğer durumlar: Experimental

### 13.5 Sanat Yönlendirmesi

Atmosfer ve baskın duygulara göre şu sanat parametreleri üretilir:

- Renk paleti
- Kompozisyon türü
- Işıklandırma türü
- Fırça stili
- Sembolizm

Örnek eşleştirmeler:

- Huzurlu: Soft Blue, White, Silver
- Nostaljik: Sepia, Gold, Brown
- Mutlu: Yellow, Orange, Cream
- Üzgün: Navy, Gray, Dark Blue
- Melankolik: Indigo, Fog Gray, Violet
- Kaygılı: Graphite, Deep Purple, Steel Blue

### 13.6 Görsel Boyutu / Oranı

Kullanıcı görsel üretmeden önce görsel oranını seçebilir.

Desteklenen oranlar:

- `1:1`
- `16:9`
- `9:16`

Seçilen oran promptun sonuna `Preferred aspect ratio` bilgisi olarak eklenir. Böylece aynı duygu profili farklı görsel boyut/oran tercihiyle yeniden üretilebilir.

### 13.7 FLUX Görsel Üretimi

Görsel üretimi `src/flux_image_generator.py` üzerinden Hugging Face Inference API ile yapılır.

Kullanılan model:

- `black-forest-labs/FLUX.1-schnell`

API endpoint adayları:

- `https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell`
- `https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell`

API key kaynakları:

- Sidebar `Hugging Face API Key` alanı
- `.env` içindeki `HF_API_KEY`
- Ortam değişkeni `HF_API_KEY`

Key doğrulama:

- Hugging Face key boş olamaz
- `hf_` ile başlaması beklenir

### 13.8 Cache Sistemi

FLUX üretimi maliyetli ve yavaş olabileceği için prompt bazlı cache kullanılır.

Cache mantığı:

1. Prompt normalize edilir
2. SHA256 hash hesaplanır
3. `cache/images/<prompt_hash>.png` dosyası aranır
4. Dosya varsa API çağrısı yapılmadan görsel yüklenir
5. Dosya yoksa FLUX API çağrısı yapılır
6. Dönen görsel PNG olarak cache klasörüne kaydedilir

Bu yapı aynı prompt için tekrar tekrar API maliyeti oluşmasını engeller.

### 13.9 Hata Yönetimi

FLUX entegrasyonu ayrı hata sınıfları kullanır:

- `FluxTimeoutError`
- `FluxConnectionError`
- `FluxServerError`
- `FluxRateLimitError`
- `FluxInvalidResponseError`
- `FluxImageGenerationError`

Arayüzde kullanıcıya teknik detayları sadeleştirilmiş, API key gibi hassas bilgileri temizlenmiş hata mesajları gösterilir.

## 14) API Kullanım Güvenliği

`src/api_safety_guard.py`, görsel üretiminde gereksiz API çağrılarını ve olası maliyeti azaltmak için hazırlanmıştır.

Varsayılan limitler:

- API aktif: true
- Günlük maksimum görsel isteği: 20
- İstek başına retry limiti: 2
- Session başına istek limiti: 10

Kullanılan dosyalar:

- `config/api_limits.json`
- `logs/api_usage.json`
- `cache/images/`

Not:

- Güncel FLUX sekmesinde temel prompt cache doğrudan `flux_image_generator.py` üzerinden çalışmaktadır.
- `api_safety_guard.py` daha geniş kullanım limiti ve metadata cache altyapısı için hazır tutulmaktadır.

## 15) Geçmiş Analizler ve DynamoDB Local

Projede analiz geçmişlerini saklamak için DynamoDB Local desteği eklenmiştir.

Kaydedilebilen analiz türleri:

- Model karşılaştırma analizleri
- Gemini analizleri

Kullanım:

```bash
docker-compose up -d
```

Streamlit içindeki `Geçmiş Analizler` sekmesi DynamoDB Local çalışıyorsa kayıtları listeler ve silme işlemlerine izin verir.

## 16) Bağımlılıklar

Güncel `requirements.txt` içeriğinde öne çıkan paketler:

- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- google-genai
- google-generativeai
- Pillow
- tensorflow
- keras
- joblib
- torch
- torchaudio
- soundfile
- streamlit
- boto3
- requests

Eklenen/güncellenen önemli bağımlılıklar:

- `requests`: Hugging Face FLUX API çağrısı için
- `boto3`: DynamoDB Local entegrasyonu için
- `google-genai` ve `google-generativeai`: Gemini ve görsel üretim yardımcıları için

## 17) Çalıştırma Talimatları

### 17.1 Sanal Ortam

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 17.2 Streamlit Uygulaması

```bash
python -m streamlit run app.py
```

Varsayılan adres:

```text
http://localhost:8501
```

### 17.3 DynamoDB Local

```bash
docker-compose up -d
```

### 17.4 Hugging Face API Key

Görsel üretimi için `.env` dosyasına şu değer eklenebilir:

```text
HF_API_KEY=hf_xxxxxxxxxxxxxxxxx
```

Alternatif olarak key Streamlit sidebar içinden girilebilir.

### 17.5 Gemini API Key

Gemini analiz sekmelerinde API key arayüzden girilir. Uygulama yeni Gemini key formatı için doğrulama yapar.

## 18) Eğitim Komutları

Baseline CNN:

```bash
python src/train_deam_cnn_va.py
```

CNN deneyleri:

```bash
python src/train_cnn_experiments.py --preset cnn_optimized --experiment-name cnn_optimized
```

CRNN ve Transformer deneyleri:

```bash
python src/train_crnn_experiments.py --preset crnn_v2 --experiment-name crnn_v2
python src/train_crnn_experiments.py --preset transformer_v2 --experiment-name transformer_v2
```

En iyi model seçimi:

```bash
python src/train_cnn_experiments.py --select-best-cnn
python src/train_crnn_experiments.py --select-best-crnn
```

## 19) Güncel Katkılar ve Yenilikler

Bu güncel sürümde önceki dokümana göre öne çıkan eklemeler:

- Hugging Face FLUX.1-schnell ile soyut görsel üretimi
- Görsel oranı seçimi: `1:1`, `16:9`, `9:16`
- Deterministik Emotion-to-Art Engine
- Anket verisinden renk, atmosfer, kompozisyon ve sembolizm çıkarımı
- Prompt hash tabanlı PNG cache sistemi
- Hugging Face API key alanı
- FLUX özel hata sınıfları
- DynamoDB Local geçmiş analiz sekmesi
- Ensemble model yaklaşımı
- Ensemble ve stacking karşılaştırma metrikleri

## 20) Bilinen Durumlar ve İyileştirme Önerileri

1. Gemini istemci birliği

`app.py` içinde `google.generativeai`, bazı yardımcı dosyalarda `google-genai` kullanımı vardır. Uzun vadede tek istemciye geçilmesi bakım kolaylığı sağlar.

2. FLUX oran parametresi

Güncel akışta görsel oranı prompt metnine eklenmektedir. Hugging Face endpointi desteklediği ölçüde ileride doğrudan API parametresi olarak da gönderilebilir.

3. API safety guard entegrasyonu

`api_safety_guard.py` hazır durumdadır. FLUX sekmesindeki üretim butonu günlük/session limit kontrolüyle daha sıkı entegre edilebilir.

4. Encoding temizliği

Bazı eski dosyalarda Türkçe karakterler mojibake görünebilmektedir. Sunum veya tez teslimi öncesinde tüm `.py` ve `.md` dosyalarında UTF-8 karakter kontrolü yapılması önerilir.

5. Ensemble seçimi

Uygulamada manuel ağırlıklı ensemble kullanılmaktadır. Stacking ensemble sonuçları da güçlüdür; ancak canlı tahmin akışına alınmadan önce veri sızıntısı, çapraz doğrulama ve genellenebilirlik açısından dikkatle değerlendirilmelidir.

## 21) Sonuç

Projenin güncel hali, klasik müzik duygu tahmini ile üretken yapay zekayı birleştiren kapsamlı bir analiz platformudur. Sistem yalnızca bir şarkının Valence-Arousal skorunu tahmin etmekle kalmaz; insan algısını, derin öğrenme modellerini, Gemini yorumunu ve FLUX ile üretilen soyut duygu görselini aynı uygulamada bir araya getirir.

Bu yönüyle proje, müzik duygu analizi alanında hem sayısal karşılaştırma hem de yaratıcı görselleştirme sunan bütünleşik bir bitirme projesi seviyesine ulaşmıştır.
