# Bitirme Projesi - Kapsamli Proje Dokumani

Hazirlama Tarihi: 24.04.2026

## 1) Proje Ozeti

Bu proje, muzikten duygu analizi (Valence-Arousal) yapan bir sistemdir. Iki ana ekseni vardir:

1. DEAM veri seti ile egitilmis derin ogrenme modelleri (CNN, CRNN, Transformer, CNN+Transformer)
2. Insan anket cevaplari ile model ve Gemini AI sonuclarini karsilastiran Streamlit arayuzu

Temel hedefler:

- Muzik parcalarinin Valence (pozitiflik/negatiflik) ve Arousal (enerji) skorlarini tahmin etmek
- Insan algisi ile model tahminlerini karsilastirmak
- Gemini 2.5 Flash ile yapilan muzik yorumunu insan ve CNN sonuclariyla birlikte degerlendirmek

## 2) Proje Mimarisi

Sistem 3 katmandan olusur:

1. Veri ve egitim katmani
- DEAM annotation birlestirme
- MP3 -> WAV donusumu
- Sabit train/val/test split
- Deney tabanli egitim scriptleri

2. Model/artefakt katmani
- Egitilmis agirliklar: models klasoru
- Metrikler: results/model_comparison.csv
- Egitim gecmisi: logs klasoru

3. Uygulama katmani
- Streamlit arayuzu: app.py
- Anket analizi: src/advanced_analytics.py
- Anketten VA turetme: src/survey_utils.py
- Gemini entegrasyonu

## 3) Dizin Yapisi ve Dosya Rolleri

### Klasorler

- app.py: Streamlit ana uygulama
- src/: tum egitim ve analiz scriptleri
- configs/: deney konfig dosyalari
- models/: best/last checkpoint dosyalari
- logs/: epoch bazli egitim gecmisleri + deney logu
- results/: tahmin dosyalari, karsilastirma tablosu, grafik artefaktlari
- data/deam/: audio ve annotation verisi

### Onemli kaynak dosyalar

- src/train_deam_cnn_va.py: Baseline CNN egitimi (Asama 1)
- src/train_cnn_experiments.py: CNN varyasyon deneyleri (Asama 2)
- src/train_crnn_experiments.py: CRNN/Transformer/CNN-Transformer deneyleri (Asama 3)
- src/merge_deam_annotations.py: annotation birlestirme
- src/convert_mp3_to_wav.py: ses donusumu
- src/advanced_analytics.py: demografik, degisim, insan-vs-AI analizleri
- src/survey_utils.py: survey_row_to_va fonksiyonu
- test_gemini_models.py: google.genai ile model listeleme testi

## 4) Veri Katmani

### 4.1 DEAM veri ozetleri

- Annotation satir sayisi: 1802
- Annotation kolonlari: song_id, valence, arousal
- Song ID araligi: 2 - 2058

Kaynak dosyalar:

- static_annotations_averaged_songs_1_2000.csv
- static_annotations_averaged_songs_2000_2058.csv
- data/deam/annotations.csv (birlesmis cikti)

### 4.2 Anket CSV ozetleri

Kullanilan dosya:

- Duygu Durumu Analiz Anketi (Yanıtlar) - Form Yanıtları 1 (3).csv

Profil:

- Toplam satir: 298
- Toplam kolon: 28
- Sarki kolonu: Hangi müziği dinlediniz
- Muzik dinleme amaci kolonu: Genellikle hangi amaçla müzik dinlersiniz?
- Sarki kolonu dolu satir: 281
- Benzersiz sarki sayisi: 2

Sarki dagilimi:

- Barber: Adagio for Strings -> 242
- Married Life - Micheal Giaccihino (Enstrumental Version) -> 39

Not: Uygulama, sarki kolonu seciminde "muzik dinleme amaci" kolonunu karistirmayacak sekilde guncellenmistir.

## 5) Ozellik Cikarma ve On Isleme

Ses isleme sabitleri:

- Sample rate: 22050
- Sure: 30 saniye
- FFT: 2048
- Hop length: 512
- Mel bin: 128

Akis:

1. WAV/MP3 yuklenir veya dosyadan okunur
2. Mono hale getirilir
3. Gerekirse 22050 Hz'e resample edilir
4. 30 saniyeye pad/trim yapilir
5. MelSpectrogram + AmplitudeToDB
6. Z-score benzeri normalizasyon: (x - mean)/(std + 1e-6)

Augmentation secenekleri:

- none
- specaugment
- specaugment_noise (specaugment + hafif gürültü)

## 6) Anketten Valence-Arousal Hesabi

survey_row_to_va akisinda:

1. Likert 1-5 cevaplari 0-1 araligina normalize edilir
2. Pozitif/negatif duygu gruplari olusturulur
3. Yuksek/dusuk enerji gruplari olusturulur
4. Valence ve Arousal fark temelli hesaplanir
5. Sonuc 1-9 araligina map edilir

Kullanilan duygusal kolonlar:

- huzurlu ve sakin
- neseli ve enerjik
- nostaljik
- uzgun veya melankolik
- hayranlik ve saskinlik
- gergin veya huzursuz
- guclu, motive olmus

## 7) Model Aileleri ve Yukleme Mimarisi

Model aileleri:

1. CNN
- Baseline: 32-64-128
- Optimized: 32-64-128-256
- AdaptiveAvgPool + MLP head

2. CRNN
- CNN ozellik cikarici + LSTM head
- bidirectional ve katman sayisi checkpointten okunur

3. CRNN Legacy
- Eski checkpoint uyumlulugu icin fallback

4. Transformer
- Patch embedding + Transformer Encoder

5. CNN+Transformer
- CNN ile ozellik cikarimi + Transformer temporal modelleme

Dinamik model yukleme:

- Checkpoint anahtarlarindan aile tespit edilir
- Uygun model sinifi olusturulur
- State dict yuklenir

Auto-discovery:

- models/*_best.pt ve models/*_last.pt dosyalari otomatik listelenir
- Arayuz model secimi bu listeye gore olusur

## 8) Arayuz (Streamlit) Tam Detay

### 8.1 Sidebar

- Model Sec
- Anket CSV Dosyasi yukleme
- Analiz Sarki Filtresi
  - Tum sarkilar
  - CSV'deki sarki adlari
- Secili model metrikleri (varsa): RMSE, Pearson, MAE

### 8.2 CSV ve sarki kolon yonetimi

- Sarki kolonu dinamik bulunur
- Baslik skorlamasi:
  - "dinlediniz" gecen kolonlar oncelik alir
  - "amaç/amac" gecen kolonlar cezalandirilir
- Bos olmayan satir sayisina gore esitlik bozulur

### 8.3 Sarki ismi normalizasyonu

- Yazim varyasyonlari teklestirilir
- Ozellikle Married Life yazim varyasyonlari tek isimde toplanir

### 8.4 Tablar

1. Model Karsilastirma
- Sarki secimi + audio yukleme
- Insan ortalama VA ile model VA karsilastirma
- Delta metrikleri
- Bar chart + VA uzayi scatter

2. Demografik Analiz
- Yas gruplarina gore VA
- Cinsiyete gore VA
- Muzik dinleme sikligi vs duygu yogunlugu
- Cinsiyet bazli VA uzayi

3. Duygu Degisimi
- Dinlemeden once/sonra dagilimlari
- En etkili sarkilar (derinden etkiledi)

4. Insan vs Gemini AI
- API key ile Gemini analizi
- Secili sarki icin insan ortalamalariyla karsilastirma
- Delta ve VA uzayi gorselleri
- Detayli Gemini yorumu

5. Insan vs CNN vs Gemini
- Uclu karsilastirma
- Metrik kartlari + bar chart + VA uzayi
- Hangi model insanlara daha yakin yorumu

Cache kullanimi:

- Model yukleme: cache_resource
- Model metrik tablosu: cache_data

## 9) Gemini Entegrasyonu

Mevcut uygulama davranisi:

- app.py icinde google.generativeai kullaniliyor
- Ses dosyasi upload edilip modelden JSON formatinda cevap bekleniyor
- 1-9 araligina clamp yapiliyor

Ayrica test scriptinde:

- test_gemini_models.py icinde google.genai kullanimi bulunuyor

Not:

- Ortam uyarilarina gore google.generativeai paketi deprecated durumda
- requirements icinde google-genai tanimli
- Gelecek bakim icin app.py tarafinin da google.genai'ye alinmasi onerilir

## 10) Deney Yonetimi

Sabit split:

- Dosya: configs/fixed_split_seed42.csv
- Seed: 42
- Train/Val/Test: 1261 / 271 / 270

Deney artefaktlari:

- Her deney icin:
  - best model
  - last model
  - predictions csv
  - history csv
  - loss curve
  - scatter (valence/arousal)
  - error histogram

Karsilastirma tablosu:

- results/model_comparison.csv
- Tum modelleri tek tabloda tutar

## 11) Tum Model Metrikleri

Kaynak: results/model_comparison.csv

| model_name | rmse_v | rmse_a | pearson_v | pearson_a | mae_v | mae_a | avg_rmse | avg_pearson | notes |
|---|---|---|---|---|---|---|---|---|---|
| cnn_baseline | 0.8404 | 0.8332 | 0.7152 | 0.7757 | 0.6658 | 0.6601 | 0.8368 | 0.7455 |  |
| cnn_optimized | 0.8448 | 0.8174 | 0.7095 | 0.7789 | 0.6709 | 0.6455 | 0.8311 | 0.7442 | deeper cnn 32-64-128-256 |
| cnn_transformer_v1 | 1.0852 | 1.2899 | 0.4716 | 0.4838 | 0.8713 | 1.0354 | 1.1875 | 0.4777 | cnn + transformer hybrid |
| cnn_v2_dropout_03 | 0.8455 | 0.8381 | 0.7062 | 0.7648 | 0.6708 | 0.6742 | 0.8418 | 0.7355 | dropout 0.3 |
| cnn_v2_dropout_04 | 0.8456 | 0.8612 | 0.6975 | 0.7598 | 0.6766 | 0.6905 | 0.8534 | 0.7286 | dropout 0.4 |
| cnn_v2_dropout_05 | 0.8522 | 0.8773 | 0.6888 | 0.7372 | 0.6824 | 0.6993 | 0.8648 | 0.713 | dropout 0.5 |
| cnn_v2_lr_1e3 | 0.8556 | 0.8904 | 0.6901 | 0.749 | 0.6878 | 0.7111 | 0.873 | 0.7196 | lr 1e-3 |
| cnn_v2_lr_1e4 | 0.8524 | 0.8336 | 0.7059 | 0.7709 | 0.6699 | 0.6616 | 0.843 | 0.7384 | lr 1e-4 |
| cnn_v2_lr_3e4 | 0.8456 | 0.8612 | 0.6975 | 0.7598 | 0.6766 | 0.6905 | 0.8534 | 0.7286 | lr 3e-4 |
| cnn_v2_mse | 0.8456 | 0.8612 | 0.6975 | 0.7598 | 0.6766 | 0.6905 | 0.8534 | 0.7286 | loss mse |
| cnn_v2_mse_ccc | 0.8392 | 0.8414 | 0.7082 | 0.7669 | 0.6697 | 0.6707 | 0.8403 | 0.7375 | loss mse + ccc |
| cnn_v2_noaug | 0.8456 | 0.8612 | 0.6975 | 0.7598 | 0.6766 | 0.6905 | 0.8534 | 0.7286 | v2 no augmentation |
| cnn_v2_opt_adam | 0.8456 | 0.8612 | 0.6975 | 0.7598 | 0.6766 | 0.6905 | 0.8534 | 0.7286 | optimizer adam |
| cnn_v2_smoothl1 | 0.8493 | 0.8516 | 0.6993 | 0.7575 | 0.6798 | 0.6781 | 0.8505 | 0.7284 | loss smoothl1 |
| cnn_v2_specaugment | 0.86 | 0.8453 | 0.7092 | 0.7703 | 0.6896 | 0.6709 | 0.8527 | 0.7397 | specaugment |
| cnn_v2_specaugment_noise | 0.8339 | 0.8493 | 0.7098 | 0.7602 | 0.6665 | 0.6723 | 0.8416 | 0.735 | specaugment + mild noise |
| cnn_v2_wd_0 | 0.8476 | 0.8637 | 0.7015 | 0.7559 | 0.6834 | 0.6874 | 0.8556 | 0.7287 | weight decay 0 |
| cnn_v2_wd_1e4 | 0.8555 | 0.8467 | 0.7024 | 0.7632 | 0.6785 | 0.6747 | 0.8511 | 0.7328 | weight decay 1e-4 |
| cnn_v2_wd_1e5 | 0.8456 | 0.8612 | 0.6975 | 0.7598 | 0.6766 | 0.6905 | 0.8534 | 0.7286 | weight decay 1e-5 |
| crnn_v1 | 0.8561 | 0.8422 | 0.7022 | 0.7692 | 0.6783 | 0.6754 | 0.8492 | 0.7357 | crnn v1 simple |
| crnn_v1_bilstm | 0.8599 | 0.8477 | 0.6985 | 0.7622 | 0.676 | 0.6754 | 0.8538 | 0.7303 | crnn v1 bilstm |
| crnn_v2 | 0.85 | 0.8363 | 0.716 | 0.7678 | 0.6919 | 0.6624 | 0.8431 | 0.7419 | crnn v2: stronger cnn + bilstm(2) + mse_ccc + specaugment |
| transformer_v1 | 0.9441 | 0.9936 | 0.6115 | 0.6582 | 0.7547 | 0.7829 | 0.9689 | 0.6348 | audio transformer baseline |
| transformer_v2 | 0.8963 | 0.9025 | 0.6597 | 0.7205 | 0.728 | 0.7021 | 0.8994 | 0.6901 | audio transformer v2: lower lr + higher dropout + specaugment |

### 11.1 Ham CSV (Birebir)

Asagidaki blok, results/model_comparison.csv dosyasinin birebir ham icerigidir (tam precision):

```csv
model_name,rmse_v,rmse_a,pearson_v,pearson_a,mae_v,mae_a,avg_rmse,avg_pearson,notes
cnn_baseline,0.8404441275838667,0.833243336597497,0.7152336860695186,0.7757169281604743,0.6658183036027131,0.6601205653614468,0.8368437320906819,0.7454753071149964,
cnn_v2_opt_adam,0.8456118593865665,0.8612112507118387,0.6974690773561761,0.7598110914316036,0.6766045685167665,0.6904983948778223,0.8534115550492025,0.7286400843938898,optimizer adam
cnn_v2_lr_1e3,0.855558868925534,0.8904141506883626,0.6901030613850003,0.7490143437511986,0.687846921991419,0.7110766079690721,0.8729865098069483,0.7195587025680994,lr 1e-3
cnn_v2_lr_3e4,0.8456118593865665,0.8612112507118387,0.6974690773561761,0.7598110914316036,0.6766045685167665,0.6904983948778223,0.8534115550492025,0.7286400843938898,lr 3e-4
cnn_v2_lr_1e4,0.8523516708596786,0.8336351916236175,0.7058619513889981,0.7708882608610904,0.6699245982699924,0.6616089948901424,0.842993431241648,0.7383751061250443,lr 1e-4
cnn_v2_dropout_03,0.845463643874014,0.8380918192517671,0.7062214870126554,0.7648086538682211,0.6707984774201005,0.6742395952895835,0.8417777315628905,0.7355150704404383,dropout 0.3
cnn_v2_dropout_04,0.8456118593865665,0.8612112507118387,0.6974690773561761,0.7598110914316036,0.6766045685167665,0.6904983948778223,0.8534115550492025,0.7286400843938898,dropout 0.4
cnn_v2_dropout_05,0.8521985417766034,0.8773110886112694,0.6888388306705763,0.7371977969930588,0.6823610296955815,0.6992667785397282,0.8647548151939364,0.7130183138318176,dropout 0.5
cnn_v2_wd_0,0.8475900547818531,0.863694970472584,0.7014619468478027,0.7559135401410113,0.6833933724297417,0.6874107321103414,0.8556425126272186,0.728687743494407,weight decay 0
cnn_v2_wd_1e5,0.8456118593865665,0.8612112507118387,0.6974690773561761,0.7598110914316036,0.6766045685167665,0.6904983948778223,0.8534115550492025,0.7286400843938898,weight decay 1e-5
cnn_v2_wd_1e4,0.8554581512964349,0.8466845154427113,0.7024076483199975,0.7632026136237257,0.6785451553486012,0.6746950621958132,0.8510713333695731,0.7328051309718615,weight decay 1e-4
cnn_v2_noaug,0.8456118593865665,0.8612112507118387,0.6974690773561761,0.7598110914316036,0.6766045685167665,0.6904983948778223,0.8534115550492025,0.7286400843938898,v2 no augmentation
cnn_v2_specaugment,0.8599723428161664,0.8453338279041284,0.7091620356228616,0.7702797186233337,0.6896033808037086,0.6708833787176344,0.8526530853601474,0.7397208771230976,specaugment
cnn_v2_specaugment_noise,0.8339346390847255,0.8492667649652011,0.7097943585460273,0.7601607175272267,0.6664641777674357,0.672322518295712,0.8416007020249634,0.734977538036627,specaugment + mild noise
cnn_v2_mse,0.8456118593865665,0.8612112507118387,0.6974690773561761,0.7598110914316036,0.6766045685167665,0.6904983948778223,0.8534115550492025,0.7286400843938898,loss mse
cnn_v2_smoothl1,0.8493451815009322,0.8515826554941632,0.6993303791553263,0.7575003554669603,0.6797704740806862,0.6781453675693936,0.8504639184975478,0.7284153673111433,loss smoothl1
cnn_v2_mse_ccc,0.8391565018871632,0.8413891495740325,0.7081791455510668,0.7668845673641809,0.6697468271961918,0.6707430790971827,0.8402728257305978,0.7375318564576239,loss mse + ccc
cnn_optimized,0.8448238791633067,0.8173670840985127,0.7095359532661284,0.7789135835945606,0.6709278945569639,0.6454790155092875,0.8310954816309097,0.7442247684303445,deeper cnn 32-64-128-256
crnn_v1,0.856135137264743,0.8422307109593246,0.7021748229192686,0.7692279121609676,0.6783446426744815,0.6754099823810437,0.8491829241120338,0.7357013675401181,crnn v1 simple
crnn_v1_bilstm,0.8599020113366034,0.8477068422634132,0.6985224066579572,0.7621653462855104,0.6760483176619918,0.6753849996460809,0.8538044268000082,0.7303438764717338,crnn v1 bilstm
crnn_v2,0.8499622461521826,0.8363099348641375,0.715962926308561,0.7678478860688851,0.691886677565398,0.6623899746824193,0.8431360905081601,0.741905406188723,crnn v2: stronger cnn + bilstm(2) + mse_ccc + specaugment
transformer_v1,0.9441322885071556,0.9936244056869328,0.6114597540613707,0.6581831985806688,0.7546996134298819,0.7828986287117005,0.968878347097044,0.6348214763210197,audio transformer baseline
transformer_v2,0.8963114502762933,0.9025372826455288,0.6597149155054409,0.7205299781553826,0.7280376681575069,0.7021323685292844,0.8994243664609111,0.6901224468304117,audio transformer v2: lower lr + higher dropout + specaugment
cnn_transformer_v1,1.0851695845422529,1.2898833834711458,0.4715841245987638,0.48383469912112925,0.8712763450763844,1.0354112479421826,1.1875264840066992,0.47770941185994653,cnn + transformer hybrid
```

### 11.2 En iyi skorlar

- En iyi avg_rmse: cnn_optimized (0.8311)
- En iyi avg_pearson: cnn_baseline (0.7455)
- En iyi rmse_v: cnn_v2_specaugment_noise (0.8339)
- En iyi rmse_a: cnn_optimized (0.8174)
- En iyi pearson_v: crnn_v2 (0.7160)
- En iyi pearson_a: cnn_optimized (0.7789)

## 12) Egitim Gecmisi Ozeti (Tum Deneyler)

Kaynak: logs/*_history.csv

| experiment | epochs_ran | best_epoch | best_val_loss | last_val_loss |
|---|---|---|---|---|
| cnn_baseline | 35 | 32 | 0.8129 | 0.8323 |
| cnn_optimized | 27 | 21 | 0.8809 | 0.9613 |
| cnn_transformer_v1 | 13 | 1 | 1.8192 | 2.4391 |
| cnn_v2_dropout_03 | 23 | 17 | 0.8155 | 0.8838 |
| cnn_v2_dropout_04 | 11 | 5 | 0.8756 | 0.9067 |
| cnn_v2_dropout_05 | 12 | 6 | 0.9072 | 0.9593 |
| cnn_v2_lr_1e3 | 9 | 3 | 0.8787 | 1.0835 |
| cnn_v2_lr_1e4 | 29 | 23 | 0.8228 | 0.8507 |
| cnn_v2_lr_3e4 | 11 | 5 | 0.8756 | 0.9067 |
| cnn_v2_mse | 11 | 5 | 0.8756 | 0.9067 |
| cnn_v2_mse_ccc | 21 | 15 | 0.9572 | 0.958 |
| cnn_v2_noaug | 11 | 5 | 0.8756 | 0.9067 |
| cnn_v2_opt_adam | 11 | 5 | 0.8756 | 0.9067 |
| cnn_v2_smoothl1 | 12 | 6 | 0.3669 | 0.4227 |
| cnn_v2_specaugment | 23 | 17 | 0.8275 | 0.8294 |
| cnn_v2_specaugment_noise | 18 | 12 | 0.83 | 0.8514 |
| cnn_v2_wd_0 | 12 | 6 | 0.8632 | 0.9712 |
| cnn_v2_wd_1e4 | 17 | 11 | 0.8315 | 0.8342 |
| cnn_v2_wd_1e5 | 11 | 5 | 0.8756 | 0.9067 |
| crnn_v1 | 21 | 15 | 0.8373 | 0.8491 |
| crnn_v1_bilstm | 19 | 13 | 0.8528 | 0.9359 |
| crnn_v2 | 58 | 48 | 0.9622 | 0.9795 |
| transformer_v1 | 20 | 8 | 1.118 | 1.2186 |
| transformer_v2 | 37 | 22 | 1.0322 | 1.1913 |

## 13) Konfig Envanteri

Toplam config dosyasi: 25

- cnn_baseline.yaml
- cnn_optimized.yaml
- cnn_transformer_v1.yaml
- cnn_v2_dropout_03.yaml
- cnn_v2_dropout_04.yaml
- cnn_v2_dropout_05.yaml
- cnn_v2_lr_1e3.yaml
- cnn_v2_lr_1e4.yaml
- cnn_v2_lr_3e4.yaml
- cnn_v2_mse.yaml
- cnn_v2_mse_ccc.yaml
- cnn_v2_noaug.yaml
- cnn_v2_opt_adam.yaml
- cnn_v2_opt_adamw.yaml
- cnn_v2_smoothl1.yaml
- cnn_v2_specaugment.yaml
- cnn_v2_specaugment_noise.yaml
- cnn_v2_wd_0.yaml
- cnn_v2_wd_1e4.yaml
- cnn_v2_wd_1e5.yaml
- crnn_v1.yaml
- crnn_v1_bilstm.yaml
- crnn_v2.yaml
- transformer_v1.yaml
- transformer_v2.yaml

## 14) Model Artefakt Envanteri

- Best checkpoint sayisi: 24
- Last checkpoint sayisi: 24

Best checkpoint listesi:

- cnn_baseline_best.pt
- cnn_optimized_best.pt
- cnn_transformer_v1_best.pt
- cnn_v2_dropout_03_best.pt
- cnn_v2_dropout_04_best.pt
- cnn_v2_dropout_05_best.pt
- cnn_v2_lr_1e3_best.pt
- cnn_v2_lr_1e4_best.pt
- cnn_v2_lr_3e4_best.pt
- cnn_v2_mse_best.pt
- cnn_v2_mse_ccc_best.pt
- cnn_v2_noaug_best.pt
- cnn_v2_opt_adam_best.pt
- cnn_v2_smoothl1_best.pt
- cnn_v2_specaugment_best.pt
- cnn_v2_specaugment_noise_best.pt
- cnn_v2_wd_0_best.pt
- cnn_v2_wd_1e4_best.pt
- cnn_v2_wd_1e5_best.pt
- crnn_v1_best.pt
- crnn_v1_bilstm_best.pt
- crnn_v2_best.pt
- transformer_v1_best.pt
- transformer_v2_best.pt

## 15) Bagimliliklar

requirements.txt icerigi:

- numpy>=1.24.0,<2.0
- pandas>=2.0.0
- scikit-learn>=1.3.0
- matplotlib>=3.7.0
- seaborn>=0.12.0
- google-genai>=0.2.0
- Pillow>=10.0.0
- jupyter>=1.0.0
- tensorflow>=2.15.0
- keras>=2.15.0
- joblib>=1.3.0
- torch>=2.0.0
- torchaudio>=2.0.0
- soundfile>=0.12.0
- streamlit>=1.28.0

## 16) Calistirma Talimatlari

### 16.1 Uygulama

Windows + venv:

1. venv aktif et
2. su komutu calistir

python -m streamlit run app.py

Varsayilan URL:

- Local: http://localhost:8501

### 16.2 Egitim

Baseline:

python src/train_deam_cnn_va.py

CNN deneyleri:

python src/train_cnn_experiments.py --preset cnn_optimized --experiment-name cnn_optimized

CRNN/Transformer deneyleri:

python src/train_crnn_experiments.py --preset crnn_v2 --experiment-name crnn_v2
python src/train_crnn_experiments.py --preset transformer_v2 --experiment-name transformer_v2

En iyi model secimi:

python src/train_cnn_experiments.py --select-best-cnn
python src/train_crnn_experiments.py --select-best-crnn

## 17) Kritik Notlar ve Bilinen Durumlar

1. Deprecated Gemini istemcisi
- app.py, google.generativeai kullaniyor
- test scripti ve requirements, google.genai kullaniyor
- Tek istemciye gecis (google.genai) onerilir

2. Deney logu ve metrik tablosu farki
- logs/experiment_log.csv icinde cnn_v2_opt_adamw var
- results/model_comparison.csv icinde bu modele ait satir yok
- Bu deney tekrar degerlendirilip tabloya yazilmali

3. CSV kolon adlarinda bosluklar
- Anket kolonlarinda son bosluklar mevcut
- Kod su an bu kolon adlarini bekliyor; farkli exportta dikkat edilmeli

4. Coklu/tekrarli basliklar
- Anket CSV'sinde benzer sarki kolonlari olabilir
- Uygulama en dogru kolonu secmek icin skor tabanli yontem kullaniyor

## 18) Sonuc

Bu proje:

- Ucten fazla model ailesini tek platformda yonetir
- Sabit split ile adil deney karsilastirmasi sunar
- Tum deney metriklerini tek tabloya toplar
- Anket tabanli insan algisini model ve Gemini ile birlestiren uygulamayi calistirir

Mevcut durumda sistem uretim/tez sunumu icin yeterli olgunluktadir. En onemli teknik iyilestirme adimi, Gemini istemcisinin modern API'ye tasinmasidir.
