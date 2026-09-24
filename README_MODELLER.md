# Eğitilen Modeller ve Sonuçlar README

Hazırlama Tarihi: 09.06.2026

Bu dosya, projede eğitilen tüm müzik duygu analizi modellerini, deney ayarlarını, checkpoint envanterini, eğitim geçmişlerini, test metriklerini ve ensemble sonuçlarını tek yerde özetler.

## 1) Kapsam

Projede DEAM veri seti üzerinde Valence-Arousal regresyonu için birden fazla model ailesi eğitilmiştir:

- CNN baseline
- CNN varyasyon deneyleri
- Optimized CNN
- CRNN
- Transformer
- CNN + Transformer hibrit model
- Simple average ensemble
- Manual weighted ensemble
- Stacking ensemble

Temel çıktı iki boyutludur:

- `Valence`: Duygunun negatif-pozitif ekseni
- `Arousal`: Duygunun sakin-enerjik ekseni

## 2) Veri Bölünmesi

Tüm ana deneylerde sabit split kullanılmıştır.

| Özellik | Değer |
|---|---:|
| Seed | 42 |
| Train örnek sayısı | 1261 |
| Validation örnek sayısı | 271 |
| Test örnek sayısı | 270 |
| Split dosyası | `configs/fixed_split_seed42.csv` |
| Ground truth | `ground_truth.csv` |

Sabit split kullanımı, model aileleri ve hiperparametre deneyleri arasında daha adil karşılaştırma yapılmasını sağlar.

## 3) Metrik Açıklamaları

| Metrik | Anlamı | Yorum |
|---|---|---|
| `rmse_v` | Valence RMSE | Düşük daha iyi |
| `rmse_a` | Arousal RMSE | Düşük daha iyi |
| `mae_v` | Valence MAE | Düşük daha iyi |
| `mae_a` | Arousal MAE | Düşük daha iyi |
| `pearson_v` | Valence Pearson korelasyonu | Yüksek daha iyi |
| `pearson_a` | Arousal Pearson korelasyonu | Yüksek daha iyi |
| `avg_rmse` | Valence ve Arousal RMSE ortalaması | Düşük daha iyi |
| `avg_pearson` | Valence ve Arousal Pearson ortalaması | Yüksek daha iyi |
| `ccc_v`, `ccc_a` | Concordance Correlation Coefficient | Yüksek daha iyi |

Ana karşılaştırma dosyaları:

- `results/model_comparison.csv`
- `results/ensemble_comparison.csv`
- `results/model_comparison_with_stacking.csv`
- `results/learned_weights.txt`

## 4) En İyi Tekil Model Skorları

Tekil model karşılaştırması `results/model_comparison.csv` üzerinden yapılmıştır.

| Kriter | En iyi model | Skor |
|---|---|---:|
| En iyi `avg_rmse` | `cnn_optimized` | 0.8311 |
| En iyi `avg_pearson` | `cnn_baseline` | 0.7455 |
| En iyi `rmse_v` | `cnn_v2_specaugment_noise` | 0.8339 |
| En iyi `rmse_a` | `cnn_optimized` | 0.8174 |
| En iyi `pearson_v` | `crnn_v2` | 0.7160 |
| En iyi `pearson_a` | `cnn_optimized` | 0.7789 |
| En iyi `mae_v` | `cnn_baseline` | 0.6658 |
| En iyi `mae_a` | `cnn_optimized` | 0.6455 |

Genel tekil model yorumu:

- `cnn_optimized`, ortalama hata ve Arousal başarısı açısından en güçlü tekil modeldir.
- `cnn_baseline`, Pearson ortalaması ve Valence MAE açısından çok güçlüdür.
- `crnn_v2`, Valence Pearson korelasyonunda en yüksek tekil skoru üretmiştir.
- Transformer tabanlı modeller bu veri ve deney ayarlarında CNN/CRNN modellerinin gerisinde kalmıştır.

## 5) Tüm Tekil Model Test Sonuçları

Tablo `avg_rmse` değerine göre küçükten büyüğe sıralanmıştır.

| rank | model | rmse_v | rmse_a | pearson_v | pearson_a | mae_v | mae_a | avg_rmse | avg_pearson | notes |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `cnn_optimized` | 0.8448 | 0.8174 | 0.7095 | 0.7789 | 0.6709 | 0.6455 | 0.8311 | 0.7442 | deeper cnn 32-64-128-256 |
| 2 | `cnn_baseline` | 0.8404 | 0.8332 | 0.7152 | 0.7757 | 0.6658 | 0.6601 | 0.8368 | 0.7455 |  |
| 3 | `cnn_v2_mse_ccc` | 0.8392 | 0.8414 | 0.7082 | 0.7669 | 0.6697 | 0.6707 | 0.8403 | 0.7375 | loss mse + ccc |
| 4 | `cnn_v2_specaugment_noise` | 0.8339 | 0.8493 | 0.7098 | 0.7602 | 0.6665 | 0.6723 | 0.8416 | 0.7350 | specaugment + mild noise |
| 5 | `cnn_v2_dropout_03` | 0.8455 | 0.8381 | 0.7062 | 0.7648 | 0.6708 | 0.6742 | 0.8418 | 0.7355 | dropout 0.3 |
| 6 | `cnn_v2_lr_1e4` | 0.8524 | 0.8336 | 0.7059 | 0.7709 | 0.6699 | 0.6616 | 0.8430 | 0.7384 | lr 1e-4 |
| 7 | `crnn_v2` | 0.8500 | 0.8363 | 0.7160 | 0.7678 | 0.6919 | 0.6624 | 0.8431 | 0.7419 | crnn v2: stronger cnn + bilstm(2) + mse_ccc + specaugment |
| 8 | `crnn_v1` | 0.8561 | 0.8422 | 0.7022 | 0.7692 | 0.6783 | 0.6754 | 0.8492 | 0.7357 | crnn v1 simple |
| 9 | `cnn_v2_smoothl1` | 0.8493 | 0.8516 | 0.6993 | 0.7575 | 0.6798 | 0.6781 | 0.8505 | 0.7284 | loss smoothl1 |
| 10 | `cnn_v2_wd_1e4` | 0.8555 | 0.8467 | 0.7024 | 0.7632 | 0.6785 | 0.6747 | 0.8511 | 0.7328 | weight decay 1e-4 |
| 11 | `cnn_v2_specaugment` | 0.8600 | 0.8453 | 0.7092 | 0.7703 | 0.6896 | 0.6709 | 0.8527 | 0.7397 | specaugment |
| 12 | `cnn_v2_dropout_04` | 0.8456 | 0.8612 | 0.6975 | 0.7598 | 0.6766 | 0.6905 | 0.8534 | 0.7286 | dropout 0.4 |
| 13 | `cnn_v2_wd_1e5` | 0.8456 | 0.8612 | 0.6975 | 0.7598 | 0.6766 | 0.6905 | 0.8534 | 0.7286 | weight decay 1e-5 |
| 14 | `cnn_v2_noaug` | 0.8456 | 0.8612 | 0.6975 | 0.7598 | 0.6766 | 0.6905 | 0.8534 | 0.7286 | v2 no augmentation |
| 15 | `cnn_v2_lr_3e4` | 0.8456 | 0.8612 | 0.6975 | 0.7598 | 0.6766 | 0.6905 | 0.8534 | 0.7286 | lr 3e-4 |
| 16 | `cnn_v2_opt_adam` | 0.8456 | 0.8612 | 0.6975 | 0.7598 | 0.6766 | 0.6905 | 0.8534 | 0.7286 | optimizer adam |
| 17 | `cnn_v2_mse` | 0.8456 | 0.8612 | 0.6975 | 0.7598 | 0.6766 | 0.6905 | 0.8534 | 0.7286 | loss mse |
| 18 | `crnn_v1_bilstm` | 0.8599 | 0.8477 | 0.6985 | 0.7622 | 0.6760 | 0.6754 | 0.8538 | 0.7303 | crnn v1 bilstm |
| 19 | `cnn_v2_wd_0` | 0.8476 | 0.8637 | 0.7015 | 0.7559 | 0.6834 | 0.6874 | 0.8556 | 0.7287 | weight decay 0 |
| 20 | `cnn_v2_dropout_05` | 0.8522 | 0.8773 | 0.6888 | 0.7372 | 0.6824 | 0.6993 | 0.8648 | 0.7130 | dropout 0.5 |
| 21 | `cnn_v2_lr_1e3` | 0.8556 | 0.8904 | 0.6901 | 0.7490 | 0.6878 | 0.7111 | 0.8730 | 0.7196 | lr 1e-3 |
| 22 | `transformer_v2` | 0.8963 | 0.9025 | 0.6597 | 0.7205 | 0.7280 | 0.7021 | 0.8994 | 0.6901 | audio transformer v2: lower lr + higher dropout + specaugment |
| 23 | `transformer_v1` | 0.9441 | 0.9936 | 0.6115 | 0.6582 | 0.7547 | 0.7829 | 0.9689 | 0.6348 | audio transformer baseline |
| 24 | `cnn_transformer_v1` | 1.0852 | 1.2899 | 0.4716 | 0.4838 | 0.8713 | 1.0354 | 1.1875 | 0.4777 | cnn + transformer hybrid |

## 6) Ensemble Sonuçları

Kaynak dosya: `results/ensemble_comparison.csv`

| model | rmse_v | rmse_a | pearson_v | pearson_a | avg_rmse | avg_pearson |
|---|---:|---:|---:|---:|---:|---:|
| `ensemble_weighted` | 0.8239 | 0.8033 | 0.7317 | 0.7911 | 0.8136 | 0.7614 |
| `ensemble_simple_average` | 0.8242 | 0.8052 | 0.7309 | 0.7926 | 0.8147 | 0.7617 |
| `cnn_optimized` | 0.8448 | 0.8174 | 0.7095 | 0.7789 | 0.8311 | 0.7442 |
| `cnn_baseline` | 0.8404 | 0.8332 | 0.7152 | 0.7757 | 0.8368 | 0.7455 |
| `crnn_v2` | 0.8500 | 0.8363 | 0.7160 | 0.7678 | 0.8431 | 0.7419 |

Ensemble bileşenleri:

- `cnn_optimized`
- `cnn_baseline`
- `crnn_v2`

Uygulamada kullanılan manuel ağırlıklar:

| Bileşen | Ağırlık |
|---|---:|
| `cnn_optimized` | 0.30 |
| `cnn_baseline` | 0.30 |
| `crnn_v2` | 0.40 |

Ensemble yorumu:

- `ensemble_weighted`, tekil modellerden daha düşük `avg_rmse` üretmiştir.
- `ensemble_simple_average`, en yüksek `avg_pearson` değerini üretmiştir.
- Ensemble yaklaşımı, üç güçlü ama farklı hata profiline sahip modeli birleştirdiği için genel performansı iyileştirmiştir.

## 7) Stacking Ensemble Sonuçları

Kaynak dosyalar:

- `results/model_comparison_with_stacking.csv`
- `results/learned_weights.txt`

| model | rmse_v | rmse_a | pearson_v | pearson_a | ccc_v | ccc_a | avg_rmse | avg_pearson | avg_ccc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cnn_optimized` | 0.8448 | 0.8174 | 0.7095 | 0.7789 | 0.6651 | 0.7362 | 0.8311 | 0.7442 | 0.7006 |
| `cnn_baseline` | 0.8404 | 0.8332 | 0.7152 | 0.7757 | 0.6438 | 0.7103 | 0.8368 | 0.7455 | 0.6771 |
| `crnn_v2` | 0.8500 | 0.8363 | 0.7160 | 0.7678 | 0.6686 | 0.7224 | 0.8431 | 0.7419 | 0.6955 |
| `simple_average_ensemble` | 0.8242 | 0.8052 | 0.7309 | 0.7926 | 0.6707 | 0.7347 | 0.8147 | 0.7617 | 0.7027 |
| `weighted_ensemble_manual` | 0.8239 | 0.8033 | 0.7317 | 0.7911 | 0.6720 | 0.7402 | 0.8136 | 0.7614 | 0.7061 |
| `stacking_ensemble_cv_ridge_alpha_1.0` | 0.8112 | 0.8163 | 0.7234 | 0.7771 | 0.6909 | 0.7585 | 0.8137 | 0.7503 | 0.7247 |

Stacking CV sonuçları:

| Stacking modeli | RMSE V | RMSE A | Pearson V | Pearson A | CCC V | CCC A |
|---|---:|---:|---:|---:|---:|---:|
| `linear_regression` | 0.8123 | 0.8180 | 0.7226 | 0.7762 | 0.6909 | 0.7581 |
| `ridge_alpha_1.0` | 0.8112 | 0.8163 | 0.7234 | 0.7771 | 0.6909 | 0.7585 |
| `lasso_alpha_0.001` | 0.8121 | 0.8180 | 0.7227 | 0.7762 | 0.6907 | 0.7579 |

Seçilen stacking modeli:

- `ridge_alpha_1.0`

Ridge stacking öğrenilmiş ağırlıkları:

| Hedef | cnn_optimized | cnn_baseline | crnn_v2 | intercept |
|---|---:|---:|---:|---:|
| Valence | 0.1517 | 0.4352 | 0.4851 | -0.1556 |
| Arousal | 0.3658 | 0.3732 | 0.4341 | -0.7975 |

Stacking notu:

- Ridge stacking `avg_ccc` açısından en güçlü sonucu üretmiştir.
- Canlı uygulama için manuel weighted ensemble daha sade ve yorumlanabilir olduğu için tercih edilmiştir.
- Stacking yaklaşımı canlı tahmine alınacaksa eğitim/test ayrımı ve veri sızıntısı tekrar denetlenmelidir.

## 8) Deney Konfigürasyon Özeti

Kaynak dosya: `logs/experiment_log.csv`

| experiment | optimizer | lr | dropout | wd | loss | aug | architecture | notes |
|---|---|---:|---:|---:|---|---|---|---|
| `cnn_baseline` |  |  |  |  |  |  |  |  |
| `cnn_v2_opt_adam` | adam | 0.0003 | 0.4 | 1e-05 | mse | none | baseline | optimizer adam |
| `cnn_v2_opt_adamw` | adamw | 0.0003 | 0.4 | 1e-05 | mse | none | baseline | optimizer adamw |
| `cnn_v2_lr_1e3` | adam | 0.001 | 0.4 | 1e-05 | mse | none | baseline | lr 1e-3 |
| `cnn_v2_lr_3e4` | adam | 0.0003 | 0.4 | 1e-05 | mse | none | baseline | lr 3e-4 |
| `cnn_v2_lr_1e4` | adam | 0.0001 | 0.4 | 1e-05 | mse | none | baseline | lr 1e-4 |
| `cnn_v2_dropout_03` | adam | 0.0003 | 0.3 | 1e-05 | mse | none | baseline | dropout 0.3 |
| `cnn_v2_dropout_04` | adam | 0.0003 | 0.4 | 1e-05 | mse | none | baseline | dropout 0.4 |
| `cnn_v2_dropout_05` | adam | 0.0003 | 0.5 | 1e-05 | mse | none | baseline | dropout 0.5 |
| `cnn_v2_wd_0` | adam | 0.0003 | 0.4 | 0.0 | mse | none | baseline | weight decay 0 |
| `cnn_v2_wd_1e5` | adam | 0.0003 | 0.4 | 1e-05 | mse | none | baseline | weight decay 1e-5 |
| `cnn_v2_wd_1e4` | adam | 0.0003 | 0.4 | 0.0001 | mse | none | baseline | weight decay 1e-4 |
| `cnn_v2_noaug` | adam | 0.0003 | 0.4 | 1e-05 | mse | none | baseline | v2 no augmentation |
| `cnn_v2_specaugment` | adam | 0.0003 | 0.4 | 1e-05 | mse | specaugment | baseline | specaugment |
| `cnn_v2_specaugment_noise` | adam | 0.0003 | 0.4 | 1e-05 | mse | specaugment_noise | baseline | specaugment + mild noise |
| `cnn_v2_mse` | adam | 0.0003 | 0.4 | 1e-05 | mse | none | baseline | loss mse |
| `cnn_v2_smoothl1` | adam | 0.0003 | 0.4 | 1e-05 | smoothl1 | none | baseline | loss smoothl1 |
| `cnn_v2_mse_ccc` | adam | 0.0003 | 0.4 | 1e-05 | mse_ccc | none | baseline | loss mse + ccc |
| `cnn_optimized` | adamw | 0.0003 | 0.4 | 0.0001 | mse_ccc | specaugment_noise | optimized | deeper cnn 32-64-128-256 |
| `crnn_v1` | adam | 0.0003 | 0.4 | 1e-05 | mse | none |  | crnn v1 simple |
| `crnn_v1_bilstm` | adam | 0.0003 | 0.4 | 1e-05 | mse | none |  | crnn v1 bilstm |
| `crnn_v2` | adam | 0.001 | 0.3 | 1e-05 | mse_ccc | specaugment |  | crnn v2: stronger cnn + bilstm(2) + mse_ccc + specaugment |
| `transformer_v1` | adamw | 0.0003 | 0.3 | 0.0001 | mse_ccc | specaugment_noise |  | audio transformer baseline |
| `transformer_v2` | adamw | 0.0001 | 0.4 | 0.0001 | mse_ccc | specaugment |  | audio transformer v2: lower lr + higher dropout + specaugment |
| `cnn_transformer_v1` | adamw | 0.0003 | 0.3 | 0.0001 | mse_ccc | specaugment_noise |  | cnn + transformer hybrid |

Not:

- `cnn_v2_opt_adamw` için config ve deney logu vardır, ancak `results/model_comparison.csv` içinde karşılık gelen test metrik satırı bulunmamaktadır.
- Bu model tekrar değerlendirilmeli veya tabloya bilinçli olarak dahil edilmediği açıklanmalıdır.

## 9) Eğitim Geçmişi Özeti

Kaynak dosyalar: `logs/*_history.csv`

| experiment | epochs | best_epoch | best_val_loss | last_val_loss | last_train_loss |
|---|---:|---:|---:|---:|---:|
| `cnn_baseline` | 35 | 32 | 0.8129 | 0.8323 | 1.6378 |
| `cnn_optimized` | 27 | 21 | 0.8809 | 0.9613 | 1.4370 |
| `cnn_transformer_v1` | 13 | 1 | 1.8192 | 2.4391 | 2.0553 |
| `cnn_v2_dropout_03` | 23 | 17 | 0.8155 | 0.8838 | 1.2055 |
| `cnn_v2_dropout_04` | 11 | 5 | 0.8756 | 0.9067 | 1.6063 |
| `cnn_v2_dropout_05` | 12 | 6 | 0.9072 | 0.9593 | 2.0060 |
| `cnn_v2_lr_1e3` | 9 | 3 | 0.8787 | 1.0835 | 1.5684 |
| `cnn_v2_lr_1e4` | 29 | 23 | 0.8228 | 0.8507 | 1.7521 |
| `cnn_v2_lr_3e4` | 11 | 5 | 0.8756 | 0.9067 | 1.6063 |
| `cnn_v2_mse_ccc` | 21 | 15 | 0.9572 | 0.9580 | 1.6964 |
| `cnn_v2_mse` | 11 | 5 | 0.8756 | 0.9067 | 1.6063 |
| `cnn_v2_noaug` | 11 | 5 | 0.8756 | 0.9067 | 1.6063 |
| `cnn_v2_opt_adam` | 11 | 5 | 0.8756 | 0.9067 | 1.6063 |
| `cnn_v2_smoothl1` | 12 | 6 | 0.3669 | 0.4227 | 0.6107 |
| `cnn_v2_specaugment` | 23 | 17 | 0.8275 | 0.8294 | 1.5105 |
| `cnn_v2_specaugment_noise` | 18 | 12 | 0.8300 | 0.8514 | 1.6402 |
| `cnn_v2_wd_0` | 12 | 6 | 0.8632 | 0.9712 | 1.6312 |
| `cnn_v2_wd_1e4` | 17 | 11 | 0.8315 | 0.8342 | 1.5216 |
| `cnn_v2_wd_1e5` | 11 | 5 | 0.8756 | 0.9067 | 1.6063 |
| `crnn_v1_bilstm` | 19 | 13 | 0.8528 | 0.9359 | 1.4259 |
| `crnn_v1` | 21 | 15 | 0.8373 | 0.8491 | 1.2738 |
| `crnn_v2` | 58 | 48 | 0.9622 | 0.9795 | 0.9634 |
| `transformer_v1` | 20 | 8 | 1.1180 | 1.2186 | 1.1993 |
| `transformer_v2` | 37 | 22 | 1.0322 | 1.1913 | 0.7448 |

Eğitim yorumu:

- `cnn_baseline`, validation loss açısından en düşük kayıplardan birini üretmiştir.
- `crnn_v2`, en uzun eğitim koşan deneydir ve 58 epoch çalışmıştır.
- `cnn_transformer_v1`, ilk epochta en iyi validation loss değerini görmüş, sonraki epochlarda belirgin şekilde kötüleşmiştir.
- `cnn_v2_smoothl1` farklı loss ölçeği kullandığı için validation loss değeri diğer MSE tabanlı deneylerle doğrudan kıyaslanmamalıdır.

## 10) Checkpoint Envanteri

Checkpoint klasörü:

- `models/`

Genel durum:

| Tür | Sayı |
|---|---:|
| Ana deney best checkpoint | 24 |
| Ana deney last checkpoint | 24 |
| Eski tekil checkpoint | 1 |
| Toplam `.pt` dosyası | 49 |

### 10.1 Best Checkpoint Dosyaları

- `cnn_baseline_best.pt`
- `cnn_optimized_best.pt`
- `cnn_transformer_v1_best.pt`
- `cnn_v2_dropout_03_best.pt`
- `cnn_v2_dropout_04_best.pt`
- `cnn_v2_dropout_05_best.pt`
- `cnn_v2_lr_1e3_best.pt`
- `cnn_v2_lr_1e4_best.pt`
- `cnn_v2_lr_3e4_best.pt`
- `cnn_v2_mse_best.pt`
- `cnn_v2_mse_ccc_best.pt`
- `cnn_v2_noaug_best.pt`
- `cnn_v2_opt_adam_best.pt`
- `cnn_v2_smoothl1_best.pt`
- `cnn_v2_specaugment_best.pt`
- `cnn_v2_specaugment_noise_best.pt`
- `cnn_v2_wd_0_best.pt`
- `cnn_v2_wd_1e4_best.pt`
- `cnn_v2_wd_1e5_best.pt`
- `crnn_v1_best.pt`
- `crnn_v1_bilstm_best.pt`
- `crnn_v2_best.pt`
- `transformer_v1_best.pt`
- `transformer_v2_best.pt`

### 10.2 Last Checkpoint Dosyaları

- `cnn_baseline_last.pt`
- `cnn_optimized_last.pt`
- `cnn_transformer_v1_last.pt`
- `cnn_v2_dropout_03_last.pt`
- `cnn_v2_dropout_04_last.pt`
- `cnn_v2_dropout_05_last.pt`
- `cnn_v2_lr_1e3_last.pt`
- `cnn_v2_lr_1e4_last.pt`
- `cnn_v2_lr_3e4_last.pt`
- `cnn_v2_mse_last.pt`
- `cnn_v2_mse_ccc_last.pt`
- `cnn_v2_noaug_last.pt`
- `cnn_v2_opt_adam_last.pt`
- `cnn_v2_smoothl1_last.pt`
- `cnn_v2_specaugment_last.pt`
- `cnn_v2_specaugment_noise_last.pt`
- `cnn_v2_wd_0_last.pt`
- `cnn_v2_wd_1e4_last.pt`
- `cnn_v2_wd_1e5_last.pt`
- `crnn_v1_last.pt`
- `crnn_v1_bilstm_last.pt`
- `crnn_v2_last.pt`
- `transformer_v1_last.pt`
- `transformer_v2_last.pt`

### 10.3 Legacy Checkpoint

- `deam_cnn_va.pt`

Bu dosya eski/baseline eğitimden kalmış legacy checkpoint olarak değerlendirilmelidir.

## 11) Config Envanteri

Config klasörü:

- `configs/`

Toplam ana config dosyası:

- 25 model config dosyası
- 1 sabit split dosyası

Config dosyaları:

- `cnn_baseline.yaml`
- `cnn_optimized.yaml`
- `cnn_transformer_v1.yaml`
- `cnn_v2_dropout_03.yaml`
- `cnn_v2_dropout_04.yaml`
- `cnn_v2_dropout_05.yaml`
- `cnn_v2_lr_1e3.yaml`
- `cnn_v2_lr_1e4.yaml`
- `cnn_v2_lr_3e4.yaml`
- `cnn_v2_mse.yaml`
- `cnn_v2_mse_ccc.yaml`
- `cnn_v2_noaug.yaml`
- `cnn_v2_opt_adam.yaml`
- `cnn_v2_opt_adamw.yaml`
- `cnn_v2_smoothl1.yaml`
- `cnn_v2_specaugment.yaml`
- `cnn_v2_specaugment_noise.yaml`
- `cnn_v2_wd_0.yaml`
- `cnn_v2_wd_1e4.yaml`
- `cnn_v2_wd_1e5.yaml`
- `crnn_v1.yaml`
- `crnn_v1_bilstm.yaml`
- `crnn_v2.yaml`
- `transformer_v1.yaml`
- `transformer_v2.yaml`

Sabit split:

- `fixed_split_seed42.csv`

## 12) Sonuç Artefaktları

Her ana deney için genel olarak şu artefaktlar üretilmiştir:

- Test tahminleri: `results/<experiment>_predictions.csv`
- Eğitim geçmişi: `logs/<experiment>_history.csv`
- Loss grafiği: `results/<experiment>_loss_curve.png`
- Valence scatter grafiği: `results/<experiment>_scatter_valence.png`
- Arousal scatter grafiği: `results/<experiment>_scatter_arousal.png`
- Hata histogramı: `results/<experiment>_error_hist.png`
- Best checkpoint: `models/<experiment>_best.pt`
- Last checkpoint: `models/<experiment>_last.pt`

Toplu sonuç dosyaları:

- `results/model_comparison.csv`
- `results/ensemble_comparison.csv`
- `results/model_comparison_with_stacking.csv`
- `results/ensemble_predictions.csv`
- `results/ensemble_weighted_predictions.csv`
- `results/stacking_predictions.csv`
- `results/learned_weights.txt`

## 13) Model Ailelerine Göre Değerlendirme

### 13.1 CNN Ailesi

CNN ailesi projedeki en dengeli sonuçları üretmiştir.

Öne çıkanlar:

- `cnn_optimized`: En iyi tekil `avg_rmse`
- `cnn_baseline`: En iyi tekil `avg_pearson`
- `cnn_v2_mse_ccc`: Loss fonksiyonu değişikliği ile güçlü sonuç
- `cnn_v2_specaugment_noise`: En iyi Valence RMSE

Sonuç:

- CNN tabanlı mimariler, DEAM veri seti ve Mel-Spectrogram girdisi için en güvenilir model ailesidir.

### 13.2 CRNN Ailesi

CRNN ailesi CNN özellik çıkarımı ile LSTM temporal modellemeyi birleştirir.

Öne çıkanlar:

- `crnn_v2`: En iyi Valence Pearson
- `crnn_v1`: Dengeli ama CNN optimized kadar güçlü olmayan sonuç
- `crnn_v1_bilstm`: BiLSTM eklemesi beklenen iyileştirmeyi sağlamamıştır

Sonuç:

- CRNN, korelasyon tarafında güçlüdür ancak hata metriklerinde optimized CNN ve ensemble modellerin gerisindedir.

### 13.3 Transformer Ailesi

Transformer ailesi bu deneylerde CNN/CRNN kadar başarılı olmamıştır.

Öne çıkanlar:

- `transformer_v2`, `transformer_v1` modeline göre belirgin iyileşmiştir.
- Yine de `avg_rmse` ve `avg_pearson` değerleri CNN ailesinin gerisindedir.

Sonuç:

- Veri miktarı, model kapasitesi veya hiperparametreler Transformer modeller için yeterince uygun olmayabilir.

### 13.4 CNN + Transformer Hibrit

`cnn_transformer_v1`, tüm ana deneyler içinde en zayıf sonucu üretmiştir.

Muhtemel nedenler:

- Model kapasitesinin veriye göre yüksek kalması
- Hiperparametrelerin oturmaması
- Erken overfitting
- CNN özellikleri ile Transformer katmanları arasında yeterli temsil uyumu kurulamaması

## 14) Uygulamada Kullanım

Streamlit uygulaması `models/*_best.pt` ve `models/*_last.pt` dosyalarını otomatik keşfeder.

Ana dosya:

- `app.py`

Model seçim mantığı:

- Checkpoint dosyaları listelenir
- Model ailesi checkpoint anahtarlarından tespit edilir
- İlgili model sınıfı oluşturulur
- State dict yüklenir
- Kullanıcı sidebar üzerinden model seçer

Özel ensemble seçimi:

- `ensemble_best`

Uygulamadaki ensemble bileşenleri:

- `cnn_optimized`
- `cnn_baseline`
- `crnn_v2`

## 15) Yeniden Eğitim Komutları

Baseline:

```bash
python src/train_deam_cnn_va.py
```

CNN optimized:

```bash
python src/train_cnn_experiments.py --preset cnn_optimized --experiment-name cnn_optimized
```

CRNN v2:

```bash
python src/train_crnn_experiments.py --preset crnn_v2 --experiment-name crnn_v2
```

Transformer v2:

```bash
python src/train_crnn_experiments.py --preset transformer_v2 --experiment-name transformer_v2
```

CNN + Transformer:

```bash
python src/train_crnn_experiments.py --preset cnn_transformer_v1 --experiment-name cnn_transformer_v1
```

En iyi model seçimi:

```bash
python src/train_cnn_experiments.py --select-best-cnn
python src/train_crnn_experiments.py --select-best-crnn
```

## 16) Önerilen Sunum Sıralaması

Tez veya sunumda modeller şu sırayla anlatılabilir:

1. Problem: Müzikten Valence-Arousal tahmini
2. Veri: DEAM ve sabit split
3. Baseline: `cnn_baseline`
4. CNN deneyleri: learning rate, dropout, weight decay, augmentation, loss
5. En iyi tekil model: `cnn_optimized`
6. Temporal modelleme: CRNN ailesi
7. Attention tabanlı modelleme: Transformer ailesi
8. Hibrit model: CNN + Transformer
9. Ensemble: simple average, weighted, stacking
10. Genel sonuç: Ensemble modeller tekil modellerden daha iyi hata değerlerine ulaştı

## 17) Kritik Notlar

1. `cnn_v2_opt_adamw` eksik metrik

Bu deney `logs/experiment_log.csv` ve `configs/` içinde vardır; ancak `results/model_comparison.csv` içinde test sonucu bulunmamaktadır.

2. Loss ölçekleri

`smoothl1` gibi farklı loss kullanan deneylerde validation loss doğrudan MSE tabanlı loss değerleriyle karşılaştırılmamalıdır.

3. Best ve last checkpoint ayrımı

Sunum ve uygulama için genellikle `*_best.pt` dosyaları tercih edilmelidir. `*_last.pt`, eğitim sonundaki son ağırlıkları saklar ve overfitting sonrası performansı temsil edebilir.

4. Ensemble dikkat noktası

Stacking ensemble güçlü sonuçlar üretmiştir; ancak canlı uygulamada kullanılmadan önce genellenebilirlik ve veri sızıntısı riski ayrıca denetlenmelidir.

## 18) Kısa Sonuç

Bu model deneyleri sonucunda en güçlü tekil model `cnn_optimized` olmuştur. Bununla birlikte `cnn_baseline` ve `crnn_v2` modellerinin farklı metriklerde güçlü performans göstermesi, ensemble yaklaşımını anlamlı hale getirmiştir. Manual weighted ensemble, tekil modellerden daha düşük ortalama RMSE üretmiş ve uygulama içinde kullanılmaya uygun, sade ve açıklanabilir bir çözüm sunmuştur.
