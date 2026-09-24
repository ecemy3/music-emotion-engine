"""
Anket verilerinden Valence-Arousal türetme fonksiyonları
"""

import numpy as np


def likert_to_01(x):
    """
    Likert ölçeğinden (1-5) 0-1 aralığına dönüştür
    
    Args:
        x: Likert değeri (1-5)
    
    Returns:
        float: 0-1 aralığında normalize değer
    """
    return (x - 1) / 4


def z_to_1_9(z):
    """
    Z-score'u 1-9 aralığına dönüştür
    
    Args:
        z: Z-score değeri
    
    Returns:
        float: 1-9 aralığında değer
    """
    return 5 + 4 * z


def survey_row_to_va(row):
    """
    Anket satırından Valence ve Arousal değerlerini hesapla
    
    Args:
        row: Pandas DataFrame satırı (anket cevapları)
    
    Returns:
        tuple: (valence, arousal) değerleri (1-9 aralığında)
    """
    # Likert ölçeğindeki cevapları 0-1 aralığına normalize et (kolon adlarının sonunda boşluk var!)
    calm = likert_to_01(row["Bu müzik beni huzurlu ve sakin hissettirdi. "])
    happy = likert_to_01(row["Bu müzik beni neşeli ve enerjik hissettirdi. "])
    nostalgic = likert_to_01(row["Bu müzik bana nostaljik bir his verdi. "])
    sad = likert_to_01(row["Bu müzik beni üzgün veya melankolik hissettirdi. "])
    admiration = likert_to_01(row["Bu müzik bana hayranlık ve şaşkınlık hisleri uyandırdı. "])
    tense = likert_to_01(row["Bu müzik beni gergin veya huzursuz hissettirdi. "])
    motivated = likert_to_01(row["Bu müzik beni güçlü, motive olmuş hissettirdi. "])

    # Pozitif ve negatif duyguların ortalaması
    pos = np.mean([happy, motivated, admiration])
    neg = np.mean([sad, tense])
    
    # Yüksek ve düşük enerjinin ortalaması
    high = np.mean([happy, tense, admiration, motivated])
    low = np.mean([calm, nostalgic])

    # Valence ve Arousal için z-score hesaplama
    zV = pos - neg  # Valence: pozitif - negatif
    zA = high - low  # Arousal: yüksek enerji - düşük enerji

    # Z-score'u 1-9 aralığına dönüştür
    return z_to_1_9(zV), z_to_1_9(zA)
