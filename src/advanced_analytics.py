"""
İleri Düzey Anket Analiz Fonksiyonları
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.survey_utils import survey_row_to_va


def _find_song_column(df):
    """Şarkı kolonunu kolon adından dinamik olarak bul."""
    def _score(col_name):
        low = str(col_name).casefold()
        score = 0
        if "dinlediniz" in low:
            score += 4
        if "hangi" in low:
            score += 2
        if "müzi" in low or "muz" in low:
            score += 2
        if "amaç" in low or "amac" in low:
            score -= 5
        return score

    candidates = []
    for col in df.columns:
        score = _score(col)
        if score <= 0:
            continue

        non_empty_count = int(
            df[col].astype(str).str.strip().replace("", np.nan).notna().sum()
        )
        candidates.append((score, non_empty_count, col))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][2]


def demographic_analysis(df):
    """
    Demografik analizler: yaş, cinsiyet, müzik dinleme sıklığı
    
    Args:
        df: Anket DataFrame
    
    Returns:
        dict: Analiz sonuçları
    """
    results = {}
    
    # DataFrame kopyasını al (orijinalini değiştirmemek için)
    df = df.copy()
    
    # Yaş kolonunu numeric'e çevir (string'leri temizle)
    df['Yaşınız'] = pd.to_numeric(df['Yaşınız'], errors='coerce')
    
    # Yaş grupları oluştur
    df['Yaş_Grubu'] = pd.cut(df['Yaşınız'], bins=[0, 20, 25, 30, 100], 
                             labels=['<20', '20-25', '25-30', '30+'])
    
    # Her satır için VA hesapla
    va_list = []
    for _, row in df.iterrows():
        try:
            v, a = survey_row_to_va(row)
            va_list.append({'Valence': v, 'Arousal': a})
        except:
            va_list.append({'Valence': np.nan, 'Arousal': np.nan})
    
    va_df = pd.DataFrame(va_list)
    df['Valence'] = va_df['Valence']
    df['Arousal'] = va_df['Arousal']
    
    # 1. Yaş gruplarına göre VA ortalamaları
    results['age_groups'] = df.groupby('Yaş_Grubu')[['Valence', 'Arousal']].mean()
    
    # 2. Cinsiyete göre VA ortalamaları
    results['gender'] = df.groupby('Cinsiyetiniz?')[['Valence', 'Arousal']].mean()
    
    # 3. Müzik dinleme sıklığına göre
    results['frequency'] = df.groupby('Müzik dinleme sıklığınız?')[['Valence', 'Arousal']].mean()
    
    # 4. Korelasyonlar
    emotion_cols = [
        'Bu müzik beni huzurlu ve sakin hissettirdi. ',
        'Bu müzik beni neşeli ve enerjik hissettirdi. ',
        'Bu müzik bana nostaljik bir his verdi. ',
        'Bu müzik beni üzgün veya melankolik hissettirdi. ',
        'Bu müzik beni gergin veya huzursuz hissettirdi. ',
        'Bu müzik beni güçlü, motive olmuş hissettirdi. '
    ]
    
    # Duygu yoğunluğu (tüm duyguların ortalaması)
    df['Duygu_Yoğunluğu'] = df[emotion_cols].mean(axis=1)
    
    results['frequency_intensity'] = df.groupby('Müzik dinleme sıklığınız?')['Duygu_Yoğunluğu'].mean()
    
    return results, df


def emotion_change_analysis(df):
    """
    Duygu durumu değişimi analizi: dinlemeden önce vs sonra
    
    Args:
        df: Anket DataFrame
    
    Returns:
        dict: Analiz sonuçları
    """
    results = {}
    
    # Önce-sonra karşılaştırması
    before_col = 'Müziği dinlemeden önceki duygu durumunuz'
    after_col = 'Müziği dinledikten sonraki duygu durumunuz.'
    
    results['before_distribution'] = df[before_col].value_counts()
    results['after_distribution'] = df[after_col].value_counts()
    
    # Değişim matrisi
    change_matrix = pd.crosstab(df[before_col], df[after_col], 
                                normalize='index', margins=True) * 100
    results['change_matrix'] = change_matrix
    
    # Şarkılara göre duygu değişimi
    emotion_cols = [
        'Bu müzik beni huzurlu ve sakin hissettirdi. ',
        'Bu müzik beni neşeli ve enerjik hissettirdi. ',
        'Bu müzik bana nostaljik bir his verdi. ',
        'Bu müzik beni üzgün veya melankolik hissettirdi. ',
        'Bu müzik beni gergin veya huzursuz hissettirdi. ',
        'Bu müzik beni güçlü, motive olmuş hissettirdi. '
    ]
    
    # Her şarkının ortalama duygu etkisi
    song_col = _find_song_column(df)
    if song_col is None:
        raise ValueError("Şarkı kolonu bulunamadı")

    results['song_emotion_impact'] = df.groupby(song_col)[emotion_cols].mean()
    
    # Etki skoru (derinden etkiledi)
    results['song_deep_impact'] = df.groupby(song_col)['Bu müzik beni derinden etkiledi.'].mean().sort_values(ascending=False)
    
    return results


def human_vs_ai_perception(df):
    """
    İnsan vs AI algısı analizi
    
    Args:
        df: Anket DataFrame
    
    Returns:
        dict: Analiz sonuçları
    """
    results = {}
    df = df.copy()
    
    # Kolon adlarını dinamik olarak bul (encoding sorunlarını önlemek için)
    human_col = None
    ai_col = None
    reason_col = None
    intensity_col = None
    
    for col in df.columns:
        if 'hangi duyguyu' in col.lower() and 'yansıt' in col.lower():
            human_col = col
        elif 'yapay' in col.lower() and 'etiket' in col.lower():
            ai_col = col
        elif 'farkl' in col.lower() and 'neden' in col.lower():
            reason_col = col
        elif 'yoğunlu' in col.lower():
            intensity_col = col
    
    # Kolonlar bulunamazsa hata ver
    if not human_col or not ai_col:
        raise ValueError(f"Gerekli kolonlar bulunamadı! İnsan: {human_col}, AI: {ai_col}")
    
    # İnsan ve AI etiketlerinin dağılımı
    results['human_labels'] = df[human_col].value_counts()
    results['ai_labels'] = df[ai_col].value_counts()
    
    # Yoğunluk dağılımı
    if intensity_col:
        results['intensity_distribution'] = df[intensity_col].value_counts().sort_index()
    
    # Uyuşma oranı
    df['Agreement'] = df[human_col] == df[ai_col]
    results['agreement_rate'] = df['Agreement'].mean() * 100
    
    # Uyuşmama nedenleri (serbest metin)
    if reason_col:
        disagreement_reasons = df[df['Agreement'] == False][reason_col].dropna()
        results['disagreement_reasons'] = disagreement_reasons.tolist()
        results['disagreement_count'] = len(disagreement_reasons)
    else:
        results['disagreement_reasons'] = []
        results['disagreement_count'] = 0
    
    # Şarkılara göre uyuşma oranı
    song_col = _find_song_column(df)
    if song_col is None:
        raise ValueError("Şarkı kolonu bulunamadı")

    results['song_agreement'] = df.groupby(song_col)['Agreement'].mean() * 100
    results['song_agreement'] = results['song_agreement'].sort_values(ascending=False)
    
    # Duygu yoğunluğuna göre uyuşma oranı
    if intensity_col:
        intensity_agreement = df.groupby(intensity_col)['Agreement'].mean() * 100
        results['intensity_agreement'] = intensity_agreement.sort_index()
    
    # Duygu türüne göre detaylı analiz
    if human_col and ai_col:
        emotion_comparison = pd.crosstab(df[human_col], df[ai_col], margins=False)
        results['emotion_comparison'] = emotion_comparison
    
    return results


def create_demographic_plots(results, df):
    """
    Demografik analiz görselleştirmeleri
    
    Returns:
        list: Matplotlib figure'ları
    """
    figures = []
    
    # 1. Yaş gruplarına göre VA
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    age_data = results['age_groups']
    x = range(len(age_data))
    
    ax1.bar(x, age_data['Valence'], alpha=0.7, color='skyblue', label='Valence')
    ax1.set_xlabel('Yaş Grubu')
    ax1.set_ylabel('Valence')
    ax1.set_title('Yaş Gruplarına Göre Valence')
    ax1.set_xticks(x)
    ax1.set_xticklabels(age_data.index)
    ax1.grid(axis='y', alpha=0.3)
    
    ax2.bar(x, age_data['Arousal'], alpha=0.7, color='coral', label='Arousal')
    ax2.set_xlabel('Yaş Grubu')
    ax2.set_ylabel('Arousal')
    ax2.set_title('Yaş Gruplarına Göre Arousal')
    ax2.set_xticks(x)
    ax2.set_xticklabels(age_data.index)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    figures.append(fig)
    
    # 2. Cinsiyete göre VA
    fig, ax = plt.subplots(figsize=(10, 6))
    
    gender_data = results['gender']
    x = np.arange(len(gender_data))
    width = 0.35
    
    ax.bar(x - width/2, gender_data['Valence'], width, label='Valence', alpha=0.8, color='skyblue')
    ax.bar(x + width/2, gender_data['Arousal'], width, label='Arousal', alpha=0.8, color='coral')
    
    ax.set_xlabel('Cinsiyet')
    ax.set_ylabel('Skor')
    ax.set_title('Cinsiyete Göre Valence-Arousal Karşılaştırması')
    ax.set_xticks(x)
    ax.set_xticklabels(gender_data.index)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    figures.append(fig)
    
    # 3. Müzik dinleme sıklığı vs Duygu yoğunluğu
    fig, ax = plt.subplots(figsize=(12, 6))
    
    freq_data = results['frequency_intensity'].sort_values(ascending=False)
    
    ax.barh(range(len(freq_data)), freq_data.values, alpha=0.7, color='mediumpurple')
    ax.set_yticks(range(len(freq_data)))
    ax.set_yticklabels(freq_data.index)
    ax.set_xlabel('Ortalama Duygu Yoğunluğu')
    ax.set_title('Müzik Dinleme Sıklığı vs Duygu Yoğunluğu')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    figures.append(fig)
    
    # 4. VA Space - Cinsiyet bazlı scatter
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for gender in df['Cinsiyetiniz?'].unique():
        gender_df = df[df['Cinsiyetiniz?'] == gender]
        ax.scatter(gender_df['Valence'], gender_df['Arousal'], 
                  alpha=0.5, s=50, label=gender)
    
    ax.set_xlim(1, 9)
    ax.set_ylim(1, 9)
    ax.set_xlabel('Valence', fontsize=12)
    ax.set_ylabel('Arousal', fontsize=12)
    ax.set_title('Valence-Arousal Uzayı (Cinsiyet Bazlı)', fontsize=14)
    ax.axhline(5, color='gray', linewidth=0.5, alpha=0.5)
    ax.axvline(5, color='gray', linewidth=0.5, alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    figures.append(fig)
    
    return figures


def create_emotion_change_plots(results, top_n=10):
    """
    Duygu değişimi görselleştirmeleri
    
    Returns:
        list: Matplotlib figure'ları
    """
    figures = []
    
    # 1. Önce-Sonra dağılımları
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    before = results['before_distribution']
    after = results['after_distribution']
    
    ax1.barh(range(len(before)), before.values, alpha=0.7, color='lightcoral')
    ax1.set_yticks(range(len(before)))
    ax1.set_yticklabels(before.index)
    ax1.set_xlabel('Kişi Sayısı')
    ax1.set_title('Müzik Dinlemeden Önceki Duygu Durumu')
    ax1.grid(axis='x', alpha=0.3)
    
    ax2.barh(range(len(after)), after.values, alpha=0.7, color='lightgreen')
    ax2.set_yticks(range(len(after)))
    ax2.set_yticklabels(after.index)
    ax2.set_xlabel('Kişi Sayısı')
    ax2.set_title('Müzik Dinledikten Sonraki Duygu Durumu')
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    figures.append(fig)
    
    # 2. En etkili şarkılar (derinden etkiledi skoru)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    top_impact = results['song_deep_impact'].head(top_n)
    
    ax.barh(range(len(top_impact)), top_impact.values, alpha=0.7, color='gold')
    ax.set_yticks(range(len(top_impact)))
    ax.set_yticklabels(top_impact.index)
    ax.set_xlabel('Ortalama Etki Skoru (1-5)')
    ax.set_title(f'En Etkili {len(top_impact)} Şarkı (Derinden Etkiledi)')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    figures.append(fig)
    
    return figures


def create_human_ai_plots(results):
    """
    İnsan vs AI algısı görselleştirmeleri
    
    Returns:
        list: Matplotlib figure'ları
    """
    figures = []
    
    # 1. İnsan vs AI etiket dağılımları
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    human = results['human_labels']
    ai = results['ai_labels']
    
    ax1.bar(range(len(human)), human.values, alpha=0.7, color='skyblue')
    ax1.set_xticks(range(len(human)))
    ax1.set_xticklabels(human.index, rotation=45, ha='right')
    ax1.set_ylabel('Sayı')
    ax1.set_title('İnsan Algısı - Duygu Etiketleri')
    ax1.grid(axis='y', alpha=0.3)
    
    ax2.bar(range(len(ai)), ai.values, alpha=0.7, color='coral')
    ax2.set_xticks(range(len(ai)))
    ax2.set_xticklabels(ai.index, rotation=45, ha='right')
    ax2.set_ylabel('Sayı')
    ax2.set_title('AI Algısı - Duygu Etiketleri')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    figures.append(fig)
    
    # 2. Duygu yoğunluğu dağılımı ve uyuşma
    if 'intensity_distribution' in results and 'intensity_agreement' in results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        intensity = results['intensity_distribution']
        ax1.bar(range(len(intensity)), intensity.values, alpha=0.7, color='mediumpurple')
        ax1.set_xticks(range(len(intensity)))
        ax1.set_xticklabels(intensity.index, rotation=45, ha='right')
        ax1.set_ylabel('Kişi Sayısı')
        ax1.set_title('Duygu Yoğunluğu Dağılımı')
        ax1.grid(axis='y', alpha=0.3)
        
        intensity_agree = results['intensity_agreement']
        colors = ['green' if x > 50 else 'orange' if x > 30 else 'red' for x in intensity_agree.values]
        ax2.bar(range(len(intensity_agree)), intensity_agree.values, alpha=0.7, color=colors)
        ax2.set_xticks(range(len(intensity_agree)))
        ax2.set_xticklabels(intensity_agree.index, rotation=45, ha='right')
        ax2.set_ylabel('Uyuşma Oranı (%)')
        ax2.set_title('Yoğunluğa Göre İnsan-AI Uyuşması')
        ax2.axhline(50, color='red', linestyle='--', alpha=0.5, label='%50 eşiği')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        figures.append(fig)
    
    # 3. Uyuşma oranı pasta grafiği
    fig, ax = plt.subplots(figsize=(8, 8))
    
    agreement_rate = results['agreement_rate']
    sizes = [agreement_rate, 100 - agreement_rate]
    labels = ['Uyuşuyor', 'Farklı']
    colors = ['lightgreen', 'lightcoral']
    explode = (0.1, 0)
    
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
           shadow=True, startangle=90, textprops={'fontsize': 14})
    ax.set_title(f'İnsan-AI Algısı Uyuşma Oranı\n(Toplam Uyuşma: %{agreement_rate:.1f})', 
                fontsize=14)
    
    plt.tight_layout()
    figures.append(fig)
    
    # 4lode = (0.1, 0)
    
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
           shadow=True, startangle=90)
    ax.set_title(f'İnsan-AI Algısı Uyuşma Oranı\n(Toplam Uyuşma: %{agreement_rate:.1f})', 
                fontsize=14)
    
    plt.tight_layout()
    figures.append(fig)
    
    # 3. Şarkılara göre uyuşma oranı
    fig, ax = plt.subplots(figsize=(12, 8))
    
    song_agreement = results['song_agreement'].head(15)
    
    colors = ['green' if x > 50 else 'orange' if x > 30 else 'red' for x in song_agreement.values]
    
    ax.barh(range(len(song_agreement)), song_agreement.values, alpha=0.7, color=colors)
    ax.set_yticks(range(len(song_agreement)))
    ax.set_yticklabels(song_agreement.index)
    ax.set_xlabel('Uyuşma Oranı (%)')
    ax.set_title('Şarkılara Göre İnsan-AI Uyuşma Oranı (En Yüksek 15)')
    ax.axvline(50, color='red', linestyle='--', alpha=0.5, label='%50 eşiği')
    ax.grid(axis='x', alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    figures.append(fig)
    
    return figures
