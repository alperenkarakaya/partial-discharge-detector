# 🔌 Kısmi Boşalma (PD) Tespit Sistemi

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)](https://gradio.app/)
[![Gemini](https://img.shields.io/badge/Gemini-2.0%20Flash-green.svg)](https://ai.google.dev/)

https://huggingface.co/spaces/alperenkk/partial-discharge-detector

Gemini Flash 2.0 destekli, yapay zeka tabanlı Kısmi Boşalma (Partial Discharge - PD) tespit ve analiz sistemi. Spektrum verilerini analiz eder, öğrenir ve interaktif chatbot ile sonuçları tartışmanıza olanak sağlar.

## ✨ Özellikler

- 🤖 **Gemini 2.0 Flash AI:** Google'ın en gelişmiş AI modeli ile analiz
- 📊 **NPY Dosya Desteği:** NumPy dizilerinden spektrum analizi
- 🧠 **Öğrenme Sistemi:** Her analizden öğrenen akıllı sistem
- 💬 **Chatbot:** Analiz sonuçlarını tartışabileceğiniz AI asistan
- 📈 **Görselleştirme:** Matplotlib ile detaylı spektrum grafikleri
- 📁 **CSV Export:** Analiz sonuçlarını CSV formatında kaydetme
- 🎯 **Doğruluk Takibi:** Etiketli verilerle model performans izleme


**Gerekli Paketler:**
- `google-generativeai>=0.3.0` - Gemini API
- `gradio>=4.0.0` - Web arayüzü
- `numpy>=1.21.0` - Veri işleme
- `pandas>=1.3.0` - Veri analizi
- `matplotlib>=3.4.0` - Görselleştirme
- `python-dotenv>=0.19.0` - Çevre değişkenleri

###  Gemini API Key Alma

1. [Google AI Studio](https://aistudio.google.com/apikey) adresine gidin
2. Google hesabınızla giriş yapın
3. **"Create API Key"** butonuna tıklayın
4. API anahtarınızı kopyalayın (ücretsiz!)

**Windows PowerShell için:**
```powershell
echo "GEMINI_API_KEY=AIza_sizin_api_key_buraya" | Out-File -Encoding UTF8 .env
```

### Klasör Yapısı

```
pd-detection-system/
├── .env                          # API anahtarları (GİZLİ!)
├── pd_detection_demo.py          # Ana program
├── requirements.txt              # Python paketleri
├── README.md                     # Bu dosya
├── .gitignore                    # Git ignore dosyası
│
├── pd_values_for_training/       # Eğitim dosyaları
│   ├── 500927.npy
│   ├── 500945.npy
│   ├── 501057.npy
│   └── 501084.npy
│
└── npy_examples/                 # Test dosyaları
    ├── pd/                       # PD içeren örnekler
    │   ├── 575454.npy
    │   └── 575923.npy
    └── no_pd/                    # Normal örnekler
        ├── 576415.npy
        ├── 730183.npy
        └── 730270.npy
```

## 📖 Kullanım Kılavuzu

### Sistemi Başlatma

1. Sistemi başlat düğmesine basarak sistemi başlatın.
2. Analiz edilmesini istediğiniz belgeyi (şu anlık belgenin yolunu ekleyin.)


### Dosya Analizi

**📄 Tek Dosya Analizi** sekmesine gidin:

1. **Dosya Yolu Girin:**
   ```
   npy_examples/pd/575454.npy
   ```

2. **Etiket Seçin** (opsiyonel):
   - `Bilinmiyor` - Sadece tahmin
   - `PD YOK` - Normal sinyal (eğitim için)
   - `PD VAR` - PD içeren sinyal (eğitim için)

3. **🔍 Analiz Et** butonuna tıklayın

4. **Sonuçları İnceleyin:**
   - 🎯 Tahmin (PD VAR/YOK)
   - 📈 Güven skoru (%)
   - 📊 İstatistikler (ortalama, std, pik)
   - 📈 Spektrum grafiği
   - 🧠 Öğrenilen bilgiler

### Adım 4: Öğrenme İstatistikleri

**🧠 Öğrenme İstatistikleri** sekmesinde:

- Toplam analiz sayısı
- Doğruluk oranı
- Son öğrenilen bilgiler
- Model performansı

**Gereksinimler:**
- ✅ 1 boyutlu NumPy dizisi
- ✅ Float değerler (dBm)
- ✅ Finite değerler (NaN/Inf olmamalı)
- ✅ Minimum 100 veri noktası önerilir
