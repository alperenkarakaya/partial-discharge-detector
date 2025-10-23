# 🔌 Kısmi Boşalma (PD) Tespit Sistemi

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)](https://gradio.app/)
[![Gemini](https://img.shields.io/badge/Gemini-2.0%20Flash-green.svg)](https://ai.google.dev/)

(directory konumu yoluyla çalışan versiyon) https://huggingface.co/spaces/alperenkk/partial-discharge-detector

(belgeleri manuel eklemeniz gereken versiyon) https://huggingface.co/spaces/alperenkk/PartialDischargeDetector
### (belgeleri manuel eklemeniz gereken versiyon için) Belgeler pd veya no_pd olarak belgeler kısmına yüklenmiştir buradan alarak kodu deneyebilirsiniz. Öğrenim - Analiz farklı farklı tablara ayrılmıştır, çizilen grafikleri de inceleyebilirsiniz. 

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
🧩 Genel Amaç
Bu proje, elektrik sistemlerinde PD (Partial Discharge - Kısmi Boşalma) sinyallerini tespit etmek için geliştirilen bir analiz aracıdır.
Sinyaller .npy (NumPy array) veya .csv dosyaları halinde gelir ve sistem bunları Gemini Flash 2.0 yapay zekasıyla analiz eder.

Başlatma

Script’i çalıştırdığında (python pd_detection_demo.py), Gradio arayüzü açılır.
Buradan tek tek .npy dosyaları seçip analiz yapabilirsin.

🧠 Temel İş Akışı
1. Dosya Analizi (parse_npy_file)
.npy dosyasını okur (tek boyutlu bir sinyal dizisi olmalı).
Ortalama, standart sapma, pik, minimum, aralık, medyan gibi istatistikleri çıkarır.
Ani sıçramaları (|fark| > 10) sayar.
Dosya adında pd veya no_pd geçiyorsa, etiketi otomatik çıkarır.

2. Gemini ile Analiz (analyze_with_gemini_npy)
Yukarıdaki istatistikleri alıp Gemini modeline bir prompt gönderir.
Gemini şu formatta yanıt verir:

TAHMIN: 1
GUVEN: 83
ACIKLAMA: Sinyalde yüksek pikler ve ani sıçramalar gözlendi, PD var.
OGRENME_NOTU: PD verilerinde benzer pattern görüldü.

Sonuçlar saklanır (pd_learning_database_demo.json) ve “öğrenme geçmişi” oluşturulur.

3. Gradio Arayüzü
Arayüzde 2 ana sekme vardır:

📄 Tek Dosya Analizi: .npy dosyası girilir, analiz sonucu + grafik çıkar.
🧠 Öğrenme İstatistikleri: Doğruluk oranı ve son öğrenilen veriler listelenir.
📊 Görsel ve Çıktılar

Matplotlib grafiği: sinyalin spektrum şeklinde gösterimi.
CSV çıktısı: analiz istatistiklerinin kaydedilebileceği mini tablo.
Öğrenme özeti: geçmiş analizlerden öğrenilen örnekler.

💾 Öğrenme Mekanizması
Her çalıştırmada sonuçlar pd_learning_database_demo.json dosyasına eklenir.
Böylece sonraki analizlerde geçmiş sonuçlardan kısa bir “öğrenme bağlamı” eklenir.

🚀 Özetle
Bu sistem:
PD sinyallerini otomatik analiz eder,
Gemini’ye istatistikleri gönderip PD olup olmadığını tahmin ettirir,
Sonuçları kaydeder, görselleştirir ve zamanla “öğrenir”.

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


## **Local bilgisayarımda çalıştırabildiğim şekilde maalesef çalıştıramadım, belge yükleme mantığı olduğundan dolayı yapıyı tekrar kuramadım ve bozdum. Belge yüklemeli çalışan versiyonu bu şekilde, yine de belgeyi kendiniz indirip kurmak isterseniz farklı bir huggingface linki olarak buraya ekleyeceğim.** https://huggingface.co/spaces/alperenkk/PartialDischargeDetector

<img width="1764" height="903" alt="image" src="https://github.com/user-attachments/assets/7ab222b2-e6db-464a-b085-56cdb38f1a42" />
<img width="1649" height="885" alt="image" src="https://github.com/user-attachments/assets/f8ab09e4-460c-4eae-a669-6ed4371ee314" />
<img width="1747" height="865" alt="image" src="https://github.com/user-attachments/assets/6b909200-9bc6-4656-b8c4-66fb86e9a5bd" />
<img width="1843" height="886" alt="image" src="https://github.com/user-attachments/assets/4474bd92-3b12-451a-8ae4-06ff176445c8" />

**Bu linkten ona da ulaşabilirsiniz, belgeler kısmında örnekler yine mevcut. Teşekkürler**
https://huggingface.co/spaces/alperenkk/PartialDischargeDetector
