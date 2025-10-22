# 🔌 Kısmi Boşalma (PD) Tespit Sistemi

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)](https://gradio.app/)
[![Gemini](https://img.shields.io/badge/Gemini-2.0%20Flash-green.svg)](https://ai.google.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Gemini Flash 2.0 destekli, yapay zeka tabanlı Kısmi Boşalma (Partial Discharge - PD) tespit ve analiz sistemi. Spektrum verilerini analiz eder, öğrenir ve interaktif chatbot ile sonuçları tartışmanıza olanak sağlar.

## ✨ Özellikler

- 🤖 **Gemini 2.0 Flash AI:** Google'ın en gelişmiş AI modeli ile analiz
- 📊 **NPY Dosya Desteği:** NumPy dizilerinden spektrum analizi
- 🧠 **Öğrenme Sistemi:** Her analizden öğrenen akıllı sistem
- 💬 **Chatbot:** Analiz sonuçlarını tartışabileceğiniz AI asistan
- 📈 **Görselleştirme:** Matplotlib ile detaylı spektrum grafikleri
- 📁 **CSV Export:** Analiz sonuçlarını CSV formatında kaydetme
- 🎯 **Doğruluk Takibi:** Etiketli verilerle model performans izleme

## 🚀 Hızlı Başlangıç

### 1. Gereksinimler

```bash
pip install google-generativeai gradio numpy pandas matplotlib python-dotenv
```

**Gerekli Paketler:**
- `google-generativeai>=0.3.0` - Gemini API
- `gradio>=4.0.0` - Web arayüzü
- `numpy>=1.21.0` - Veri işleme
- `pandas>=1.3.0` - Veri analizi
- `matplotlib>=3.4.0` - Görselleştirme
- `python-dotenv>=0.19.0` - Çevre değişkenleri

### 2. Gemini API Key Alma

1. [Google AI Studio](https://aistudio.google.com/apikey) adresine gidin
2. Google hesabınızla giriş yapın
3. **"Create API Key"** butonuna tıklayın
4. API anahtarınızı kopyalayın (ücretsiz!)

### 3. Proje Kurulumu

```bash
# Projeyi klonlayın
git clone https://github.com/kullaniciadi/pd-detection-system.git
cd pd-detection-system

# Gerekli paketleri yükleyin
pip install -r requirements.txt

# .env dosyası oluşturun
echo "GEMINI_API_KEY=AIza_sizin_api_key_buraya" > .env
```

**Windows PowerShell için:**
```powershell
echo "GEMINI_API_KEY=AIza_sizin_api_key_buraya" | Out-File -Encoding UTF8 .env
```

### 4. Klasör Yapısı Oluşturma

Projenizde şu klasör yapısını oluşturun:

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

### 5. Programı Çalıştırma

```bash
python pd_detection_demo.py
```

Tarayıcınızda otomatik olarak açılacak veya şu adresi ziyaret edin:
```
http://127.0.0.1:7860
```

## 📖 Kullanım Kılavuzu

### Adım 1: Sistemi Başlatma

1. **API Key Kontrolü:** Arayüz açıldığında, üst kısımda API key'in .env dosyasından yüklendiğini görmelisiniz
2. **Başlat Butonu:** 🚀 **"Sistemi Başlat"** butonuna tıklayın
3. **Durum Kontrolü:** "✅ Sistem başlatıldı!" mesajını göreceksiniz

### Adım 2: Chatbot ile Konuşma

**💬 Chatbot** sekmesine gidin ve AI asistanla konuşmaya başlayın:

**Örnek Sorular:**
```
- "PD nedir ve nasıl tespit edilir?"
- "Son analizlerim nasıl?"
- "Doğruluk oranım ne kadar?"
- "575454.npy dosyasının analizi nasıldı?"
- "Hangi parametreler PD göstergesidir?"
- "Sistem kaç dosya analiz etti?"
```

### Adım 3: Dosya Analizi

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

## 🌐 Hugging Face Spaces'te Yayınlama

### Hugging Face Token Gerekli mi?

**HAYIR!** Hugging Face Spaces'te sadece okuma işlemleri için token gerekmez. Ancak şu durumlarda gerekir:

- ✅ **Private space** oluşturuyorsanız
- ✅ **Model yükleme/kaydetme** yapacaksanız
- ✅ **Hugging Face Hub'a** dosya yazacaksanız

**Bu proje için token gerekmez** çünkü:
- Sadece Gemini API kullanıyor
- Lokal JSON dosyasına yazıyor
- Hugging Face Hub'a yazmıyor

### Hugging Face'e Yükleme Adımları

#### 1. Repository Oluşturma

```bash
# Hugging Face CLI yükleyin
pip install huggingface_hub

# Giriş yapın (opsiyonel, public space için)
huggingface-cli login
```

#### 2. Gerekli Dosyaları Hazırlama

**`requirements.txt` oluşturun:**
```txt
google-generativeai>=0.3.0
gradio>=4.0.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
python-dotenv>=0.19.0
```

**`app.py` oluşturun** (pd_detection_demo.py'yi kopyalayın):
```bash
cp pd_detection_demo.py app.py
```

**`README.md` ekleyin** (Hugging Face için):
```markdown
---
title: PD Detection System
emoji: 🔌
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---

# Kısmi Boşalma Tespit Sistemi

AI destekli PD tespit sistemi. Detaylı kullanım için GitHub'ı ziyaret edin.
```

#### 3. Space Oluşturma

1. [Hugging Face](https://huggingface.co/) hesabınıza giriş yapın
2. **Spaces** → **Create new Space**
3. **Space name:** `pd-detection-system`
4. **License:** MIT
5. **SDK:** Gradio
6. **Hardware:** CPU (ücretsiz)
7. **Create Space** butonuna tıklayın

#### 4. Dosyaları Yükleme

**Git ile:**
```bash
git clone https://huggingface.co/spaces/kullaniciadi/pd-detection-system
cd pd-detection-system

# Dosyaları kopyalayın
cp ../pd-detection-system/app.py .
cp ../pd-detection-system/requirements.txt .

# Commit ve push
git add .
git commit -m "Initial commit"
git push
```

**Web arayüzünden:**
- **Files** sekmesine gidin
- **Add file** → **Upload files**
- `app.py` ve `requirements.txt` dosyalarını sürükleyin
- **Commit** butonuna tıklayın

#### 5. Secrets Ayarlama (API Key)

**ÖNEMLİ:** API anahtarınızı Hugging Face Secrets'te saklayın:

1. Space sayfanızda **Settings** → **Variables and secrets**
2. **New secret** butonuna tıklayın
3. **Name:** `GEMINI_API_KEY`
4. **Value:** `AIza_sizin_api_key_buraya`
5. **Save** butonuna tıklayın

**Kodda otomatik okunur:**
```python
# .env yerine Hugging Face secrets'ten okur
GLOBAL_API_KEY = os.getenv('GEMINI_API_KEY')
```

#### 6. Demo Dosyalarını Ekleme (Opsiyonel)

Eğer örnek NPY dosyalarını da yüklemek isterseniz:

```bash
# Git LFS yükleyin (büyük dosyalar için)
git lfs install

# Dosyaları ekleyin
git lfs track "*.npy"
git add .gitattributes
git add npy_examples/
git add pd_values_for_training/
git commit -m "Add demo files"
git push
```

### 7. Space Hazır!

- Space'iniz otomatik olarak build olacak (2-3 dakika)
- **App** sekmesinde çalışan uygulamanızı göreceksiniz
- URL: `https://huggingface.co/spaces/kullaniciadi/pd-detection-system`

## 🔒 Güvenlik Notları

### .env Dosyası

**ASLA** `.env` dosyasını Git'e eklemeyin!

**`.gitignore` oluşturun:**
```gitignore
# API Keys
.env
*.env

# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/

# Öğrenme veritabanı
pd_learning_database_demo.json

# IDE
.vscode/
.idea/
```

### Hugging Face Secrets

- ✅ API anahtarlarını **Secrets** olarak saklayın
- ✅ Public space'lerde asla API key yazmayın
- ✅ Private space kullanmak için Hugging Face Pro gerekir

## 📊 Veri Formatı

### NPY Dosyası Formatı

```python
import numpy as np

# 1D NumPy dizisi
magnitude_data = np.array([
    -65.2, -64.8, -66.1, -65.5, ...
])

# Kaydetme
np.save('example.npy', magnitude_data)
```

**Gereksinimler:**
- ✅ 1 boyutlu NumPy dizisi
- ✅ Float değerler (dBm)
- ✅ Finite değerler (NaN/Inf olmamalı)
- ✅ Minimum 100 veri noktası önerilir

## 🎯 Örnek Senaryolar

### Senaryo 1: İlk Analiz

```
1. Sistemi başlat
2. Chatbot'a sor: "Bana PD tespiti hakkında bilgi ver"
3. Dosya analizi: npy_examples/pd/575454.npy
4. Etiket seç: "PD VAR"
5. Sonucu incele ve chatbot'a sor: "Bu analiz doğru mu?"
```

### Senaryo 2: Toplu Test

```python
# Toplu analiz için script
test_files = [
    ("npy_examples/pd/575454.npy", 1),
    ("npy_examples/pd/575923.npy", 1),
    ("npy_examples/no_pd/576415.npy", 0),
    ("npy_examples/no_pd/730183.npy", 0),
]

# Her dosyayı arayüzden analiz edin
# Sonra chatbot'a sorun: "Doğruluk oranım ne kadar?"
```

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Push edin (`git push origin feature/amazing`)
5. Pull Request açın

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 🙏 Teşekkürler

- [Google Gemini](https://ai.google.dev/) - AI modeli
- [Gradio](https://gradio.app/) - Web arayüzü
- [Hugging Face](https://huggingface.co/) - Hosting

## 📧 İletişim

- GitHub: [@kullaniciadi](https://github.com/kullaniciadi)
- Email: email@example.com

## 🐛 Sorun Bildirimi

Bir hata bulduysanız [Issues](https://github.com/kullaniciadi/pd-detection-system/issues) sayfasından bildirin.

---

⭐ Projeyi beğendiyseniz yıldız vermeyi unutmayın!