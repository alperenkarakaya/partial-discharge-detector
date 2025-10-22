# ==============================================================================
# PROJE: ÖDEV 4 - NPY DOSYALARI İÇİN GEMINI ANALİZÖRÜ (Gelişmiş Arayüz)
# AÇIKLAMA: Kullanıcının yüklediği .npy, .csv, .bin dosyalarını
# analiz eder, PD konseptini öğrenir ve interaktif sohbet sunar.
# ==============================================================================

# ------------------------------------------------------------------------------
# BÖLÜM 1: KÜTÜPHANELERİN YÜKLENMESİ
# ------------------------------------------------------------------------------
import os
import io
import json
import numpy as np
import pandas as pd # !!! YENİ: CSV okumak için eklendi !!!
import google.generativeai as genai
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib # Matplotlib backend ayarı
matplotlib.use('Agg')
import gradio as gr
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Optional, Any

print("Kütüphaneler yüklendi.")

# ------------------------------------------------------------------------------
# BÖLÜM 2: API ve GEMINI MODELİNİN YAPILANDIRMASI
# ------------------------------------------------------------------------------
load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    # Bu kısmı Gradio arayüzünde girmek için hatayı kaldırıyoruz
    print("API Key .env dosyasında bulunamadı. Lütfen arayüzden girin.")
    # raise ValueError("API key gerekli! GEMINI_API_KEY env variable set edin.")

# Model konfigürasyonunu sisteme taşıdık
# genai.configure(api_key=api_key)
# gemini_model = genai.GenerativeModel('gemini-2.0-flash') 
# print("Gemini API ve model başarıyla yapılandırıldı.")

# ------------------------------------------------------------------------------
# BÖLÜM 3: PD TESPİT SİSTEMİ SINIFI (NPY için güncellendi)
# ------------------------------------------------------------------------------
class PDDetectionSystemNPY:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key: raise ValueError("API key gerekli!")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        self.learning_db_path = Path('pd_learning_database_npy.json')
        self.learning_data = self.load_learning_data()
        print("✓ PD Tespit Sistemi (NPY) başlatıldı")

    def load_learning_data(self) -> List[Dict]:
        if self.learning_db_path.exists():
            try:
                with open(self.learning_db_path, 'r', encoding='utf-8') as f: return json.load(f)
            except json.JSONDecodeError: return []
        return []

    def save_learning_data(self):
        self.learning_data = self.learning_data[-200:]
        with open(self.learning_db_path, 'w', encoding='utf-8') as f:
            json.dump(self.learning_data, f, indent=2, ensure_ascii=False)

    def parse_npy_file(self, file_path: str) -> Dict:
        """
        .npy dosyasını (sadece genlik verisi içerdiği varsayılan) parse et
        (GÜNCELLENDİ: .item() eklenerek JSON serializable hatası düzeltildi)
        """
        print(f"\n📄 Parsing NPY: {file_path}")
        try:
            mag_array = np.load(file_path)
            # Yüklenen verinin tek boyutlu ve sayısal olduğundan emin ol
            if mag_array.ndim != 1 or not np.issubdtype(mag_array.dtype, np.number):
                 raise ValueError(f"Beklenen formatta değil. Tek boyutlu sayısal dizi olmalı, shape={mag_array.shape}, dtype={mag_array.dtype}")

            # NaN veya Inf değerleri temizle
            mag_array = mag_array[np.isfinite(mag_array)]
            if len(mag_array) < 5: # İstatistik için minimum veri
                 raise ValueError("Temizleme sonrası yeterli veri kalmadı.")

            # İstatistikler (!!! DÜZELTME: .item() eklendi !!!)
            statistics = {
                'mean': np.mean(mag_array).item(), 
                'std': np.std(mag_array).item(),
                'peak': np.max(mag_array).item(), 
                'min': np.min(mag_array).item(),
                'range': np.ptp(mag_array).item(), 
                'median': np.median(mag_array).item()
            }

            # Ani sıçramalar (genlik ekseninde) (!!! DÜZELTME: .item() eklendi !!!)
            mag_diff = np.diff(mag_array)
            sudden_jumps = np.sum(np.abs(mag_diff) > 10).item() # 10 dB'den fazla ani değişim varsayımı

            print(f"   ✓ {len(mag_array)} veri noktası okundu")
            print(f"   ✓ Ort. Güç: {statistics['mean']:.2f} dBm, Std: {statistics['std']:.2f} dB, Pik: {statistics['peak']:.2f} dBm")

            # Dosya adından etiket tahmini
            file_name_lower = Path(file_path).name.lower()
            inferred_label = None
            if 'pd' in file_name_lower and 'no' not in file_name_lower:
                inferred_label = 1
                print("   ✓ Dosya adından PD etiketi çıkarıldı.")
            elif 'no_pd' in file_name_lower or 'normal' in file_name_lower or 'bg' in file_name_lower: # Arka plan gürültüsü
                inferred_label = 0
                print("   ✓ Dosya adından Normal (PD Yok) etiketi çıkarıldı.")
            else:
                 print("   ! Dosya adından etiket çıkarılamadı.")


            return {
                'file_path': file_path,
                'statistics': statistics,
                'data_points': len(mag_array),
                'magnitudes': mag_array, # NumPy array olarak sakla
                'sudden_jumps': int(sudden_jumps), # int() ile son bir güvence
                'inferred_label': inferred_label # Dosya adından çıkarılan etiket
            }
        except Exception as e:
            raise ValueError(f".npy dosyası parse edilirken hata: {e}")

    def create_analysis_prompt_npy(self, parsed_data: Dict, file_name: str, true_label: Optional[int] = None) -> str:
        """Gemini için NPY analiz promptu oluştur"""
        # Öğrenme context'i
        learning_context = ""
        if self.learning_data:
            recent_learning = self.learning_data[-10:] # Son 10 öğrenme
            learning_context = "\n\n=== ÖNCEKİ ÖĞRENME DENEYİMLERİ (Son 10) ===\n"
            for i, entry in enumerate(recent_learning, 1):
                label_text = 'PD VAR' if entry.get('true_label') == 1 else 'PD YOK' if entry.get('true_label') == 0 else 'Bilinmiyor'
                pred_text = 'PD VAR' if entry.get('prediction') == 1 else 'PD YOK' if entry.get('prediction') == 0 else 'Belirsiz'
                correct_text = '✓ DOĞRU' if entry.get('correct') else '✗ YANLIŞ' if entry.get('correct') is False else '-'
                confidence_text = f"{entry.get('confidence', 0):.1f}%"
                lesson_text = entry.get('lesson', '')
                learning_context += (
                    f"\n{i}. Dosya: {entry.get('file_name', 'Bilinmiyor')}\n"
                    f"   Gerçek: {label_text} | Tahmin: {pred_text} ({confidence_text}) | Sonuç: {correct_text}\n"
                    f"   Öğrenilen: {lesson_text}\n" if lesson_text and lesson_text != "YOK" else ""
                )
            learning_context += "\n⚠️ Bu deneyimlerden öğrendiklerini kullanarak tahmini yap!\n"


        # Gerçek etiket (kullanıcı girdisi veya dosya adından gelen)
        final_true_label = true_label if true_label is not None else parsed_data.get('inferred_label')
        true_label_info = ""
        if final_true_label is not None:
            true_label_info = f"\n\n🎯 GERÇEK ETİKET (Kullanıcı Girdisi/Dosya Adı): {'PD VAR (1)' if final_true_label == 1 else 'PD YOK (0)'}\n⚠️ Tahminini yaptıktan sonra bu etiketle karşılaştır ve detaylı öğrenme notu çıkar!"

        prompt = f"""Sen bir güç sistemleri uzmanı ve PD (Partial Discharge) tespit uzmanısın.

=== GENLİK VERİSİ ANALİZİ ===
📁 Dosya: {file_name}

📊 İSTATİSTİKSEL ÖZET (Sadece Genlik Verisi):
• Toplam Veri Noktası: {parsed_data['data_points']}
• Ortalama Güç: {parsed_data['statistics']['mean']:.2f} dBm
• Standart Sapma: {parsed_data['statistics']['std']:.2f} dB
• Pik Değer: {parsed_data['statistics']['peak']:.2f} dBm
• Minimum Değer: {parsed_data['statistics']['min']:.2f} dBm
• Dinamik Aralık: {parsed_data['statistics']['range']:.2f} dB
• Median: {parsed_data['statistics']['median']:.2f} dBm
• Ani Sıçrama Sayısı (>10dB): {parsed_data['sudden_jumps']}

⚠️ NOT: Frekans bilgisi mevcut değil. Analizini sadece genlik verisinin istatistiksel dağılımına göre yapmalısın. Yüksek standart sapma, yüksek pik değer ve çok sayıda ani sıçrama, genellikle PD sinyallerinin zaman domenindeki keskin ve ani yapısını yansıtır, bu da dolaylı olarak yüksek frekans içeriğine işaret edebilir.
{learning_context}{true_label_info}

=== GÖREVİN ===
1. Verilen genlik istatistiklerini analiz et.
2. PD karakteristiklerini (yüksek pik, yüksek standart sapma, ani sıçramalar) ara.
3. PD VAR (1) veya PD YOK (0) tahmini yap.
4. 0-100 arası güven skoru belirle.
5. Teknik açıklama yaz (2-3 cümle).
6. Eğer gerçek etiket belirtilmişse, detaylı öğrenme notu çıkar.

⚠️ CEVAP FORMATI (TAM OLARAK BU SATIRLARLA):
TAHMIN: [sadece 0 veya 1]
GUVEN: [sadece sayı 0-100]
ACIKLAMA: [2-3 cümle teknik analiz]
OGRENME_NOTU: [eğer gerçek etiket varsa detaylı not, yoksa YOK]
"""
        return prompt

    def analyze_with_gemini_npy(self, parsed_data: Dict, file_name: str, true_label: Optional[int] = None) -> Dict:
        """Gemini ile NPY verisi analizi"""
        final_true_label = true_label if true_label is not None else parsed_data.get('inferred_label')
        prompt = self.create_analysis_prompt_npy(parsed_data, file_name, final_true_label)
        print(f"\n🤖 Gemini NPY analizi yapıyor...")
        try:
            response = self.model.generate_content(prompt, generation_config={'temperature': 0.2})
            result_text = response.text
            print(f"   ✓ Yanıt alındı.")
            
            prediction, confidence, explanation, lesson = None, 0.0, "Analiz yapılamadı.", "YOK"
            for line in result_text.split('\n'):
                 line = line.strip()
                 if line.startswith('TAHMIN:'):
                     try: prediction = int(line.split(':')[1].strip())
                     except: pass
                 elif line.startswith('GUVEN:'):
                     try: confidence = float(line.split(':')[1].strip())
                     except: pass
                 elif line.startswith('ACIKLAMA:'):
                     explanation = line.split(':', 1)[1].strip()
                 elif line.startswith('OGRENME_NOTU:'):
                     lesson_text = line.split(':', 1)[1].strip()
                     if lesson_text and lesson_text != "YOK": lesson = lesson_text

            correct = None
            if final_true_label is not None and prediction is not None:
                correct = (prediction == final_true_label)
            
            result = {
                'file_name': file_name, 'prediction': prediction, 'confidence': confidence,
                'true_label': final_true_label, # Kullanıcı veya dosya adından gelen etiket
                'correct': correct, 'explanation': explanation, 'lesson': lesson,
                'statistics': parsed_data['statistics'],
                'timestamp': datetime.now().isoformat()
            }
            pred_text = "PD VAR ⚠️" if prediction == 1 else "PD YOK ✓" if prediction == 0 else "BELİRSİZ ?"
            print(f"\n   📊 Tahmin: {pred_text} (Güven: {confidence:.1f}%)")
            if final_true_label is not None: print(f"   {'✓ DOĞRU' if correct else '✗ YANLIŞ'} (Gerçek: {'PD VAR' if final_true_label == 1 else 'PD YOK'})")
            print(f"   💭 {explanation}")
            if lesson: print(f"   🧠 Öğrenilen: {lesson}")

            self.learning_data.append(result)
            self.save_learning_data()
            return result
        except Exception as e:
            print(f"   ✗ Gemini Hatası: {e}")
            return {'file_name': file_name, 'error': str(e), 'timestamp': datetime.now().isoformat()}

    def analyze_npy_file(self, file_path: str, true_label: Optional[int] = None) -> Dict:
        """Tek NPY dosyası analizi"""
        try:
            parsed_data = self.parse_npy_file(file_path)
            file_name = Path(file_path).name
            result = self.analyze_with_gemini_npy(parsed_data, file_name, true_label)
            # Grafik verisi
            result['plot_data'] = {'magnitudes': parsed_data['magnitudes']} # Listeye çevirmeden array olarak bırak
            return result
        except Exception as e:
            return {'file_name': Path(file_path).name, 'error': str(e), 'timestamp': datetime.now().isoformat()}

    def analyze_npy_batch(self, file_paths: List[str], labels: Optional[List[int]] = None) -> List[Dict]:
        """Toplu NPY dosyası analizi"""
        results = []
        labels = labels or ([None] * len(file_paths))
        print(f"\n{'='*60}\n🚀 TOPLU NPY ANALİZİ BAŞLIYOR: {len(file_paths)} dosya\n{'='*60}")
        for i, (file_path, label) in enumerate(zip(file_paths, labels), 1):
            print(f"\n[{i}/{len(file_paths)}] İşleniyor...")
            results.append(self.analyze_npy_file(file_path, label))
        return results

    def generate_report_text(self, results: List[Dict]) -> str:
        # Önceki generate_report fonksiyonunun metin çıktısını üreten versiyonu
        output = "=" * 80 + "\nKISMI BOŞALMA (PD) TESPİT RAPORU (.npy)\n" + "=" * 80 + "\n"
        valid_results = [r for r in results if 'error' not in r and r.get('prediction') is not None]
        pd_detected = sum(1 for r in valid_results if r['prediction'] == 1)
        no_pd = sum(1 for r in valid_results if r['prediction'] == 0)
        avg_confidence = np.mean([r['confidence'] for r in valid_results if 'confidence' in r]) if valid_results else 0
        labeled_results = [r for r in valid_results if r.get('true_label') is not None]
        correct_predictions = sum(1 for r in labeled_results if r.get('correct'))
        accuracy = (correct_predictions / len(labeled_results) * 100) if labeled_results else None
        output += "ÖZET İSTATİSTİKLER\n" + "-" * 80 + "\n"
        output += f"PD Tespit Edildi: {pd_detected}\nPD Tespit Edilmedi: {no_pd}\nOrtalama Güven: {avg_confidence:.2f}%\n"
        if accuracy is not None: output += f"Doğruluk Oranı: {accuracy:.2f}% ({correct_predictions}/{len(labeled_results)})\n"
        output += "\nDETAYLI SONUÇLAR\n" + "-" * 80 + "\n\n"

        for i, result in enumerate(results, 1):
            output += f"{i}. {result['file_name']}\n"
            if 'error' in result: output += f"   HATA: {result['error']}\n\n"; continue
            pred_text = "PD VAR" if result.get('prediction') == 1 else "PD YOK" if result.get('prediction') == 0 else "Belirsiz"
            conf_text = f"{result.get('confidence', 0):.2f}%"
            output += f"   Tahmin: {pred_text} ({conf_text})\n"
            if result.get('true_label') is not None:
                 true_text = "PD VAR" if result['true_label'] == 1 else "PD YOK"
                 status = "DOĞRU ✓" if result.get('correct') else "YANLIŞ ✗"
                 output += f"   Gerçek Etiket: {true_text}\n   Sonuç: {status}\n"
            output += f"   Açıklama: {result.get('explanation', '-')}\n"
            if result.get('lesson') and result['lesson'] != "YOK": output += f"   Öğrenilen: {result['lesson']}\n"
            stats = result.get('statistics', {})
            output += f"\n   İstatistikler:\n"
            output += f"     - Ort: {stats.get('mean', np.nan):.2f}, Std: {stats.get('std', np.nan):.2f}, Pik: {stats.get('peak', np.nan):.2f}\n\n"
        print(f"\n✓ Metin Raporu oluşturuldu.")
        return output

    def generate_csv_data(self, results: List[Dict]) -> str:
        header = "Dosya,Tahmin,Güven (%),Gerçek Etiket,Doğru,Ortalama Güç (dBm),Std Sapma (dB),Pik (dBm),Dinamik Aralık (dB),Açıklama\n"
        csv_data = header
        for r in results:
             if 'error' not in r and r.get('prediction') is not None:
                 pred = 'PD VAR' if r['prediction'] == 1 else 'PD YOK'
                 conf = f"{r.get('confidence', 0):.2f}"
                 true = 'PD VAR' if r.get('true_label') == 1 else 'PD YOK' if r.get('true_label') == 0 else '-'
                 correct = 'EVET' if r.get('correct') else 'HAYIR' if r.get('correct') is False else '-'
                 stats = r.get('statistics', {})
                 mean = f"{stats.get('mean', np.nan):.2f}"
                 std = f"{stats.get('std', np.nan):.2f}"
                 peak = f"{stats.get('peak', np.nan):.2f}"
                 range_ = f"{stats.get('range', np.nan):.2f}"
                 expl = f"\"{r.get('explanation', '').replace('\"', '\"\"')}\""
                 csv_data += f"{r['file_name']},{pred},{conf},{true},{correct},{mean},{std},{peak},{range_},{expl}\n"
        print(f"✓ CSV verisi oluşturuldu.")
        return csv_data

# ------------------------------------------------------------------------------
# BÖLÜM 5: GRADIO ARAYÜZÜ FONKSİYONLARI (!!! YENİ FONKSİYONLAR EKLENDİ !!!)
# ------------------------------------------------------------------------------
def initialize_system_gr(api_key):
    """Gradio için sistemi başlatan ana fonksiyon"""
    if not api_key: return "❌ Lütfen API anahtarını girin!", None
    try:
        system_instance = PDDetectionSystemNPY(api_key=api_key)
        count = len(system_instance.learning_data)
        return f"✅ Sistem (NPY) başlatıldı! Öğrenme Veritabanı: {count} kayıt", system_instance
    except Exception as e: return f"❌ Sistem başlatılamadı: {str(e)}", None

def analyze_single_file_gr(file, true_label_str, system_state):
    """Gradio için tek dosya analizi (Veri Önizleme Eklendi)"""
    if system_state is None:
        return "❌ Sistem başlatılmadı.", None, None, "Veri yüklenmedi."
    if file is None:
        return "❌ Lütfen bir .npy dosyası yükleyin!", None, None, "Veri yüklenmedi."

    data_preview = "Veri okunamadı."
    try:
        true_label = None
        if true_label_str and true_label_str != "Bilinmiyor":
            true_label = 1 if true_label_str == "PD VAR" else 0

        result = system_state.analyze_npy_file(file.name, true_label=true_label) 

        if 'error' in result:
            return f"❌ Analiz Hatası: {result['error']}", None, None, f"Hata: {result['error']}"

        # Rapor Metni
        output = "=" * 60 + "\n📊 NPY ANALİZ SONUCU\n" + "=" * 60 + "\n\n"
        pred_emoji = "🔴" if result.get('prediction') == 1 else "🟢"
        pred_text = "PD VAR" if result.get('prediction') == 1 else "PD YOK" if result.get('prediction') == 0 else "Belirsiz"
        output += f"🎯 TAHMİN: {pred_emoji} {pred_text}\n"
        output += f"📈 GÜVEN: {result.get('confidence', 0):.2f}%\n\n"
        if result.get('true_label') is not None:
             true_text = "PD VAR" if result['true_label'] == 1 else "PD YOK"
             status = "✅ DOĞRU" if result.get('correct') else "❌ YANLIŞ"
             output += f"🏷️  GERÇEK ETİKET: {true_text}\n📊 SONUCU: {status}\n\n"
        output += f"💭 AÇIKLAMA:\n{result.get('explanation', '-')}\n\n"
        if result.get('lesson') and result['lesson'] != "YOK": output += f"🧠 ÖĞRENİLEN:\n{result['lesson']}\n\n"
        output += "-" * 60 + "\n📈 İSTATİSTİKLER (Genlik)\n" + "-" * 60 + "\n" + "\n".join([f"{k}: {v:.2f}" for k, v in result.get('statistics', {}).items()]) + "\n\n"

        # Grafik
        fig, ax = plt.subplots(figsize=(10, 4))
        plot_data = result.get('plot_data', {})
        mags = plot_data.get('magnitudes') # Bu artık bir numpy array

        if isinstance(mags, np.ndarray) and mags.ndim == 1 and len(mags) > 0:
            ax.plot(mags, 'b-', linewidth=0.8, alpha=0.7)
            ax.set_xlabel('Örnek Numarası (Index)')
            ax.set_ylabel('Güç (dBm)')
            ax.set_title(f'Genlik Verisi - Tahmin: {pred_text}')
            ax.grid(True, alpha=0.3)
            stats = result.get('statistics', {})
            stats_text = f"Ort: {stats.get('mean', np.nan):.1f}\nStd: {stats.get('std', np.nan):.1f}\nPik: {stats.get('peak', np.nan):.1f}"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, va='top', bbox=dict(boxstyle='round', fc='lightblue', alpha=0.7))
        plt.tight_layout()

        # CSV verisi
        csv_data = system_state.generate_csv_data([result])

        # Veri Önizleme
        if isinstance(mags, np.ndarray) and mags.ndim == 1 and len(mags) > 0:
            preview_length = min(200, len(mags))
            data_preview_list = [f"{val:.4f}" for val in mags[:preview_length]]
            data_preview = " ".join(data_preview_list)
            if len(mags) > preview_length:
                data_preview += " ..."
        else:
            data_preview = "Grafik çizdirilemedi veya geçerli veri yok."

        return output, fig, csv_data, data_preview

    except Exception as e:
        plt.close()
        return f"❌ Hata: {str(e)}", None, None, f"Hata oluştu: {str(e)}"
    
def analyze_batch_files_gr(files, system_state):
    """Gradio için toplu NPY analizi"""
    if system_state is None: return "❌ Sistem başlatılmadı.", None, None
    if not files: return "❌ Lütfen analiz için .npy dosyalarını yükleyin!", None, None
    try:
        file_paths = [f.name for f in files]
        results = system_state.analyze_npy_batch(file_paths, labels=None)
        output = system_state.generate_report_text(results)

        # Özet Grafik
        fig = None
        valid_results = [r for r in results if 'error' not in r and r.get('prediction') is not None]
        if valid_results:
             fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
             pd_count = sum(1 for r in valid_results if r['prediction'] == 1)
             no_pd_count = sum(1 for r in valid_results if r['prediction'] == 0)
             labels_pie = ['PD VAR', 'PD YOK']
             sizes = [pd_count, no_pd_count]
             colors_pie = ['#ff6b6b', '#51cf66']
             ax1.pie(sizes, labels=labels_pie, colors=colors_pie, autopct='%1.1f%%', startangle=90)
             ax1.set_title('Tahmin Dağılımı (.npy)')

             file_names = [r['file_name'][:15] + ('...' if len(r['file_name'])>15 else '') for r in valid_results]
             confidences = [r['confidence'] for r in valid_results]
             bar_colors = ['#ff6b6b' if r['prediction'] == 1 else '#51cf66' for r in valid_results]
             ax2.barh(file_names, confidences, color=bar_colors, alpha=0.7)
             ax2.set_xlabel('Güven Skoru (%)')
             ax2.set_title('Dosya Bazlı Güven Skorları (.npy)')
             ax2.grid(True, alpha=0.3, axis='x')
             plt.tight_layout()
        else: plt.close()

        csv_data = system_state.generate_csv_data(results)
        return output, fig, csv_data
    except Exception as e:
        plt.close(); return f"❌ Toplu Analiz Hatası: {str(e)}", None, None

def get_learning_stats_gr(system_state):
    """Gradio için öğrenme istatistiklerini getir"""
    if system_state is None: return "❌ Sistem başlatılmadı."
    if not system_state.learning_data: return "📊 Henüz öğrenme verisi yok."
    total = len(system_state.learning_data)
    labeled = [d for d in system_state.learning_data if d.get('true_label') is not None]
    correct = [d for d in labeled if d.get('correct') == True]
    accuracy = (len(correct) / len(labeled) * 100) if labeled else 0
    output = f"📊 Toplam Analiz: {total}\n🏷️ Etiketli Veri: {len(labeled)}\n✅ Doğru Tahmin: {len(correct)}\n❌ Yanlış Tahmin: {len(labeled) - len(correct)}\n📈 Doğruluk Oranı: {accuracy:.2f}%\n\n"
    recent_with_lessons = [d for d in system_state.learning_data[-10:] if d.get('lesson') and d['lesson'] != "YOK"][-5:]
    if recent_with_lessons:
         output += "📚 SON 5 ÖĞRENİLEN DERS\n" + "-" * 60 + "\n"
         for i, d in enumerate(recent_with_lessons, 1):
             output += f"\n{i}. {d['file_name']}\n   {d['lesson']}\n"
    return output


def reset_learning_db_gr(system_state):
    """Gradio için öğrenme veritabanını sıfırla"""
    if system_state is None: return "❌ Sistem başlatılmadı!"
    try:
        system_state.learning_data = []
        system_state.save_learning_data()
        return "✅ Öğrenme veritabanı sıfırlandı!"
    except Exception as e: return f"❌ Hata: {str(e)}"

# ------------------------------------------------------------------------------
# !!! BÖLÜM 5.5: İSTEKLERİNİZ İÇİN YENİ FONKSİYONLAR (GÜNCELLENDİ) !!!
# ------------------------------------------------------------------------------

def learn_pd_concept_from_npy(files, system_state, progress=gr.Progress()):
    # (Bu fonksiyon değişmedi)
    """
    (YENİ - TAB 2) Yüklenen .npy dosyalarından PD konseptini öğrenmek için Gemini'ye sorar.
    """
    if system_state is None: return "❌ Sistem başlatılmadı.", [], gr.update(visible=False)
    if not files: return "❌ Lütfen en az bir .npy dosyası yükleyin.", [], gr.update(visible=False)
    all_stats = []
    file_names = []
    progress(0, desc="Dosyalar okunuyor...")
    for i, file in enumerate(files):
        try:
            progress(i / len(files), desc=f"Okunuyor: {Path(file.name).name}")
            parsed_data = system_state.parse_npy_file(file.name)
            stats = parsed_data['statistics']
            all_stats.append({
                'file': Path(file.name).name, 'mean': stats['mean'],
                'std': stats['std'], 'peak': stats['peak'],
                'sudden_jumps': parsed_data['sudden_jumps']
            })
            file_names.append(Path(file.name).name)
        except Exception as e:
            print(f"Dosya okuma hatası: {e}")
            continue
    if not all_stats:
        return "❌ Geçerli .npy dosyası okunamadı.", [], gr.update(visible=False)
    progress(1, desc="Gemini düşünüyor...")
    stats_summary = json.dumps(all_stats, indent=2, ensure_ascii=False)
    prompt = f"""Sen bir güç sistemleri ve sinyal işleme uzmanısın.
Sana {len(all_stats)} adet farklı sinyal dosyasından (.npy) çıkarılan istatistiksel özetleri veriyorum. 
Bu dosyalar Kısmi Boşalma (PD) tespitiyle ilgilidir.
VERİ ÖZETİ:
{stats_summary}
GÖREVİN:
Bu istatistiklere dayanarak, Kısmi Boşalma (PD) fenomenini detaylıca açıkla.
Açıklaman aşağıdaki soruları mutlaka cevaplamalı:
1.  **PD Nedir?**
2.  **Sinyal Karakteristiği Nedir?**
3.  **Nasıl Tespit Edilir?**
Lütfen cevabını net, eğitici ve teknik bir dille yaz.
"""
    try:
        response = system_state.model.generate_content(prompt)
        explanation = response.text
        chat_history = [[
            None, 
            f"Merhaba! {len(file_names)} adet dosyayı analiz ettim ve PD hakkında bir özet çıkardım. Bu bilgilere dayanarak sorularınızı yanıtlayabilirim.\n\n{explanation}"
        ]]
        return explanation, chat_history, gr.update(visible=True)
    except Exception as e:
        return f"❌ Gemini ile iletişim hatası: {e}", [], gr.update(visible=False)


# !!! YENİ YARDIMCI FONKSİYON !!!
def get_confidence_label(confidence: float) -> gr.Label:
    """Güven skorunu alır ve ona göre renkli bir Gradio Label döndürür."""
    if confidence is None:
        confidence = 0.0
        
    value = f"{confidence:.1f}%"
    
    if confidence >= 80:
        # Yeşil (Emin)
        return gr.Label(value=value, label="Güven Skoru (Yüksek)", color="#51cf66")
    elif confidence >= 40:
        # Sarı (Orta)
        return gr.Label(value=value, label="Güven Skoru (Orta)", color="#fcc419")
    else:
        # Kırmızı (Düşük)
        return gr.Label(value=value, label="Güven Skoru (Düşük)", color="#ff6b6b")


def process_and_visualize_other_formats(file, csv_col_name, bin_dtype, system_state):
    """
    (YENİ - TAB 3) .csv veya .bin dosyalarını okur, sinyal grafiği çizer ve temel analiz yapar.
    (GÜNCELLENDİ: Artık 'TAHMIN' ve 'GUVEN' skoru istiyor ve renkli skala döndürüyor)
    """
    if system_state is None: 
        return None, "❌ Sistem başlatılmadı.", None, gr.update(visible=False), "N/A", gr.Label(value="0%", label="Güven Skoru")
    if file is None: 
        return None, "❌ Lütfen bir dosya yükleyin.", None, gr.update(visible=False), "N/A", gr.Label(value="0%", label="Güven Skoru")

    file_path = file.name
    file_name = Path(file_path).name
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Başlangıç değerleri
    analysis_text = "Analiz yapılamadı."
    chat_context = None 
    image_bytes = None
    prediction = None
    confidence = 0.0
    explanation = "N/A"
    
    # Başarısızlık durumunda döndürülecek varsayılan değerler
    default_confidence_label = gr.Label(value="0%", label="Güven Skoru", color="#ff6b6b")
    default_return = (None, "Hata oluştu.", None, gr.update(visible=False), "HATA", default_confidence_label)

    try:
        print(f"İşleniyor: {file_name}")

        if file_name.endswith('.csv'):
            
            # --- METİN TABANLI Başlık (Header) Satırını Bulma Mantığı ---
            header_row_index = None
            key_col_1_search = 'freq. [hz]'
            key_col_2_search = 'magnitude [dbm]'
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines() 
                    for i, line in enumerate(lines):
                        if i > 100: break
                        line_lower = line.lower()
                        if key_col_1_search in line_lower and key_col_2_search in line_lower:
                            header_row_index = i
                            print(f"   ✓ Başlık satırı {i}. indekste (satır {i+1}) bulundu.")
                            break
            except Exception as preview_e:
                raise ValueError(f"Dosya metin olarak okunurken hata: {preview_e}")

            if header_row_index is None:
                raise ValueError(f"Dosyada '{key_col_1_search}' ve '{key_col_2_search}' sütunlarını içeren başlık satırı ilk 100 satırda bulunamadı.")
            # --- Başlık Bulma Mantığı Sonu ---

            # Veriyi, bulunan başlık satırına ve özel formata göre oku
            try:
                df = pd.read_csv(
                    file_path, 
                    sep=';', header=header_row_index, decimal=',',
                    thousands='.', on_bad_lines='skip', encoding_errors='ignore'
                )
            except Exception as read_e:
                raise ValueError(f"Pandas okuma hatası: {read_e}")
            
            if df.empty: raise ValueError("CSV dosyası okundu ancak asıl veri bloğu boş.")

            df.columns = df.columns.str.strip().str.lower()
            print(f"   ✓ Temizlenmiş Sütun Adları: {list(df.columns)}")

            # --- 2D SÜTUN ARAMA MANTIĞI ---
            mag_col = 'magnitude [dbm]'
            freq_col = 'freq. [hz]'
            
            if not (mag_col in df.columns and freq_col in df.columns):
                 raise ValueError(f"Hata: Gerekli 2D spektrum sütunları ('{mag_col}' ve '{freq_col}') temizlenmiş sütunlarda bulunamadı.")
            
            print(f"   🚀 2D Spektrum Analizi başlatılıyor (X='{freq_col}', Y='{mag_col}').")
            
            x_data_raw = pd.to_numeric(df[freq_col], errors='coerce')
            y_data_raw = pd.to_numeric(df[mag_col], errors='coerce')
            valid_mask = x_data_raw.notna() & y_data_raw.notna()
            x_data = x_data_raw[valid_mask].values
            y_data = y_data_raw[valid_mask].values

            if len(x_data) < 5:
                raise ValueError(f"'{freq_col}' ve '{mag_col}' sütunlarında yeterli (en az 5 adet) eşleşen sayısal veri bulunamadı.")
            
            print(f"   ✓ {len(x_data)} adet geçerli 2D veri noktası bulundu.")

            # 2D Grafik (X vs Y)
            ax.plot(x_data, y_data, 'r-', linewidth=0.7, alpha=0.8)
            ax.set_xlabel(f"Frekans ({freq_col})")
            ax.set_ylabel(f"Genlik ({mag_col})")
            ax.set_title(f"Spektrum Analizi - {file_name}")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            stats = {
                'peak_magnitude': np.max(y_data).item(),
                'mean_magnitude': np.mean(y_data).item(),
                'peak_frequency': x_data[np.argmax(y_data)].item()
            }
            chat_context = {'file': file_name, 'type': '2D Spektrum', 'x_col': freq_col, 'y_col': mag_col, 'stats': stats}

            # Grafiği belleğe kaydet (PNG formatında)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            image_bytes = buf.getvalue()

            # --- !!! YENİ: Gemini için Yapılandırılmış Prompt (2D) !!! ---
            prompt_parts = [
                {"mime_type": "image/png", "data": image_bytes},
                f"Sen bir PD (Kısmi Boşalma) uzmanısın. Yukarıdaki 2D Spektrum grafiğini ('{freq_col}' vs '{mag_col}') ve istatistikleri analiz et.",
                f"İSTATİSTİKLER:\n{json.dumps(stats, indent=2)}\n\n",
                f"GÖREV: Bu verilere (özellikle yüksek frekanstaki yüksek genliklere) dayanarak, bir PD sinyali olup olmadığını belirle.",
                f"PD VAR (1) veya PD YOK (0) tahmini yap ve güven skorunu (0-100) belirle.",
                f"Teknik bir açıklama yaz.",
                f"\n⚠️ CEVAP FORMATI (TAM OLARAK BU SATIRLARLA):",
                f"TAHMIN: [sadece 0 veya 1]",
                f"GUVEN: [sadece sayı 0-100]",
                f"ACIKLAMA: [2-3 cümle teknik analiz]"
            ]
                
        elif file_name.endswith('.bin'):
            # (BIN okuma mantığı değişmedi, bu her zaman 1D'dir)
            np_dtype = np.dtype(bin_dtype)
            signal_data = np.fromfile(file_path, dtype=np_dtype)
            target_col = f"BIN Verisi ({bin_dtype})" 

            if signal_data is None or len(signal_data) == 0:
                raise ValueError(f"'{target_col}' sütunundan sinyal verisi çıkarılamadı.")

            print(f"   ✓ {len(signal_data)} veri noktası okundu.")
            
            # 1D Grafik
            ax.plot(signal_data, 'r-', linewidth=0.7, alpha=0.8)
            ax.set_xlabel('Örnek Numarası (Index)')
            ax.set_ylabel(f"Değer ({target_col})")
            ax.set_title(f"1D Sinyal Grafiği: {target_col} - {file_name}")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            stats = {
                'mean': np.mean(signal_data).item(),
                'std': np.std(signal_data).item(),
                'peak': np.max(signal_data).item(),
            }
            chat_context = {'file': file_name, 'type': '1D Sinyal', 'column': target_col, 'stats': stats}
            
            # Grafiği belleğe kaydet
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            image_bytes = buf.getvalue() 

            # --- !!! YENİ: Gemini için Yapılandırılmış Prompt (1D) !!! ---
            prompt_parts = [
                {"mime_type": "image/png", "data": image_bytes},
                f"Sen bir PD (Kısmi Boşalma) uzmanısın. Yukarıdaki 1D sinyal grafiğini ('{target_col}') ve istatistikleri analiz et.",
                f"İSTATİSTİKLER:\n{json.dumps(stats, indent=2)}\n\n",
                f"GÖREV: Bu verilere (yüksek pikler, yüksek standart sapma, ani sıçramalar) dayanarak, bir PD sinyali olup olmadığını belirle.",
                f"PD VAR (1) veya PD YOK (0) tahmini yap ve güven skorunu (0-100) belirle.",
                f"Teknik bir açıklama yaz.",
                f"\n⚠️ CEVAP FORMATI (TAM OLARAK BU SATIRLARLA):",
                f"TAHMIN: [sadece 0 veya 1]",
                f"GUVEN: [sadece sayı 0-100]",
                f"ACIKLAMA: [2-3 cümle teknik analiz]"
            ]

        else:
            raise ValueError("Desteklenmeyen dosya formatı. Lütfen .csv veya .bin yükleyin.")

        # --- Gemini Analizi ve Yanıtı İşleme ---
        response = system_state.model.generate_content(prompt_parts)
        result_text = response.text
        
        # Yanıtı parse et
        for line in result_text.split('\n'):
             line = line.strip()
             if line.startswith('TAHMIN:'):
                 try: prediction = int(line.split(':')[1].strip())
                 except: pass
             elif line.startswith('GUVEN:'):
                 try: confidence = float(line.split(':')[1].strip())
                 except: confidence = 0.0
             elif line.startswith('ACIKLAMA:'):
                 explanation = line.split(':', 1)[1].strip()

        if prediction is None:
             explanation = f"Gemini'den yapısal yanıt alınamadı. Ham yanıt:\n{result_text}"
             prediction = -1 # Belirsiz durumu
             confidence = 0.0

        # --- Çıktıları Hazırla ---
        
        # 1. PD Tahmin Metni
        if prediction == 1:
            pd_prediction_text = "PD TAHMİNİ: MEVCUT 🔴"
        elif prediction == 0:
            pd_prediction_text = "PD TAHMİNİ: YOK 🟢"
        else:
            pd_prediction_text = "PD TAHMİNİ: BELİRSİZ ⚠️"

        # 2. Güven Skalası Rengi
        confidence_label = get_confidence_label(confidence)
        
        # 3. Chatbot Başlangıç Mesajı
        chat_history_msg = f"Merhaba! {file_name} dosyasını analiz ettim.\n\n**Tahminim: {pd_prediction_text} (Güven: {confidence:.1f}%)**\n\nGrafik yukarıdadır. Sinyal hakkında soru sorabilirsiniz."
        chat_history = [[
            None,
            f"{chat_history_msg}\n\n🤖 **Gemini'nin Analiz Açıklaması:**\n{explanation}"
        ]]
        
        # 4. Ana Açıklama Metni (Artık 'explanation' değişkeni)
        analysis_text = explanation
        
        # Grafiği Gradio'ya döndür
        plt.close(fig) # Artık grafiği kapattık, image_bytes'ı göndereceğiz... HAYIR, Gradio plot objesi fig'i ister.
        # plt.close(fig) satırını kaldırıp, fig'i döndürmeliyiz.
        # Grafiği belleğe kaydet (PNG formatında)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        # image_bytes = buf.getvalue() # Bu satır Gemini içindi, zaten var.
        
        # FIGÜRÜ KAPATMA! Gradio'nun ona ihtiyacı var.
        # plt.close(fig) <-- BU SATIRI YORUMA AL VEYA SİL
        
        # Gradio'ya döndür
        return fig, analysis_text, chat_history, gr.update(visible=True), pd_prediction_text, confidence_label

    except Exception as e:
        plt.close(fig) # Hata durumunda figürü kapat
        error_msg = f"❌ Hata: {str(e)}"
        print(error_msg)
        return None, error_msg, None, gr.update(visible=False), "HATA", default_confidence_label


def chat_func_generic(message: str, history_list: List[List[str]], system_state: PDDetectionSystemNPY, chat_context: Optional[Dict] = None):
    # (Bu fonksiyon değişmedi)
    """
    (YENİ - TAB 2 ve 3) Yeni sekmeler için genel sohbet fonksiyonu.
    """
    if system_state is None:
        history_list.append([message, "❌ Sistem başlatılmadı. Lütfen önce API Key girip sistemi başlatın."])
        return history_list
    try:
        gemini_history = []
        if chat_context and not history_list:
             file_info = chat_context.get('file', 'mevcut dosya')
             stats_info = chat_context.get('stats', {})
             context_prompt = f"Şu an {file_info} adlı dosya hakkında konuşuyoruz. İstatistikleri: {json.dumps(stats_info, indent=2)}. Soruları bu bağlamda yanıtla."
             message = f"[KONTEKST: {file_info} dosyası] {message}"
        for user_msg, bot_msg in history_list:
            if user_msg:
                gemini_history.append({"role": "user", "parts": [user_msg]})
            if bot_msg:
                gemini_history.append({"role": "model", "parts": [bot_msg]})
        chat_session = system_state.model.start_chat(history=gemini_history)
        response = chat_session.send_message(message)
        history_list.append([message, response.text])
        return history_list
    except Exception as e:
        history_list.append([message, f"❌ Gemini ile iletişim hatası: {str(e)}"])
        return history_list


# ------------------------------------------------------------------------------
# !!! BÖLÜM 6: GRADIO ARAYÜZÜNÜN OLUŞTURULMASI (YENİ DÜZEN) !!!
# ------------------------------------------------------------------------------

def create_gradio_interface_npy():
    with gr.Blocks(title="PD Tespit ve Öğrenme Sistemi", theme=gr.themes.Soft()) as interface:
        
        gr.Markdown("""
        # 🔌 Kısmi Boşalma (PD) Tespit ve Öğrenme Sistemi
        ### Gemini Flash 2.0 ile Akıllı Sinyal Analizi, Öğrenme ve Soru-Cevap
        """)
        
        # --- Sistem Başlatma ---
        system_state = gr.State(None)
        with gr.Row(variant="panel"):
            with gr.Column(scale=3):
                api_key_input = gr.Textbox(
                    label="🔑 Gemini API Key", 
                    placeholder="API anahtarınızı buraya yapıştırın (AIza...)", 
                    type="password",
                    value=os.getenv('GEMINI_API_KEY')
                )
            with gr.Column(scale=1):
                init_btn = gr.Button("🚀 Sistemi Başlat", variant="primary", scale=1)
            init_output = gr.Textbox(label="Sistem Durumu", interactive=False, scale=4)
        
        init_btn.click(
            fn=initialize_system_gr, 
            inputs=[api_key_input], 
            outputs=[init_output, system_state]
        )

        with gr.Tabs():
            # --- TAB 1: Standart NPY Analizi (Değişmedi) ---
            with gr.Tab("📊 PD Analizi (.npy)"):
                gr.Markdown("Bu sekmede, `.npy` formatındaki genlik verilerini analiz edebilir, PD tahmini alabilir ve modeli eğitebilirsiniz.")
                with gr.Tabs():
                    with gr.Tab("📄 Tek Dosya Analizi"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                single_file = gr.File(label="📁 Genlik Verisi (.npy)", file_types=[".npy"])
                                true_label_single = gr.Radio(choices=["Bilinmiyor", "PD YOK", "PD VAR"], label="🏷️ Gerçek Etiket (öğrenme için)", value="Bilinmiyor")
                                analyze_single_btn = gr.Button("🔍 Analiz Et", variant="primary")
                            with gr.Column(scale=2):
                                single_output = gr.Textbox(label="📊 Analiz Sonucu", lines=15, interactive=False)
                                single_csv_output = gr.Textbox(label="💾 CSV Verileri (kopyalayın)", lines=5, interactive=False)
                                data_preview_output = gr.Textbox(label="👁️ Veri Önizleme (ilk 200 değer)", lines=5, interactive=False)
                        
                        single_plot = gr.Plot(label="📈 Genlik Grafiği (vs Index)")
                        
                        analyze_single_btn.click(
                            fn=analyze_single_file_gr,
                            inputs=[single_file, true_label_single, system_state],
                            outputs=[single_output, single_plot, single_csv_output, data_preview_output]
                        )
                    
                    with gr.Tab("📚 Toplu Analiz"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                batch_files = gr.File(label="📁 Genlik Verileri (.npy)", file_count="multiple", file_types=[".npy"])
                                analyze_batch_btn = gr.Button("🔍 Toplu Analiz Et", variant="primary")
                            with gr.Column(scale=2):
                                batch_output = gr.Textbox(label="📊 Toplu Analiz Raporu", lines=15, interactive=False)
                                batch_csv_output = gr.Textbox(label="💾 CSV Verileri (kopyalayın)", lines=5, interactive=False)
                        batch_plot = gr.Plot(label="📈 Özet Grafikler")
                        
                        analyze_batch_btn.click(
                            fn=analyze_batch_files_gr, 
                            inputs=[batch_files, system_state], 
                            outputs=[batch_output, batch_plot, batch_csv_output]
                        )

            # --- TAB 2: Yeni Öğrenme Sekmesi (Değişmedi) ---
            with gr.Tab("🔬 Gemini PD Öğrenme (NPY)"):
                gr.Markdown("Bu sekmede, birden fazla `.npy` dosyası yükleyerek Gemini'nin bu verilerden PD **konseptini** öğrenmesini sağlayabilir ve bu konu hakkında sohbet edebilirsiniz.")
                
                with gr.Row(variant="panel"):
                    with gr.Column(scale=1):
                        learn_files_npy = gr.File(label="📁 Öğrenme için .npy Dosyaları", file_count="multiple", file_types=[".npy"])
                        learn_btn_npy = gr.Button("🧠 Verilerden PD Konseptini Öğren", variant="primary")
                    with gr.Column(scale=2):
                        learn_output_npy = gr.Textbox(label="🤖 Gemini'nin Öğrenme Özeti", lines=10, interactive=False)

                with gr.Group(visible=False) as chat_box_learn:
                    gr.Markdown("#### 💬 Gemini ile Öğrenilenler Üzerine Sohbet Et")
                    chatbot_learn = gr.Chatbot(label="PD Soru-Cevap", height=400)
                    chat_msg_learn = gr.Textbox(label="Sorunuz", placeholder="Bu verilere göre PD sinyalinin en belirgin özelliği nedir?", scale=3)
                    chat_submit_learn = gr.Button("Gönder", scale=1)
                
                chat_context_learn = gr.State(None) 

                learn_btn_npy.click(
                    fn=learn_pd_concept_from_npy,
                    inputs=[learn_files_npy, system_state],
                    outputs=[learn_output_npy, chatbot_learn, chat_box_learn]
                )

                chat_msg_learn.submit(
                    fn=chat_func_generic,
                    inputs=[chat_msg_learn, chatbot_learn, system_state, chat_context_learn],
                    outputs=[chatbot_learn]
                ).then(lambda: gr.update(value=""), outputs=[chat_msg_learn])
                
                chat_submit_learn.click(
                    fn=chat_func_generic,
                    inputs=[chat_msg_learn, chatbot_learn, system_state, chat_context_learn],
                    outputs=[chatbot_learn]
                ).then(lambda: gr.update(value=""), outputs=[chat_msg_learn])


            # --- !!! TAB 3: YENİ DÜZEN !!! ---
            with gr.Tab("🧪 Diğer Formatları Test Et (CSV/BIN)"):
                gr.Markdown("Bu sekmede, `.csv` veya `.bin` formatındaki dosyaları yükleyerek bunları analiz edebilir, PD tahmini alabilir ve Gemini ile sohbet edebilirsiniz.")
                
                with gr.Row(variant="panel"):
                    # Sütun 1: Yükleme ve Ayarlar
                    with gr.Column(scale=2):
                        test_file_other = gr.File(label="📁 Test Dosyası (.csv, .bin)", file_types=[".csv", ".bin"])
                        gr.Markdown("**Yardımcı Ayarlar:**")
                        csv_col_input = gr.Textbox(label="CSV Sütun Adı (isteğe bağlı)", placeholder="örn: 'genlik' (boş bırakırsanız otomatik bulur)")
                        bin_dtype_input = gr.Dropdown(label="BIN Veri Tipi (gerekliyse)", choices=['float32', 'float64', 'int16', 'int32'], value='float32')
                        test_btn_other = gr.Button("🔍 Analiz Et ve Grafiğe Dök", variant="primary")
                    
                    # Sütun 2: Tahmin ve Güven Skalası
                    with gr.Column(scale=1):
                        gr.Markdown("#### 🤖 Gemini PD Tahmini")
                        pd_prediction_output = gr.Textbox(
                            label="PD Durumu", 
                            value="N/A", 
                            interactive=False,
                            text_align="center"
                        )
                        # Renkli Skala için gr.Label kullanıyoruz
                        confidence_label_output = gr.Label(
                            label="Güven Skoru", 
                            value="N/A",
                            color="#fca311" # Başlangıç rengi (nötr)
                        )

                # Panelin Altı: Grafik
                test_plot_other = gr.Plot(label="📈 Çıkarılan Sinyal Grafiği")
                
                # Grafiğin Altı: Açıklama ve Sohbet
                test_output_other = gr.Textbox(label="🤖 Gemini'nin Analiz Açıklaması", lines=4, interactive=False)

                with gr.Group(visible=False) as chat_box_test:
                    gr.Markdown("#### 💬 Gemini ile Analiz Edilen Sinyal Üzerine Sohbet Et")
                    chatbot_test = gr.Chatbot(label="Test Soru-Cevap", height=400)
                    chat_msg_test = gr.Textbox(label="Sorunuz", placeholder="Bu sinyaldeki pik değeri nedir?", scale=3)
                    chat_submit_test = gr.Button("Gönder", scale=1)

                chat_context_test = gr.State(None)

                # Click event'ini yeni çıktılara göre güncelle
                test_btn_other.click(
                    fn=process_and_visualize_other_formats,
                    inputs=[test_file_other, csv_col_input, bin_dtype_input, system_state],
                    outputs=[
                        test_plot_other, 
                        test_output_other,  # Gemini'nin açıklaması
                        chatbot_test, 
                        chat_box_test,
                        pd_prediction_output, # "PD TAHMİNİ: MEVCUT"
                        confidence_label_output # Renkli güven skalası
                    ]
                )

                chat_msg_test.submit(
                    fn=chat_func_generic,
                    inputs=[chat_msg_test, chatbot_test, system_state, chat_context_test],
                    outputs=[chatbot_test]
                ).then(lambda: gr.update(value=""), outputs=[chat_msg_test])

                chat_submit_test.click(
                    fn=chat_func_generic,
                    inputs=[chat_msg_test, chatbot_test, system_state, chat_context_test],
                    outputs=[chatbot_test]
                ).then(lambda: gr.update(value=""), outputs=[chat_msg_test])


            # --- TAB 4: Loglar ve İstatistikler (Değişmedi) ---
            with gr.Tab("🧠 Öğrenme & Log Kayıtları"):
                gr.Markdown("Bu sekmede, `.npy` analizlerinden elde edilen öğrenme veritabanını (logları) ve genel doğruluk istatistiklerini görebilirsiniz.")
                with gr.Row():
                    refresh_stats_btn = gr.Button("🔄 İstatistikleri ve Logları Yenile", variant="secondary")
                    reset_db_btn = gr.Button("🗑️ Öğrenme Veritabanını Sıfırla", variant="stop")
                
                learning_stats_output = gr.Textbox(label="📊 Öğrenme Veritabanı ve Loglar", lines=25, interactive=False)
                
                refresh_stats_btn.click(
                    fn=get_learning_stats_gr, 
                    inputs=[system_state], 
                    outputs=[learning_stats_output]
                )
                reset_db_btn.click(
                    fn=reset_learning_db_gr, 
                    inputs=[system_state], 
                    outputs=[learning_stats_output]
                )

        gr.Markdown("--- \n ### 💡 Kullanım İpuçları \n - **API Key:** Sistemi kullanmak için önce geçerli bir Gemini API anahtarı girmelisiniz.\n - **.npy Formatı:** 'PD Analizi' sekmesi sadece genlik verisi içeren tek boyutlu NumPy dizilerini okuyabilir. \n - **Etiketleme:** Dosya adında 'pd' (PD için) veya 'no_pd'/'normal' (Normal için) kelimeleri varsa, etiket otomatik çıkarılır. Elle etiket girerseniz, o kullanılır.")
    return interface

# ------------------------------------------------------------------------------
# BÖLÜM 7: WEB ARAYÜZÜNÜ BAŞLATMA (Değişmedi)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    load_dotenv()
    print("\n🌐 Gelişmiş Gradio web arayüzü oluşturuluyor...")
    app_interface = create_gradio_interface_npy()
    print("\n🚀 Arayüz başlatılıyor...")
    app_interface.launch(server_name="127.0.0.1", server_port=7860, share=False, show_error=True)
