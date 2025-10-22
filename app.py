"""
Kısmi Boşalma (PD) Tespiti Sistemi
Gemini Flash 2.0 ile Spektrum Analizi ve Öğrenme

Kullanım:
1. pip install google-generativeai pandas numpy matplotlib gradio python-dotenv
2. .env dosyasında GEMINI_API_KEY tanımlayın
3. python pd_detection_demo.py
"""

import os
import io
import json
import numpy as np
import pandas as pd
import google.generativeai as genai
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import gradio as gr
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Optional, Any

print("🚀 Kütüphaneler yüklendi.")

# API Key'i .env'den yükle
load_dotenv()
GLOBAL_API_KEY = os.getenv('GEMINI_API_KEY')

# ==================== DEMO DOSYA YAPISI ====================

DEMO_FILES = {
    "training": {
        "description": "Eğitim için NPY dosyaları",
        "path": "pd_values_for_training",
        "examples": ["500927.npy", "500945.npy", "501057.npy", "501084.npy"]
    },
    "pd_examples": {
        "description": "PD içeren örnekler",
        "path": "npy_examples/pd",
        "examples": ["575454.npy", "575923.npy"],
        "label": 1
    },
    "no_pd_examples": {
        "description": "PD içermeyen örnekler",
        "path": "npy_examples/no_pd",
        "examples": ["576415.npy", "730183.npy", "730270.npy"],
        "label": 0
    },
    "csv_pd": {
        "description": "CSV PD örnekleri",
        "path": "csv_examples/scenario_PD",
        "examples": ["A1_1k_5M_Average.csv"],
        "label": 1
    },
    "csv_no_pd": {
        "description": "CSV PD yok örnekleri",
        "path": "csv_examples/scenario_no_PD",
        "examples": ["A1_1k_5M_Average.csv"],
        "label": 0
    }
}

# ==================== NPY ENCODER ====================

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        return json.JSONEncoder.default(self, obj)

# ==================== PD TESPİT SİSTEMİ ====================

class PDDetectionSystemNPY:
    def __init__(self, api_key: str):
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("API key gerekli! .env dosyasında GEMINI_API_KEY tanımlayın.")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        self.learning_db_path = Path('pd_learning_database_demo.json')
        self.learning_data = self.load_learning_data()
        print("✓ PD Tespit Sistemi başlatıldı")

    def load_learning_data(self) -> List[Dict]:
        if self.learning_db_path.exists():
            try:
                with open(self.learning_db_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return []
        return []

    def save_learning_data(self):
        self.learning_data = self.learning_data[-200:]
        with open(self.learning_db_path, 'w', encoding='utf-8') as f:
            json.dump(self.learning_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    def parse_npy_file(self, file_path: str) -> Dict:
        """NPY dosyasını parse et"""
        print(f"\n📄 Parsing NPY: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dosya bulunamadı: {file_path}")
        
        mag_array = np.load(file_path)
        
        if mag_array.ndim != 1:
            raise ValueError(f"Tek boyutlu dizi bekleniyor, shape={mag_array.shape}")
        
        mag_array = mag_array[np.isfinite(mag_array)]
        
        statistics = {
            'mean': float(np.mean(mag_array)),
            'std': float(np.std(mag_array)),
            'peak': float(np.max(mag_array)),
            'min': float(np.min(mag_array)),
            'range': float(np.ptp(mag_array)),
            'median': float(np.median(mag_array))
        }
        
        mag_diff = np.diff(mag_array)
        sudden_jumps = int(np.sum(np.abs(mag_diff) > 10))
        
        # Dosya adından etiket çıkar
        file_name_lower = Path(file_path).name.lower()
        inferred_label = None
        if 'pd' in file_name_lower and 'no' not in file_name_lower:
            inferred_label = 1
        elif 'no_pd' in file_name_lower or 'normal' in file_name_lower:
            inferred_label = 0
        
        print(f"   ✓ {len(mag_array)} veri noktası, Ort: {statistics['mean']:.2f}, Std: {statistics['std']:.2f}")
        
        return {
            'file_path': file_path,
            'statistics': statistics,
            'data_points': len(mag_array),
            'magnitudes': mag_array,
            'sudden_jumps': sudden_jumps,
            'inferred_label': inferred_label
        }

    def analyze_with_gemini_npy(self, parsed_data: Dict, file_name: str, true_label: Optional[int] = None) -> Dict:
        """Gemini ile NPY analizi"""
        final_true_label = true_label if true_label is not None else parsed_data.get('inferred_label')
        
        # Öğrenme context
        learning_context = ""
        if self.learning_data:
            recent = self.learning_data[-5:]
            learning_context = "\n\n=== ÖNCEKİ ÖĞRENME ===\n"
            for entry in recent:
                pred = "PD VAR" if entry.get('prediction') == 1 else "PD YOK"
                correct = "✓" if entry.get('correct') else "✗" if entry.get('correct') is False else "?"
                learning_context += f"{entry.get('file_name')}: {pred} {correct}\n"
        
        prompt = f"""Sen bir PD tespit uzmanısın.

DOSYA: {file_name}
VERİ NOKTASI: {parsed_data['data_points']}
ORTALAMA: {parsed_data['statistics']['mean']:.2f} dBm
STD SAPMA: {parsed_data['statistics']['std']:.2f} dB
PIK: {parsed_data['statistics']['peak']:.2f} dBm
ANI SIÇRAMA: {parsed_data['sudden_jumps']}

{learning_context}

{"GERÇEK ETİKET: " + ("PD VAR" if final_true_label == 1 else "PD YOK") if final_true_label is not None else ""}

GÖREVİN:
1. Bu istatistiklere göre PD olup olmadığını belirle
2. PD VAR (1) veya PD YOK (0) tahmin et
3. 0-100 güven skoru ver
4. Kısa açıklama yaz

CEVAP FORMATI:
TAHMIN: [0 veya 1]
GUVEN: [0-100]
ACIKLAMA: [2-3 cümle]
OGRENME_NOTU: [gerçek etiket varsa öğrenme notu, yoksa YOK]
"""
        
        try:
            response = self.model.generate_content(prompt, generation_config={'temperature': 0.2})
            result_text = response.text
            
            prediction, confidence, explanation, lesson = None, 0.0, "Analiz yapılamadı", "YOK"
            for line in result_text.split('\n'):
                line = line.strip()
                if line.startswith('TAHMIN:'):
                    try:
                        prediction = int(line.split(':')[1].strip())
                    except:
                        pass
                elif line.startswith('GUVEN:'):
                    try:
                        confidence = float(line.split(':')[1].strip())
                    except:
                        pass
                elif line.startswith('ACIKLAMA:'):
                    explanation = line.split(':', 1)[1].strip()
                elif line.startswith('OGRENME_NOTU:'):
                    lesson_text = line.split(':', 1)[1].strip()
                    if lesson_text and lesson_text != "YOK":
                        lesson = lesson_text
            
            correct = None
            if final_true_label is not None and prediction is not None:
                correct = (prediction == final_true_label)
            
            result = {
                'file_name': file_name,
                'prediction': prediction,
                'confidence': confidence,
                'true_label': final_true_label,
                'correct': correct,
                'explanation': explanation,
                'lesson': lesson,
                'statistics': parsed_data['statistics'],
                'timestamp': datetime.now().isoformat(),
                'plot_data': {'magnitudes': parsed_data['magnitudes']}
            }
            
            self.learning_data.append(result)
            self.save_learning_data()
            
            return result
            
        except Exception as e:
            return {'file_name': file_name, 'error': str(e)}

    def analyze_npy_file(self, file_path: str, true_label: Optional[int] = None) -> Dict:
        """Tek dosya analizi"""
        try:
            parsed_data = self.parse_npy_file(file_path)
            result = self.analyze_with_gemini_npy(parsed_data, Path(file_path).name, true_label)
            return result
        except Exception as e:
            return {'file_name': Path(file_path).name, 'error': str(e)}

# ==================== GRADIO FONKSİYONLARI ====================

def initialize_system_gr():
    """Sistemi başlat (.env'den API key kullan)"""
    if not GLOBAL_API_KEY:
        return "❌ GEMINI_API_KEY .env dosyasında bulunamadı!", None
    try:
        system_instance = PDDetectionSystemNPY(api_key=GLOBAL_API_KEY)
        count = len(system_instance.learning_data)
        
        # Demo dosya listesi
        demo_info = "\n\n📚 ÖRNEK DOSYALAR:\n"
        demo_info += "\n🎓 Eğitim: pd_values_for_training/500927.npy, 500945.npy, 501057.npy, 501084.npy"
        demo_info += "\n🔴 PD: npy_examples/pd/575454.npy, 575923.npy"
        demo_info += "\n🟢 Normal: npy_examples/no_pd/576415.npy, 730183.npy, 730270.npy"
        
        return f"✅ Sistem başlatıldı! Öğrenme: {count} kayıt{demo_info}", system_instance
    except Exception as e:
        return f"❌ Hata: {str(e)}", None

def analyze_from_path_gr(file_path_input, true_label_str, system_state):
    """Dosya yolundan analiz yap"""
    if system_state is None:
        return "❌ Sistem başlatılmadı.", None, None, "Sistem başlatılmadı."
    
    if not file_path_input or file_path_input.strip() == "":
        return "❌ Lütfen dosya yolu girin!", None, None, "Dosya yolu girilmedi."
    
    file_path = file_path_input.strip()
    
    try:
        true_label = None
        if true_label_str and true_label_str != "Bilinmiyor":
            true_label = 1 if true_label_str == "PD VAR" else 0
        
        result = system_state.analyze_npy_file(file_path, true_label=true_label)
        
        if 'error' in result:
            return f"❌ Hata: {result['error']}", None, None, f"Hata: {result['error']}"
        
        # Rapor
        pred_emoji = "🔴" if result.get('prediction') == 1 else "🟢"
        pred_text = "PD VAR" if result.get('prediction') == 1 else "PD YOK"
        
        output = "=" * 60 + "\n📊 ANALİZ SONUCU\n" + "=" * 60 + "\n\n"
        output += f"📁 Dosya: {file_path}\n"
        output += f"🎯 TAHMİN: {pred_emoji} {pred_text}\n"
        output += f"📈 GÜVEN: {result.get('confidence', 0):.2f}%\n\n"
        
        if result.get('true_label') is not None:
            true_text = "PD VAR" if result['true_label'] == 1 else "PD YOK"
            status = "✅ DOĞRU" if result.get('correct') else "❌ YANLIŞ"
            output += f"🏷️ GERÇEK: {true_text}\n📊 SONUÇ: {status}\n\n"
        
        output += f"💭 AÇIKLAMA:\n{result.get('explanation', '-')}\n\n"
        
        if result.get('lesson') and result['lesson'] != "YOK":
            output += f"🧠 ÖĞRENİLEN:\n{result['lesson']}\n\n"
        
        stats = result.get('statistics', {})
        output += "-" * 60 + "\n📈 İSTATİSTİKLER\n" + "-" * 60 + "\n"
        output += f"Ortalama: {stats.get('mean', 0):.2f} dBm\n"
        output += f"Std Sapma: {stats.get('std', 0):.2f} dB\n"
        output += f"Pik: {stats.get('peak', 0):.2f} dBm\n"
        output += f"Min: {stats.get('min', 0):.2f} dBm\n"
        output += f"Aralık: {stats.get('range', 0):.2f} dB\n"
        
        # Grafik
        fig, ax = plt.subplots(figsize=(12, 5))
        plot_data = result.get('plot_data', {})
        mags = plot_data.get('magnitudes')
        
        if isinstance(mags, np.ndarray) and len(mags) > 0:
            ax.plot(mags, 'b-', linewidth=0.8, alpha=0.7)
            ax.set_xlabel('Örnek Numarası')
            ax.set_ylabel('Güç (dBm)')
            ax.set_title(f'Spektrum Analizi - {pred_text}')
            ax.grid(True, alpha=0.3)
            
            stats_text = f"Ort: {stats.get('mean', 0):.1f}\nStd: {stats.get('std', 0):.1f}\nPik: {stats.get('peak', 0):.1f}"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, va='top',
                   bbox=dict(boxstyle='round', fc='lightblue', alpha=0.7))
        
        plt.tight_layout()
        
        # CSV
        csv_data = f"Özellik,Değer\n"
        csv_data += f"Dosya,{file_path}\n"
        csv_data += f"Tahmin,{pred_text}\n"
        csv_data += f"Güven,{result.get('confidence', 0):.2f}\n"
        for k, v in stats.items():
            csv_data += f"{k},{v:.2f}\n"
        
        # Önizleme
        preview = " ".join([f"{v:.4f}" for v in mags[:200]]) if isinstance(mags, np.ndarray) else "Veri yok"
        if isinstance(mags, np.ndarray) and len(mags) > 200:
            preview += " ..."
        
        return output, fig, csv_data, preview
        
    except Exception as e:
        return f"❌ Hata: {str(e)}", None, None, f"Hata: {str(e)}"

def get_learning_stats_gr(system_state):
    """Öğrenme istatistikleri"""
    if system_state is None:
        return "❌ Sistem başlatılmadı."
    
    if not system_state.learning_data:
        return "📊 Henüz öğrenme verisi yok."
    
    total = len(system_state.learning_data)
    labeled = [d for d in system_state.learning_data if d.get('true_label') is not None]
    correct = [d for d in labeled if d.get('correct') == True]
    accuracy = (len(correct) / len(labeled) * 100) if labeled else 0
    
    output = "=" * 60 + "\n🧠 ÖĞRENME İSTATİSTİKLERİ\n" + "=" * 60 + "\n\n"
    output += f"📊 Toplam Analiz: {total}\n"
    output += f"🏷️ Etiketli: {len(labeled)}\n"
    output += f"✅ Doğru: {len(correct)}\n"
    output += f"❌ Yanlış: {len(labeled) - len(correct)}\n"
    output += f"📈 Doğruluk: {accuracy:.2f}%\n\n"
    
    recent_with_lessons = [d for d in system_state.learning_data[-10:] if d.get('lesson') and d['lesson'] != "YOK"][-5:]
    if recent_with_lessons:
        output += "-" * 60 + "\n📚 SON ÖĞRENİLENLER\n" + "-" * 60 + "\n"
        for i, d in enumerate(recent_with_lessons, 1):
            output += f"\n{i}. {d['file_name']}\n   {d['lesson']}\n"
    
    return output

def reset_learning_db_gr(system_state):
    """Veritabanını sıfırla"""
    if system_state is None:
        return "❌ Sistem başlatılmadı!"
    try:
        system_state.learning_data = []
        system_state.save_learning_data()
        return "✅ Öğrenme veritabanı sıfırlandı!"
    except Exception as e:
        return f"❌ Hata: {str(e)}"

# ==================== GRADIO ARAYÜZÜ ====================

def create_gradio_interface_demo():
    with gr.Blocks(title="PD Tespit Sistemi", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🔌 Kısmi Boşalma (PD) Tespit Sistemi
        ### Gemini Flash 2.0 ile Akıllı Analiz ve Öğrenme
        """)
        
        system_state = gr.State(None)
        
        # Sistem Başlatma (API Key .env'den okunacak)
        with gr.Row(variant="panel"):
            with gr.Column(scale=3):
                if GLOBAL_API_KEY:
                    gr.Markdown("✅ **API Key .env dosyasından yüklendi**")
                else:
                    gr.Markdown("❌ **Uyarı: .env dosyasında GEMINI_API_KEY bulunamadı!**")
            with gr.Column(scale=1):
                init_btn = gr.Button("🚀 Sistemi Başlat", variant="primary")
        
        init_output = gr.Textbox(label="Sistem Durumu", lines=8, interactive=False)
        
        init_btn.click(
            fn=initialize_system_gr,
            inputs=[],
            outputs=[init_output, system_state]
        )
        
        with gr.Tabs():
            # Tab 1: Tek Analiz
            with gr.Tab("📄 Tek Dosya Analizi"):
                gr.Markdown("""
                ### 📝 Dosya Yolu Örnekleri:
                
                **🎓 Eğitim:**
                - `pd_values_for_training/500927.npy`
                - `pd_values_for_training/500945.npy`
                
                **🔴 PD Var:**
                - `npy_examples/pd/575454.npy`
                - `npy_examples/pd/575923.npy`
                
                **🟢 PD Yok:**
                - `npy_examples/no_pd/576415.npy`
                - `npy_examples/no_pd/730183.npy`
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        file_path_input = gr.Textbox(
                            label="📁 Dosya Yolu",
                            placeholder="npy_examples/pd/575454.npy",
                            lines=1
                        )
                        true_label_radio = gr.Radio(
                            choices=["Bilinmiyor", "PD YOK", "PD VAR"],
                            label="🏷️ Gerçek Etiket (öğrenme için)",
                            value="Bilinmiyor"
                        )
                        analyze_btn = gr.Button("🔍 Analiz Et", variant="primary")
                    
                    with gr.Column(scale=2):
                        single_output = gr.Textbox(label="📊 Analiz Sonucu", lines=20, interactive=False)
                
                single_plot = gr.Plot(label="📈 Spektrum Grafiği")
                
                with gr.Row():
                    csv_output = gr.Textbox(label="💾 CSV", lines=5, interactive=False)
                    preview_output = gr.Textbox(label="👁️ Veri Önizleme", lines=5, interactive=False)
                
                analyze_btn.click(
                    fn=analyze_from_path_gr,
                    inputs=[file_path_input, true_label_radio, system_state],
                    outputs=[single_output, single_plot, csv_output, preview_output]
                )
            
            # Tab 2: Öğrenme
            with gr.Tab("🧠 Öğrenme İstatistikleri"):
                with gr.Row():
                    refresh_btn = gr.Button("🔄 İstatistikleri Yenile", variant="secondary")
                    reset_btn = gr.Button("🗑️ Sıfırla", variant="stop")
                
                stats_output = gr.Textbox(label="📊 Öğrenme Veritabanı", lines=25, interactive=False)
                
                refresh_btn.click(
                    fn=get_learning_stats_gr,
                    inputs=[system_state],
                    outputs=[stats_output]
                )
                
                reset_btn.click(
                    fn=reset_learning_db_gr,
                    inputs=[system_state],
                    outputs=[stats_output]
                )
        
        gr.Markdown("""
        ---
        ### 💡 Kullanım İpuçları
        - **API Key:** .env dosyasında `GEMINI_API_KEY=AIza...` şeklinde tanımlayın
        - **Dosya Yolları:** Yukarıdaki örnekleri kopyalayıp yapıştırabilirsiniz
        - **Etiketleme:** Gerçek etiketi işaretlerseniz model öğrenir ve doğruluk hesaplanır
        - **Öğrenme:** Her analiz kaydedilir ve sonraki tahminlerde kullanılır
        
        **Geliştirici:** Gemini Flash 2.0 + Python + Gradio
        """)
    
    return interface

# ==================== MAIN ====================

if __name__ == "__main__":
    print("\n🌐 Gradio arayüzü oluşturuluyor...")
    
    if not GLOBAL_API_KEY:
        print("⚠️  UYARI: GEMINI_API_KEY .env dosyasında bulunamadı!")
        print("📝 .env dosyası oluşturun ve içine şunu ekleyin:")
        print("   GEMINI_API_KEY=AIza_sizin_key_buraya")
    else:
        print("✅ API Key .env'den yüklendi")
    
    app = create_gradio_interface_demo()
    print("\n🚀 Arayüz başlatılıyor...")
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


import os
import io
import json
import numpy as np
import pandas as pd
import google.generativeai as genai
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import gradio as gr
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Optional, Any

print("🚀 Kütüphaneler yüklendi.")

# ==================== DEMO DOSYA YAPISI ====================

DEMO_FILES = {
    "training": {
        "description": "Eğitim için NPY dosyaları",
        "path": "pd_values_for_training",
        "examples": ["500927.npy", "500945.npy", "501057.npy", "501084.npy"]
    },
    "pd_examples": {
        "description": "PD içeren örnekler",
        "path": "npy_examples/pd",
        "examples": ["575454.npy", "575923.npy"],
        "label": 1
    },
    "no_pd_examples": {
        "description": "PD içermeyen örnekler",
        "path": "npy_examples/no_pd",
        "examples": ["576415.npy", "730183.npy", "730270.npy"],
        "label": 0
    },
    "csv_pd": {
        "description": "CSV PD örnekleri",
        "path": "csv_examples/scenario_PD",
        "examples": ["A1_1k_5M_Average.csv"],
        "label": 1
    },
    "csv_no_pd": {
        "description": "CSV PD yok örnekleri",
        "path": "csv_examples/scenario_no_PD",
        "examples": ["A1_1k_5M_Average.csv"],
        "label": 0
    }
}

def get_demo_file_list(category: str) -> str:
    """Demo dosyalarının listesini döndür"""
    if category not in DEMO_FILES:
        return "Geçersiz kategori"
    
    info = DEMO_FILES[category]
    output = f"📁 {info['description']}\n"
    output += f"📂 Klasör: {info['path']}/\n\n"
    output += "Örnek Dosyalar:\n"
    for ex in info['examples']:
        full_path = f"{info['path']}/{ex}"
        output += f"  • {full_path}\n"
    return output

# ==================== NPY ENCODER ====================

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        return json.JSONEncoder.default(self, obj)

# ==================== PD TESPİT SİSTEMİ ====================

class PDDetectionSystemNPY:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("API key gerekli!")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        self.learning_db_path = Path('pd_learning_database_demo.json')
        self.learning_data = self.load_learning_data()
        print("✓ PD Tespit Sistemi (Demo) başlatıldı")

    def load_learning_data(self) -> List[Dict]:
        if self.learning_db_path.exists():
            try:
                with open(self.learning_db_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return []
        return []

    def save_learning_data(self):
        self.learning_data = self.learning_data[-200:]
        with open(self.learning_db_path, 'w', encoding='utf-8') as f:
            json.dump(self.learning_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    def parse_npy_file(self, file_path: str) -> Dict:
        """NPY dosyasını parse et"""
        print(f"\n📄 Parsing NPY: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dosya bulunamadı: {file_path}")
        
        mag_array = np.load(file_path)
        
        if mag_array.ndim != 1:
            raise ValueError(f"Tek boyutlu dizi bekleniyor, shape={mag_array.shape}")
        
        mag_array = mag_array[np.isfinite(mag_array)]
        
        statistics = {
            'mean': float(np.mean(mag_array)),
            'std': float(np.std(mag_array)),
            'peak': float(np.max(mag_array)),
            'min': float(np.min(mag_array)),
            'range': float(np.ptp(mag_array)),
            'median': float(np.median(mag_array))
        }
        
        mag_diff = np.diff(mag_array)
        sudden_jumps = int(np.sum(np.abs(mag_diff) > 10))
        
        # Dosya adından etiket çıkar
        file_name_lower = Path(file_path).name.lower()
        inferred_label = None
        if 'pd' in file_name_lower and 'no' not in file_name_lower:
            inferred_label = 1
        elif 'no_pd' in file_name_lower or 'normal' in file_name_lower:
            inferred_label = 0
        
        print(f"   ✓ {len(mag_array)} veri noktası, Ort: {statistics['mean']:.2f}, Std: {statistics['std']:.2f}")
        
        return {
            'file_path': file_path,
            'statistics': statistics,
            'data_points': len(mag_array),
            'magnitudes': mag_array,
            'sudden_jumps': sudden_jumps,
            'inferred_label': inferred_label
        }

    def analyze_with_gemini_npy(self, parsed_data: Dict, file_name: str, true_label: Optional[int] = None) -> Dict:
        """Gemini ile NPY analizi"""
        final_true_label = true_label if true_label is not None else parsed_data.get('inferred_label')
        
        # Öğrenme context
        learning_context = ""
        if self.learning_data:
            recent = self.learning_data[-5:]
            learning_context = "\n\n=== ÖNCEKİ ÖĞRENME ===\n"
            for entry in recent:
                pred = "PD VAR" if entry.get('prediction') == 1 else "PD YOK"
                correct = "✓" if entry.get('correct') else "✗" if entry.get('correct') is False else "?"
                learning_context += f"{entry.get('file_name')}: {pred} {correct}\n"
        
        prompt = f"""Sen bir PD tespit uzmanısın.

DOSYA: {file_name}
VERİ NOKTASI: {parsed_data['data_points']}
ORTALAMA: {parsed_data['statistics']['mean']:.2f} dBm
STD SAPMA: {parsed_data['statistics']['std']:.2f} dB
PIK: {parsed_data['statistics']['peak']:.2f} dBm
ANI SIÇRAMA: {parsed_data['sudden_jumps']}

{learning_context}

{"GERÇEK ETİKET: " + ("PD VAR" if final_true_label == 1 else "PD YOK") if final_true_label is not None else ""}

GÖREVİN:
1. Bu istatistiklere göre PD olup olmadığını belirle
2. PD VAR (1) veya PD YOK (0) tahmin et
3. 0-100 güven skoru ver
4. Kısa açıklama yaz

CEVAP FORMATI:
TAHMIN: [0 veya 1]
GUVEN: [0-100]
ACIKLAMA: [2-3 cümle]
OGRENME_NOTU: [gerçek etiket varsa öğrenme notu, yoksa YOK]
"""
        
        try:
            response = self.model.generate_content(prompt, generation_config={'temperature': 0.2})
            result_text = response.text
            
            prediction, confidence, explanation, lesson = None, 0.0, "Analiz yapılamadı", "YOK"
            for line in result_text.split('\n'):
                line = line.strip()
                if line.startswith('TAHMIN:'):
                    try:
                        prediction = int(line.split(':')[1].strip())
                    except:
                        pass
                elif line.startswith('GUVEN:'):
                    try:
                        confidence = float(line.split(':')[1].strip())
                    except:
                        pass
                elif line.startswith('ACIKLAMA:'):
                    explanation = line.split(':', 1)[1].strip()
                elif line.startswith('OGRENME_NOTU:'):
                    lesson_text = line.split(':', 1)[1].strip()
                    if lesson_text and lesson_text != "YOK":
                        lesson = lesson_text
            
            correct = None
            if final_true_label is not None and prediction is not None:
                correct = (prediction == final_true_label)
            
            result = {
                'file_name': file_name,
                'prediction': prediction,
                'confidence': confidence,
                'true_label': final_true_label,
                'correct': correct,
                'explanation': explanation,
                'lesson': lesson,
                'statistics': parsed_data['statistics'],
                'timestamp': datetime.now().isoformat(),
                'plot_data': {'magnitudes': parsed_data['magnitudes']}
            }
            
            self.learning_data.append(result)
            self.save_learning_data()
            
            return result
            
        except Exception as e:
            return {'file_name': file_name, 'error': str(e)}

    def analyze_npy_file(self, file_path: str, true_label: Optional[int] = None) -> Dict:
        """Tek dosya analizi"""
        try:
            parsed_data = self.parse_npy_file(file_path)
            result = self.analyze_with_gemini_npy(parsed_data, Path(file_path).name, true_label)
            return result
        except Exception as e:
            return {'file_name': Path(file_path).name, 'error': str(e)}

# ==================== GRADIO FONKSİYONLARI ====================

def initialize_system_gr(api_key):
    """Sistemi başlat"""
    if not api_key:
        return "❌ Lütfen API anahtarını girin veya .env dosyasında GEMINI_API_KEY tanımlayın!", None
    try:
        system_instance = PDDetectionSystemNPY(api_key=api_key)
        count = len(system_instance.learning_data)
        
        # Demo dosya listesi
        demo_info = "\n\n📚 DEMO DOSYALAR:\n"
        demo_info += "\n🎓 Eğitim Örnekleri:"
        demo_info += "\n  • pd_values_for_training/500927.npy"
        demo_info += "\n  • pd_values_for_training/500945.npy"
        demo_info += "\n  • pd_values_for_training/501057.npy"
        demo_info += "\n  • pd_values_for_training/501084.npy"
        demo_info += "\n\n🔴 PD Örnekleri:"
        demo_info += "\n  • npy_examples/pd/575454.npy"
        demo_info += "\n  • npy_examples/pd/575923.npy"
        demo_info += "\n\n🟢 Normal Örnekleri:"
        demo_info += "\n  • npy_examples/no_pd/576415.npy"
        demo_info += "\n  • npy_examples/no_pd/730183.npy"
        demo_info += "\n  • npy_examples/no_pd/730270.npy"
        demo_info += "\n\n📊 CSV PD: csv_examples/scenario_PD/A1_1k_5M_Average.csv"
        demo_info += "\n📊 CSV Normal: csv_examples/scenario_no_PD/A1_1k_5M_Average.csv"
        
        return f"✅ Sistem başlatıldı! Öğrenme: {count} kayıt{demo_info}", system_instance
    except Exception as e:
        return f"❌ Hata: {str(e)}", None

def analyze_from_path_gr(file_path_input, true_label_str, system_state):
    """Dosya yolundan analiz yap"""
    if system_state is None:
        return "❌ Sistem başlatılmadı.", None, None, "Sistem başlatılmadı."
    
    if not file_path_input or file_path_input.strip() == "":
        return "❌ Lütfen dosya yolu girin!", None, None, "Dosya yolu girilmedi."
    
    file_path = file_path_input.strip()
    
    try:
        true_label = None
        if true_label_str and true_label_str != "Bilinmiyor":
            true_label = 1 if true_label_str == "PD VAR" else 0
        
        result = system_state.analyze_npy_file(file_path, true_label=true_label)
        
        if 'error' in result:
            return f"❌ Hata: {result['error']}", None, None, f"Hata: {result['error']}"
        
        # Rapor
        pred_emoji = "🔴" if result.get('prediction') == 1 else "🟢"
        pred_text = "PD VAR" if result.get('prediction') == 1 else "PD YOK"
        
        output = "=" * 60 + "\n📊 ANALİZ SONUCU\n" + "=" * 60 + "\n\n"
        output += f"📁 Dosya: {file_path}\n"
        output += f"🎯 TAHMİN: {pred_emoji} {pred_text}\n"
        output += f"📈 GÜVEN: {result.get('confidence', 0):.2f}%\n\n"
        
        if result.get('true_label') is not None:
            true_text = "PD VAR" if result['true_label'] == 1 else "PD YOK"
            status = "✅ DOĞRU" if result.get('correct') else "❌ YANLIŞ"
            output += f"🏷️ GERÇEK: {true_text}\n📊 SONUÇ: {status}\n\n"
        
        output += f"💭 AÇIKLAMA:\n{result.get('explanation', '-')}\n\n"
        
        if result.get('lesson') and result['lesson'] != "YOK":
            output += f"🧠 ÖĞRENİLEN:\n{result['lesson']}\n\n"
        
        stats = result.get('statistics', {})
        output += "-" * 60 + "\n📈 İSTATİSTİKLER\n" + "-" * 60 + "\n"
        output += f"Ortalama: {stats.get('mean', 0):.2f} dBm\n"
        output += f"Std Sapma: {stats.get('std', 0):.2f} dB\n"
        output += f"Pik: {stats.get('peak', 0):.2f} dBm\n"
        output += f"Min: {stats.get('min', 0):.2f} dBm\n"
        output += f"Aralık: {stats.get('range', 0):.2f} dB\n"
        
        # Grafik
        fig, ax = plt.subplots(figsize=(12, 5))
        plot_data = result.get('plot_data', {})
        mags = plot_data.get('magnitudes')
        
        if isinstance(mags, np.ndarray) and len(mags) > 0:
            ax.plot(mags, 'b-', linewidth=0.8, alpha=0.7)
            ax.set_xlabel('Örnek Numarası')
            ax.set_ylabel('Güç (dBm)')
            ax.set_title(f'Spektrum Analizi - {pred_text}')
            ax.grid(True, alpha=0.3)
            
            stats_text = f"Ort: {stats.get('mean', 0):.1f}\nStd: {stats.get('std', 0):.1f}\nPik: {stats.get('peak', 0):.1f}"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, va='top',
                   bbox=dict(boxstyle='round', fc='lightblue', alpha=0.7))
        
        plt.tight_layout()
        
        # CSV
        csv_data = f"Özellik,Değer\n"
        csv_data += f"Dosya,{file_path}\n"
        csv_data += f"Tahmin,{pred_text}\n"
        csv_data += f"Güven,{result.get('confidence', 0):.2f}\n"
        for k, v in stats.items():
            csv_data += f"{k},{v:.2f}\n"
        
        # Önizleme
        preview = " ".join([f"{v:.4f}" for v in mags[:200]]) if isinstance(mags, np.ndarray) else "Veri yok"
        if isinstance(mags, np.ndarray) and len(mags) > 200:
            preview += " ..."
        
        return output, fig, csv_data, preview
        
    except Exception as e:
        return f"❌ Hata: {str(e)}", None, None, f"Hata: {str(e)}"

def get_learning_stats_gr(system_state):
    """Öğrenme istatistikleri"""
    if system_state is None:
        return "❌ Sistem başlatılmadı."
    
    if not system_state.learning_data:
        return "📊 Henüz öğrenme verisi yok."
    
    total = len(system_state.learning_data)
    labeled = [d for d in system_state.learning_data if d.get('true_label') is not None]
    correct = [d for d in labeled if d.get('correct') == True]
    accuracy = (len(correct) / len(labeled) * 100) if labeled else 0
    
    output = "=" * 60 + "\n🧠 ÖĞRENME İSTATİSTİKLERİ\n" + "=" * 60 + "\n\n"
    output += f"📊 Toplam Analiz: {total}\n"
    output += f"🏷️ Etiketli: {len(labeled)}\n"
    output += f"✅ Doğru: {len(correct)}\n"
    output += f"❌ Yanlış: {len(labeled) - len(correct)}\n"
    output += f"📈 Doğruluk: {accuracy:.2f}%\n\n"
    
    recent_with_lessons = [d for d in system_state.learning_data[-10:] if d.get('lesson') and d['lesson'] != "YOK"][-5:]
    if recent_with_lessons:
        output += "-" * 60 + "\n📚 SON ÖĞRENİLENLER\n" + "-" * 60 + "\n"
        for i, d in enumerate(recent_with_lessons, 1):
            output += f"\n{i}. {d['file_name']}\n   {d['lesson']}\n"
    
    return output

def reset_learning_db_gr(system_state):
    """Veritabanını sıfırla"""
    if system_state is None:
        return "❌ Sistem başlatılmadı!"
    try:
        system_state.learning_data = []
        system_state.save_learning_data()
        return "✅ Öğrenme veritabanı sıfırlandı!"
    except Exception as e:
        return f"❌ Hata: {str(e)}"

# ==================== GRADIO ARAYÜZÜ ====================

def create_gradio_interface_demo():
    with gr.Blocks(title="PD Tespit Demo", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🔌 Kısmi Boşalma (PD) Tespit Sistemi - DEMO
        ### Gemini Flash 2.0 ile Akıllı Analiz
        
        **Demo Modu:** Dosya yollarını yazarak test yapabilirsiniz (dosya yükleme gerekmez)
        """)
        
        system_state = gr.State(None)
        
        # API Key
        with gr.Row(variant="panel"):
            with gr.Column(scale=3):
                # .env'den oku, yoksa boş
                default_key = os.getenv('GEMINI_API_KEY', '')
                api_key_input = gr.Textbox(
                    label="🔑 Gemini API Key",
                    placeholder="AIza... (veya .env dosyasında GEMINI_API_KEY tanımlayın)",
                    type="password",
                    value=default_key
                )
                if default_key:
                    gr.Markdown("✅ **.env dosyasından API key yüklendi**")
                else:
                    gr.Markdown("⚠️ **.env dosyasında GEMINI_API_KEY bulunamadı. Lütfen manuel girin.**")
            with gr.Column(scale=1):
                init_btn = gr.Button("🚀 Başlat", variant="primary")
        
        init_output = gr.Textbox(label="Sistem Durumu", lines=10, interactive=False)
        
        init_btn.click(
            fn=initialize_system_gr,
            inputs=[api_key_input],
            outputs=[init_output, system_state]
        )
        
        with gr.Tabs():
            # Tab 1: Tek Analiz
            with gr.Tab("📄 Tek Dosya Analizi"):
                gr.Markdown("""
                ### 📝 Dosya Yolu Örnekleri:
                
                **🎓 Eğitim Dosyaları:**
                - `pd_values_for_training/500927.npy`
                - `pd_values_for_training/500945.npy`
                - `pd_values_for_training/501057.npy`
                - `pd_values_for_training/501084.npy`
                
                **🔴 PD Var:**
                - `npy_examples/pd/575454.npy`
                - `npy_examples/pd/575923.npy`
                
                **🟢 PD Yok:**
                - `npy_examples/no_pd/576415.npy`
                - `npy_examples/no_pd/730183.npy`
                - `npy_examples/no_pd/730270.npy`
                
                **📊 CSV Dosyaları:**
                - PD: `csv_examples/scenario_PD/A1_1k_5M_Average.csv`
                - Normal: `csv_examples/scenario_no_PD/A1_1k_5M_Average.csv`
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        file_path_input = gr.Textbox(
                            label="📁 Dosya Yolu",
                            placeholder="npy_examples/pd/575454.npy",
                            lines=1
                        )
                        true_label_radio = gr.Radio(
                            choices=["Bilinmiyor", "PD YOK", "PD VAR"],
                            label="🏷️ Gerçek Etiket (öğrenme için)",
                            value="Bilinmiyor"
                        )
                        analyze_btn = gr.Button("🔍 Analiz Et", variant="primary")
                        
                        gr.Markdown("""
                        **💡 İpucu:** Yukarıdaki örnek yollardan birini kopyalayıp yapıştırın.
                        Veya kendi dosya yolunuzu yazın.
                        """)
                    
                    with gr.Column(scale=2):
                        single_output = gr.Textbox(label="📊 Analiz Sonucu", lines=20, interactive=False)
                
                single_plot = gr.Plot(label="📈 Spektrum Grafiği")
                
                with gr.Row():
                    csv_output = gr.Textbox(label="💾 CSV", lines=5, interactive=False)
                    preview_output = gr.Textbox(label="👁️ Veri Önizleme", lines=5, interactive=False)
                
                analyze_btn.click(
                    fn=analyze_from_path_gr,
                    inputs=[file_path_input, true_label_radio, system_state],
                    outputs=[single_output, single_plot, csv_output, preview_output]
                )
            
            # Tab 2: Öğrenme
            with gr.Tab("🧠 Öğrenme İstatistikleri"):
                gr.Markdown("""
                Bu sekmede, yaptığınız analizlerden elde edilen öğrenme verilerini görebilirsiniz.
                """)
                
                with gr.Row():
                    refresh_btn = gr.Button("🔄 İstatistikleri Yenile", variant="secondary")
                    reset_btn = gr.Button("🗑️ Sıfırla", variant="stop")
                
                stats_output = gr.Textbox(label="📊 Öğrenme Veritabanı", lines=25, interactive=False)
                
                refresh_btn.click(
                    fn=get_learning_stats_gr,
                    inputs=[system_state],
                    outputs=[stats_output]
                )
                
                reset_btn.click(
                    fn=reset_learning_db_gr,
                    inputs=[system_state],
                    outputs=[stats_output]
                )
        
        gr.Markdown("""
        ---
        ### 📚 Demo Dosya Yapısı
        
        Projenizde şu klasör yapısını oluşturun:
        
        ```
        project/
        ├── .env (GEMINI_API_KEY=AIza...)
        ├── pd_detection_demo.py
        ├── pd_values_for_training/
        │   ├── 500927.npy
        │   ├── 500945.npy
        │   ├── 501057.npy
        │   └── 501084.npy
        ├── npy_examples/
        │   ├── pd/
        │   │   ├── 575454.npy
        │   │   └── 575923.npy
        │   └── no_pd/
        │       ├── 576415.npy
        │       ├── 730183.npy
        │       └── 730270.npy
        └── csv_examples/
            ├── scenario_PD/
            │   └── A1_1k_5M_Average.csv
            └── scenario_no_PD/
                └── A1_1k_5M_Average.csv
        ```
        
        ### 🔑 .env Dosyası Oluşturma
        
        Proje klasörünüzde `.env` dosyası oluşturun:
        
        ```bash
        # Linux/Mac
        echo "GEMINI_API_KEY=AIza_sizin_key_buraya" > .env
        
        # Windows (PowerShell)
        echo "GEMINI_API_KEY=AIza_sizin_key_buraya" | Out-File -Encoding UTF8 .env
        
        # Manuel
        # .env adında dosya oluşturun ve içine şunu yazın:
        # GEMINI_API_KEY=AIza_sizin_key_buraya
        ```
        
        **Not:** API Key'i [Google AI Studio](https://aistudio.google.com/apikey)'dan ücretsiz alabilirsiniz.
        
        ### 💡 Kullanım İpuçları
        - **API Key:** .env dosyasında tanımlayın veya arayüzden girin
        - **Dosya Yolları:** Yukarıdaki örnekleri kopyalayıp yapıştırın
        - **Eğitim Dosyaları:** Modeli eğitmek için `pd_values_for_training/` klasöründeki dosyaları kullanın
        - **Test Dosyaları:** PD tespiti için `npy_examples/` klasöründeki dosyaları test edin
        - **Etiketleme:** Gerçek etiketi işaretlerseniz model öğrenir ve doğruluk oranı hesaplanır
        - **Öğrenme:** Her analiz kaydedilir ve sonraki tahminlerde kullanılır
        
        ### 🚀 Hızlı Başlangıç
        
        ```bash
        # 1. Gerekli paketleri yükle
        pip install gradio google-generativeai numpy pandas matplotlib python-dotenv
        
        # 2. .env dosyası oluştur
        echo "GEMINI_API_KEY=sizin_key_buraya" > .env
        
        # 3. Programı çalıştır
        python pd_detection_demo.py
        
        # 4. Tarayıcıda aç
        # http://127.0.0.1:7860
        ```
        
        **Geliştirici:** Gemini Flash 2.0 + Python + Gradio (Demo Mode)
        """)
    
    return interface
def chat_with_bot(message, history):
    response = f"Sen dedin ki: {message}"
    return response

# Mevcut arayüzünün diğer kısımları (örnek olarak ekliyorum)
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🌐 Ana Sayfa")
    gr.Textbox(label="Veri Girişi", placeholder="Bir şey yaz...")
    gr.Button("Gönder")

    # --- Chatbot kısmı ---
    gr.Markdown("### 💬 Chatbot Alanı")
    chatbot = gr.ChatInterface(
        fn=chat_with_bot,
        title="Yapay Zeka Asistanı",
        chatbot=gr.Chatbot(height=400),
        textbox=gr.Textbox(placeholder="Bir şey sor..."),
        
    )
# ==================== MAIN ====================

if __name__ == "__main__":
    load_dotenv()
    print("\n🌐 Demo Gradio arayüzü oluşturuluyor...")
    print("📌 Kurulum: pip install gradio google-generativeai numpy pandas matplotlib python-dotenv")
    
    app = create_gradio_interface_demo()
    print("\n🚀 Arayüz başlatılıyor...")
    print("💡 Dosya yollarını yazarak test yapabilirsiniz!")
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )