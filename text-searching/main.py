# -*- coding: utf-8 -*-
import os
import sys
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# LangChain Importları (Sürüm Çakışmasını Önleyen Akıllı Blok)
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

# --- AKILLI IMPORT BLOĞU ---
# AgentExecutor ve create_react_agent farklı sürümlerde farklı yerlerde olabilir.
try:
    # Önce standart yeri deniyoruz (v0.1.0+)
    from langchain.agents import AgentExecutor, create_react_agent
except ImportError:
    try:
        # Bulamazsa alt modüllere bakıyoruz (v0.2.0+ veya eski sürümler)
        from langchain.agents.agent import AgentExecutor
        from langchain.agents.react.base import create_react_agent
    except ImportError:
        print("[KRİTİK HATA] LangChain kütüphanesi çok eski veya bozuk.")
        print("Lütfen terminale şunu yazın: pip install -U langchain langchain-community")
        sys.exit()

# --- 1. GÖRSEL VERİ ORTAMINI HAZIRLAMA ---
print("=" * 60)
print("SİSTEM KURULUMU: GÖRSEL VERİTABANI VE CLIP MODELİ")
print("=" * 60)

IMAGE_FOLDER = "images"
MODEL_ID = "openai/clip-vit-base-patch32"

# Klasör Kontrolü
if not os.path.exists(IMAGE_FOLDER):
    print(f"[HATA] '{IMAGE_FOLDER}' klasörü bulunamadı! Lütfen oluşturun ve içine resim koyun.")
    sys.exit()

image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(('.jpg', '.png', '.jpeg'))]
print(f">>> Klasörde {len(image_files)} adet görsel bulundu.")

# CLIP Modelini Yükleme
print(">>> CLIP Modeli (Görsel Algı Motoru) yükleniyor...")
try:
    # use_safetensors=True güvenlik hatasını önler
    clip_model = CLIPModel.from_pretrained(MODEL_ID, use_safetensors=True)
    clip_processor = CLIPProcessor.from_pretrained(MODEL_ID)
    print(">>> CLIP Modeli ve Veri Seti Hazır!")
except Exception as e:
    print(f"[KRİTİK HATA] Model yüklenemedi: {e}")
    sys.exit()

# --- 2. LLM KURULUMU ---
print("\n>>> Ollama Mistral modeli yukleniyor...")

llm = ChatOllama(
    model="mistral:latest",
    temperature=0,
    num_ctx=4096
)

# --- 3. AJANI (AGENT) OLUŞTURALIM ---

# A. Tool (Araç) Tanımlama
@tool
def resim_bul(query: str) -> str:
    """
    Kullanıcı bir görseli tarif ettiğinde (örneğin: 'kırmızı araba', 'koşan köpek', 'ofis masası')
    bu aracı kullanarak klasördeki en uygun resmi bulursun.
    Girdi (query): Aranacak görselin İngilizce veya Türkçe tarifi.
    Çıktı: Bulunan dosya adı ve güven skoru.
    """
    images = []
    valid_files = []
    try:
        # Klasördeki güncel dosyaları oku
        current_files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(('.jpg', '.png', '.jpeg'))]
        if not current_files: return "Klasörde resim yok."

        for img_file in current_files:
            try:
                path = os.path.join(IMAGE_FOLDER, img_file)
                images.append(Image.open(path))
                valid_files.append(img_file)
            except:
                continue
        
        if not images: return "Resimler okunamadı."

        # CLIP Analizi
        inputs = clip_processor(text=[query], images=images, return_tensors="pt", padding=True)
        outputs = clip_model(**inputs)
        
        probs = outputs.logits_per_image.softmax(dim=0)
        idx = probs.argmax().item()
        score = probs[idx].item()
        
        return f"Bulunan Dosya: '{valid_files[idx]}' (Güven Skoru: %{score*100:.1f})"
    except Exception as e:
        return f"Hata: {e}"

# B. Ajan Yapılandırması
tools = [resim_bul]

# ReAct Prompt Şablonu
template = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}'''

prompt = PromptTemplate.from_template(template)

# Ajanı Oluşturuyoruz
react_agent = create_react_agent(llm, tools, prompt)

# Ajan Yürütücüsü (Agent Executor)
# Pandas örneğindeki 'agent' değişkeni burada 'agent_executor' oluyor
agent_executor = AgentExecutor(
    agent=react_agent,
    tools=tools,
    verbose=True, # Düşünme adımlarını gösterir
    handle_parsing_errors=True # Hata olursa devam et
)

print(">>> Ajan başarıyla yapılandırıldı!\n")

# --- 4. DOĞAL DİL SORGULARI ---
print("=" * 60)
print("SORGULAMA BAŞLIYOR")
print("=" * 60)

# SORU LİSTESİ
# NOT: Buradaki soruları 'images' klasöründeki gerçek resimlerine göre güncellemelisin.
sorular = [
    "Kırmızı renkli bir arabanın olduğu resmi bul.",
    "İçinde bilgisayar veya ofis malzemeleri olan fotoğraf hangisi?",
    "Bana doğa manzarası veya ağaç içeren resmi getir."
]

for i, soru in enumerate(sorular, 1):
    print(f"\nSORU {i}: {soru}")
    print("-" * 30)
    
    try:
        # Soruyu ajana gönderiyoruz
        result = agent_executor.invoke({"input": soru})
        cevap = result.get('output', 'Cevap bulunamadı')
        print(f">>> CEVAP: {cevap}")
        
    except Exception as e:
        print(f"[HATA]: {e}")

print("\nIslem tamamlandi!")