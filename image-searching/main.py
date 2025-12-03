# -*- coding: utf-8 -*-
import os
import sys
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# --- IMPORT BLOĞU ---
try:
    from langchain.tools import tool
    from langchain_ollama import ChatOllama
    from langgraph.prebuilt import create_react_agent
    print("✓ Kütüphaneler başarıyla yüklendi")
except ImportError as e:
    print(f"[KRİTİK HATA] Kütüphane eksik: {e}")
    print("Çözüm: pip install langchain langchain-community langchain-ollama langgraph")
    sys.exit()

# --- 1. SİSTEM KURULUMU ---
print("\n" + "=" * 60)
print("GÖRSEL ARAMA SİSTEMİ - CLIP + MISTRAL AI")
print("=" * 60)

IMAGE_FOLDER = "C:/Users/anil6/Desktop/Table-QA/image-searching/images"
MODEL_ID = "openai/clip-vit-base-patch32"

# Klasör kontrolü
if not os.path.exists(IMAGE_FOLDER):
    print(f"[HATA] '{IMAGE_FOLDER}' klasörü bulunamadı!")
    sys.exit()

# CLIP modeli yükleme
print("\n>>> CLIP Modeli yükleniyor (ilk seferde ~30 sn sürer)...")
try:
    clip_model = CLIPModel.from_pretrained(MODEL_ID, use_safetensors=True)
    clip_processor = CLIPProcessor.from_pretrained(MODEL_ID)
    print("✓ CLIP Modeli hazır!")
except Exception as e:
    print(f"[HATA] Model yüklenemedi: {e}")
    sys.exit()

# --- 2. TOOL TANIMI ---
@tool
def resim_bul(query: str) -> str:
    """
    Kullanıcı bir görseli tarif ettiğinde (örneğin: 'kırmızı araba', 'deniz manzarası')
    bu aracı kullanarak klasördeki en uygun resmi bulursun.
    Girdi: Aranacak görselin tarifi (İngilizce veya Türkçe).
    Çıktı: Bulunan dosya adı ve güven skoru.
    """
    try:
        # Resimleri listele
        current_files = [f for f in os.listdir(IMAGE_FOLDER) 
                        if f.endswith(('.jpg', '.png', '.jpeg'))]
        if not current_files:
            return "Klasörde resim yok."

        # Resimleri yükle
        images = []
        valid_files = []
        for img_file in current_files:
            try:
                path = os.path.join(IMAGE_FOLDER, img_file)
                images.append(Image.open(path))
                valid_files.append(img_file)
            except:
                continue
        
        if not images:
            return "Resimler açılamadı."

        # CLIP ile analiz et
        inputs = clip_processor(text=[query], images=images, 
                               return_tensors="pt", padding=True)
        outputs = clip_model(**inputs)
        
        probs = outputs.logits_per_image.softmax(dim=0)
        idx = probs.argmax().item()
        score = probs[idx].item()
        
        return f"Bulunan Dosya: '{valid_files[idx]}' (Güven: %{score*100:.1f})"
    
    except Exception as e:
        return f"Hata: {e}"

# --- 3. AGENT KURULUMU ---
print("\n>>> Mistral AI ajanı hazırlanıyor...")

llm = ChatOllama(
    model="mistral:latest",
    temperature=0,
    num_ctx=4096
)

tools = [resim_bul]

# LangGraph ile agent oluştur (bu ÇALIŞIR)
agent = create_react_agent(llm, tools)

print("✓ Ajan hazır!\n")

# --- 4. SORGULAMA ---
print("=" * 60)
print("SORGULAMA BAŞLIYOR")
print("=" * 60)

# Örnek sorular - images klasörünüzdeki resimlere göre düzenleyin
sorular = [
    "Kırmızı renkli bir arabanın olduğu resmi bul",
    "Doğa manzarası veya ağaç içeren resmi getir",
    "Şehir veya bina fotoğrafı hangisi"
]

for i, soru in enumerate(sorular, 1):
    print(f"\n{'='*60}")
    print(f"SORU {i}: {soru}")
    print("="*60)
    
    try:
        # LangGraph stream kullanır
        events = list(agent.stream(
            {"messages": [("user", soru)]},
            stream_mode="values"
        ))
        
        # Son AI mesajını bul
        if events:
            last_event = events[-1]
            if "messages" in last_event:
                for msg in reversed(last_event["messages"]):
                    if hasattr(msg, 'content') and msg.content:
                        # Tool çağrılarını atla, sadece final cevabı göster
                        if hasattr(msg, 'type') and msg.type == 'ai':
                            if not msg.content.startswith('Bulunan Dosya:'):
                                print(f"\n✅ CEVAP: {msg.content}\n")
                                break
    
    except Exception as e:
        print(f"[HATA]: {e}")

print("\n" + "="*60)
print("İşlem tamamlandı!")
print("="*60)
