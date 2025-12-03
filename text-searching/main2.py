# -*- coding: utf-8 -*-
import pandas as pd
import os
from langchain_ollama import ChatOllama
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# --- 1. EXCEL DOSYASINI OKUMA ---
dosya_adi = "bmsinav.xlsx"

print(f"\n>>> '{dosya_adi}' dosyasi okunuyor...")

if not os.path.exists(dosya_adi):
    print(f"[HATA] '{dosya_adi}' dosyasi bulunamadi! Lutfen dosyayi kodun yanina koyun.")
    exit()

try:
    # Excel dosyasını DataFrame'e çeviriyoruz
    df = pd.read_excel(dosya_adi).head(10)
    
    print(f">>> Dosya basariyla yuklendi! ({len(df)} satir veri var)")
    print("=" * 60)
    print("VERİ ÖNİZLEMESİ (İlk 5 Satır):")
    # Sütun isimlerini görmen için head() fonksiyonunu stringe çevirip basıyoruz
    print(df.to_string()) 
    print("=" * 60)

except Exception as e:
    print(f"[KRİTİK HATA] Excel dosyasi okunamadi: {e}")
    exit()

# --- 2. LLM KURULUMU ---

llm = ChatOllama(
    model="mistral:latest",
    temperature=0,
    num_ctx=8192,
    base_url="http://localhost:11434"  # Ollama varsayılan portu
)


print(f"\n>>> mistral:latest modeli hazirlaniyor...")

# --- 3. AJANI (AGENT) OLUŞTURALIM ---
# Hata almamak için "zero-shot-react-description" tipini kullanıyoruz
agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True, # Düşünme adımlarını görmek için açık kalsın
    allow_dangerous_code=True,
    agent_type="zero-shot-react-description",
    agent_executor_kwargs={"handle_parsing_errors": True}
)

print(">>> Sistem hazir! Çıkmak için 'q' veya 'exit' yazabilirsin.\n")

# --- 4. İNTERAKTİF SORGULAMA DÖNGÜSÜ ---
while True:
    print("-" * 60)
    user_input = input("SORU (Türkçe sorabilirsin): ")
    
    if user_input.lower() in ["q", "exit", "cikis", "çıkış"]:
        print("Sistemden cikiliyor...")
        break
    
    if not user_input.strip():
        continue

    print(f"\n>>> Analiz yapiliyor...")
    
    try:
        # invoke metodu ile soruyu gönderiyoruz
        response = agent.invoke({"input": user_input})
        
        # Sadece cevabı temiz bir şekilde yazdıralım
        cevap = response.get('output', str(response))
        print(f"\n✅ CEVAP: {cevap}\n")
        
    except Exception as e:
        print(f"\n❌ HATA: {e}\n")