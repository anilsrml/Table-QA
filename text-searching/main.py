# -*- coding: utf-8 -*-
import pandas as pd
from langchain_ollama import ChatOllama
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# --- 1. FARAZİ VERİ SETİNİ OLUŞTURMA ---
data = {
    'Personel': ['Ayşe', 'Burak', 'Can', 'Ayşe', 'Burak', 'Can', 'Deniz'],
    'Departman': ['IT', 'Satış', 'Satış', 'Muhasebe', 'IT', 'Satış', 'Muhasebe'],
    'Maaş': [50000, 45000, 42000, 35000, 52000, 48000, 40000],
    'Şehir': ['Bursa', 'İstanbul', 'İzmir', 'Bursa', 'İstanbul', 'Ankara', 'Bursa'],
    'Satış_Adedi': [5, 12, 18, 2, 7, 20, 3] 
}

df = pd.DataFrame(data)

print("=" * 60)
print("FARAZİ VERİ SETİ (Şirket Personel Bilgileri)")
print(df.to_string(index=False))
print("=" * 60)

# --- 2. LLM KURULUMU ---
print("\n>>> Ollama Llama3.2 modeli yukleniyor...")

llm = ChatOllama(
    model="mistral:latest",
    temperature=0,
    num_ctx=8192
)

# --- 3. AJANI (AGENT) OLUŞTURALIM ---
# Pandas Agent yapılandırması
agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,
    allow_dangerous_code=True,
    
    # Hata mesajına göre güncellenen agent tipi:
    agent_type="zero-shot-react-description",
    
    # Warning mesajına göre güncellenen hata yönetimi:
    agent_executor_kwargs={"handle_parsing_errors": True}
)

print(">>> Model basariyla yuklendi!\n")

# --- 4. DOĞAL DİL SORGULARI ---
print("=" * 60)
print("SORGULAMA BAŞLIYOR")
print("=" * 60)

sorular = [
    "Şehri 'Bursa' olan personellerin ortalama maaşını hesapla.",
    "Satış Adedi en yüksek olan personelin adını ve maaşını söyle.",
    "Şehri 'İstanbul' olan personellerin isimlerini listele."
]

for i, soru in enumerate(sorular, 1):
    print(f"\nSORU {i}: {soru}")
    print("-" * 30)
    
    try:
        result = agent.invoke({"input": soru})
        cevap = result.get('output', 'Cevap bulunamadı')
        print(f">>> CEVAP: {cevap}")
        
    except Exception as e:
        print(f"[HATA]: {e}")

print("\nIslem tamamlandi!")