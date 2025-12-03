# Table-QA: Excel Tablosu Sorgulama Aracı

Excel dosyalarınızı doğal dil ile sorgulayabileceğiniz bir Python uygulaması.

## Özellikler

- Excel dosyalarını otomatik okuma ve analiz
- Türkçe doğal dil sorguları desteği
- LangChain ve Ollama (Mistral) ile AI destekli tablo analizi
- İnteraktif sorgulama arayüzü

## Gereksinimler

- Python 3.x
- Ollama (yerel olarak çalışıyor olmalı)
- Mistral modeli (`mistral:latest`)

## Kurulum

```bash
pip install pandas langchain-ollama langchain-experimental openpyxl
```

Ollama'yı başlatın ve Mistral modelini indirin:
```bash
ollama pull mistral:latest
```

## Kullanım

1. `bmsinav.xlsx` dosyasını `main2.py` ile aynı klasöre koyun
2. Scripti çalıştırın:
```bash
python main2.py
```

3. Türkçe sorularınızı sorun:
   - "En yüksek notu alan öğrenci kim?"
   - "Ortalama notu göster"
   - "Kaç öğrenci var?"

4. Çıkmak için `q` veya `exit` yazın.

## Notlar

- Script varsayılan olarak Excel dosyasının ilk 10 satırını okur
- Ollama'nın `http://localhost:11434` adresinde çalışıyor olması gerekir
- Verbose mod açık olduğu için AI'ın düşünme adımlarını görebilirsiniz

