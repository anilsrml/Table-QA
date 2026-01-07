# Text-to-Image Arama Sistemi

OpenCLIP ve FAISS kullanarak görsel içeriğine dayalı text-to-image arama sistemi. Bu sistem, herhangi bir metadata kullanmadan sadece görsel içeriğine göre doğal dil sorguları ile görselleri arama imkanı sağlar.

## Özellikler

- ✅ Metadata kullanmadan görsel içeriğine dayalı arama
- ✅ OpenCLIP ile güçlü görsel ve metin embedding'leri
- ✅ FAISS ile hızlı vektör araması (cosine similarity)
- ✅ Komut satırı arayüzü
- ✅ Batch işleme desteği
- ✅ GPU ve CPU desteği

## Kurulum

### 1. Gerekli Kütüphaneleri Yükleyin

```bash
pip install -r requirements.txt
```

> **Not:** İlk çalıştırmada OpenCLIP modeli otomatik olarak indirilecektir (~1-2 GB).

## Kullanım

### Adım 1: Görselleri İndeksleme

Öncelikle `images` klasörünüzdeki görselleri indekslemelisiniz:

```bash
python main.py index --image-dir ./images --output-dir ./indices
```

**Parametreler:**
- `--image-dir`: Görsellerin bulunduğu dizin (zorunlu)
- `--output-dir`: İndeksin kaydedileceği dizin (varsayılan: `./indices`)
- `--model`: OpenCLIP model adı (varsayılan: `ViT-B-32`)
- `--pretrained`: Pretrained model kaynağı (varsayılan: `openai`)

**Çıktı:**
```
============================================================
TEXT-TO-IMAGE ARAMA SİSTEMİ - İNDEKSLEME
============================================================

Model: ViT-B-32
Cihaz: cuda
Model yükleniyor: ViT-B-32 (openai)...
Model başarıyla yüklendi!

./images dizini taranıyor...
Toplam 150 görsel bulundu.

Görseller indeksleniyor...
İndeksleme: 100%|██████████| 5/5 [00:15<00:00,  3.12s/it]

✓ Toplam 150 görsel başarıyla indekslendi!
İndeks kaydedildi: ./indices/index.faiss
Metadata kaydedildi: ./indices/metadata.pkl

✓ İndeks kaydedildi: ./indices

============================================================
İNDEKSLEME TAMAMLANDI
============================================================
Toplam görsel: 150
Embedding boyutu: 512
============================================================
```

### Adım 2: Metin ile Arama Yapma

İndeksleme tamamlandıktan sonra doğal dilde sorgu ile arama yapabilirsiniz:

```bash
python main.py search --query "sunset over the ocean" --index-dir ./indices --top-k 5
```

**Parametreler:**
- `--query`: Arama sorgusu - doğal dil (zorunlu)
- `--index-dir`: İndeksin bulunduğu dizin (varsayılan: `./indices`)
- `--top-k`: Döndürülecek sonuç sayısı (varsayılan: `5`)
- `--model`: OpenCLIP model adı - indekslemede kullanılan ile aynı olmalı (varsayılan: `ViT-B-32`)
- `--pretrained`: Pretrained model kaynağı (varsayılan: `openai`)

**Çıktı:**
```
============================================================
TEXT-TO-IMAGE ARAMA SİSTEMİ - ARAMA
============================================================

Model: ViT-B-32
Cihaz: cuda
Model yükleniyor: ViT-B-32 (openai)...
Model başarıyla yüklendi!
İndeks yüklendi: ./indices/index.faiss
Toplam 150 görsel

✓ İndeks başarıyla yüklendi!

Sorgu: 'sunset over the ocean'
Top-K: 5

Arama yapılıyor...

============================================================
ARAMA SONUÇLARI
============================================================

1. Benzerlik: 0.8542
   Yol: ./images/beach_sunset_01.jpg

2. Benzerlik: 0.8321
   Yol: ./images/ocean_view_03.jpg

3. Benzerlik: 0.8156
   Yol: ./images/sunset_beach_02.jpg

4. Benzerlik: 0.7989
   Yol: ./images/coastal_evening.jpg

5. Benzerlik: 0.7854
   Yol: ./images/sea_sunset.png

============================================================
```

## Örnek Sorgular

İşte deneyebileceğiniz bazı örnek sorgular:

```bash
# Belirli nesneler
python main.py search --query "a red car"
python main.py search --query "person walking a dog"
python main.py search --query "coffee cup on a table"

# Sahneler
python main.py search --query "modern building architecture"
python main.py search --query "forest with tall trees"
python main.py search --query "crowded city street"

# Aktiviteler
python main.py search --query "people playing basketball"
python main.py search --query "someone reading a book"

# Duygular ve atmosfer
python main.py search --query "peaceful mountain landscape"
python main.py search --query "cozy indoor space"
```

## Desteklenen Görsel Formatları

- JPG / JPEG
- PNG
- BMP
- GIF
- WEBP

## Sistem Gereksinimleri

- **Python:** 3.8+
- **RAM:** En az 8 GB (daha fazla görsel için daha fazla)
- **GPU:** Önerilir (CUDA destekli NVIDIA GPU), ancak CPU'da da çalışır
- **Disk Alanı:** Model için ~2 GB + görselleriniz

## Dosya Yapısı

```
image-searching/
├── main.py              # Ana CLI uygulaması
├── image_search.py      # Arama motoru
├── image_embedder.py    # OpenCLIP embedding üretici
├── vector_db.py         # FAISS vektör veritabanı
├── requirements.txt     # Gerekli kütüphaneler
├── images/              # Görsel veri setiniz (sizin oluşturmanız gerekiyor)
└── indices/             # Oluşturulan indeksler (otomatik oluşturulur)
    ├── index.faiss
    └── metadata.pkl
```

## Sorun Giderme

### GPU bulunamadı hatası
Sistem otomatik olarak CPU'ya geçecektir. GPU kullanmak için CUDA ve uyumlu PyTorch kurulumu gerekir.

### Model indirilemiyor
İnternet bağlantınızı kontrol edin. İlk çalıştırmada model otomatik indirilir.

### İndeks bulunamadı
Önce `index` komutu ile görselleri indekslediğinizden emin olun.

### Bellek yetersiz hatası
- Daha küçük batch size kullanın (kod içinde `batch_size` değişkenini düşürün)
- CPU kullanıyorsanız GPU'ya geçmeyi deneyin
- Daha az görsel ile başlayın

## Performans İpuçları

1. **GPU kullanın:** CUDA destekli GPU ile 10-20x daha hızlı çalışır
2. **İndeksi bir kere oluşturun:** Aynı veri seti için tekrar indekslemeye gerek yok
3. **Model seçimi:** `ViT-B-32` hız-kalite dengesi sunar, daha iyi sonuçlar için `ViT-L-14` deneyin

## Lisans

Bu proje eğitim amaçlıdır.
