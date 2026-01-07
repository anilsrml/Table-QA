1. Proje Amacı

Bu projenin amacı, herhangi bir metadata (etiket, caption, açıklama) kullanmadan, yalnızca görsel içeriğine dayalı olarak bir görsel veri seti içinde metin ile görsel araması yapabilen bir sistem geliştirmektir.
Kullanıcı, doğal dilde bir metin girerek veri setindeki en ilgili görselleri bulabilmelidir.

2. Kapsam

Sistem yalnızca statik görseller üzerinde çalışacaktır.

Arama işlemi text → image yönünde olacaktır.

Görseller için manuel etiketleme veya açıklama zorunlu değildir.

Sistem offline / lokal çalışabilir.

3. Kullanılacak Teknolojiler

Programlama Dili: Python

Model: OpenCLIP 

Derin Öğrenme Kütüphanesi: PyTorch

Vektör Arama: FAISS

Görsel İşleme: Pillow (PIL)

Donanım: GPU

4. Fonksiyonel Gereksinimler

Sistem, veri setindeki her görsel için embedding vektörü üretmelidir.

Üretilen embedding’ler vektör veritabanında saklanmalıdır.

Kullanıcıdan alınan metin sorgusu embedding’e dönüştürülmelidir.

Metin embedding’i ile görsel embedding’leri arasında cosine similarity hesaplanmalıdır.

Sistem, en yüksek benzerliğe sahip ilk N görseli kullanıcıya döndürmelidir.

Görseller dosya yolu üzerinden gösterilmelidir.

5. Fonksiyonel Olmayan Gereksinimler

Sistem kolay anlaşılır ve modüler bir yapıda olmalıdır.

Aynı model hem text hem image embedding üretmelidir.

Arama süresi makul seviyede olmalıdır (küçük veri seti için saniyeler içinde).

Kod yapısı genişletilebilir olmalıdır.

6. Kısıtlar

Görseller için metadata, caption veya manuel açıklama kullanılmayacaktır.

Farklı embedding modelleri birlikte kullanılmayacaktır.

İnternet bağlantısı zorunlu değildir.

Sistem gerçek zamanlı video veya streaming desteklemez.

7. Beklenen Çıktılar

Text → image araması yapabilen çalışan bir Python uygulaması

OpenCLIP ile üretilmiş görsel embedding’ler

FAISS tabanlı vektör arama altyapısı

Örnek metin sorguları ile elde edilen görsel sonuçlar