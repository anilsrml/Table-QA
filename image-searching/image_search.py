"""
Text-to-image arama motoru modülü.
ImageEmbedder ve VectorDB sınıflarını kullanarak görsel indeksleme ve arama yapar.
"""
import os
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm

from image_embedder import ImageEmbedder
from vector_db import VectorDB


class ImageSearchEngine:
    """Text-to-image arama motoru."""
    
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai"):
        """
        ImageSearchEngine sınıfını başlatır.
        
        Args:
            model_name: OpenCLIP model adı
            pretrained: Pretrained model kaynağı
        """
        self.embedder = ImageEmbedder(model_name, pretrained)
        self.vector_db = None
    
    def index_images(self, image_dir: str, supported_formats: List[str] = None) -> VectorDB:
        """
        Belirtilen dizindeki tüm görselleri indeksler.
        
        Args:
            image_dir: Görsellerin bulunduğu dizin
            supported_formats: Desteklenen görsel formatları (varsayılan: jpg, jpeg, png)
            
        Returns:
            Oluşturulan VectorDB nesnesi
        """
        if supported_formats is None:
            supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
        
        # Görsel dosyalarını bul
        image_paths = []
        print(f"\n{image_dir} dizini taranıyor...")
        
        for root, _, files in os.walk(image_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in supported_formats):
                    image_paths.append(os.path.join(root, file))
        
        if not image_paths:
            print(f"Uyarı: {image_dir} dizininde görsel bulunamadı!")
            return None
        
        print(f"Toplam {len(image_paths)} görsel bulundu.")
        
        # VectorDB oluştur
        embedding_dim = self.embedder.get_embedding_dim()
        self.vector_db = VectorDB(embedding_dim)
        
        # Görselleri batch halinde işle
        print("\nGörseller indeksleniyor...")
        batch_size = 32
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="İndeksleme"):
            batch_paths = image_paths[i:i + batch_size]
            
            # Batch embedding'leri üret
            embeddings = self.embedder.encode_images_batch(batch_paths, batch_size=batch_size)
            
            if len(embeddings) > 0:
                # VectorDB'ye ekle (sadece başarılı olan görseller)
                actual_paths = batch_paths[:len(embeddings)]
                self.vector_db.add_vectors(embeddings, actual_paths)
        
        print(f"\n✓ Toplam {self.vector_db.index.ntotal} görsel başarıyla indekslendi!")
        return self.vector_db
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Metin sorgusu ile görsel arama yapar.
        
        Args:
            query: Arama sorgusu (doğal dil)
            top_k: Döndürülecek sonuç sayısı
            
        Returns:
            (görsel_yolu, benzerlik_skoru) tuple'larının listesi
        """
        if self.vector_db is None:
            raise ValueError("Önce index_images() ile görselleri indekslemelisiniz!")
        
        # Sorgu için embedding üret
        query_embedding = self.embedder.encode_text(query)
        
        # Arama yap
        results = self.vector_db.search(query_embedding, top_k)
        
        return results
    
    def save_index(self, output_dir: str):
        """
        İndeksi diske kaydeder.
        
        Args:
            output_dir: İndeksin kaydedileceği dizin
        """
        if self.vector_db is None:
            raise ValueError("Kaydedilecek indeks bulunamadı!")
        
        # Çıktı dizinini oluştur
        os.makedirs(output_dir, exist_ok=True)
        
        # Dosya yolları
        index_path = os.path.join(output_dir, "index.faiss")
        metadata_path = os.path.join(output_dir, "metadata.pkl")
        
        # Kaydet
        self.vector_db.save(index_path, metadata_path)
    
    def load_index(self, index_dir: str):
        """
        Kaydedilmiş indeksi yükler.
        
        Args:
            index_dir: İndeksin bulunduğu dizin
        """
        index_path = os.path.join(index_dir, "index.faiss")
        metadata_path = os.path.join(index_dir, "metadata.pkl")
        
        self.vector_db = VectorDB.load(index_path, metadata_path)
        print("✓ İndeks başarıyla yüklendi!")
    
    def get_stats(self) -> dict:
        """İndeks hakkında istatistikler döndürür."""
        if self.vector_db is None:
            return {"status": "İndeks yüklü değil"}
        return self.vector_db.get_stats()
