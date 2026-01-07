"""
FAISS tabanlı vektör veritabanı yönetimi modülü.
"""
import faiss
import numpy as np
import pickle
import os
from typing import List, Tuple, Optional


class VectorDB:
    """FAISS kullanarak vektör indeksleme ve arama işlemlerini yöneten sınıf."""
    
    def __init__(self, embedding_dim: int):
        """
        VectorDB sınıfını başlatır.
        
        Args:
            embedding_dim: Embedding vektörlerinin boyutu
        """
        self.embedding_dim = embedding_dim
        # Cosine similarity için IndexFlatIP (Inner Product) kullanıyoruz
        # Çünkü vektörler normalize edilmiş, inner product = cosine similarity
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.image_paths: List[str] = []
    
    def add_vectors(self, vectors: np.ndarray, image_paths: List[str]):
        """
        Vektörleri indekse ekler.
        
        Args:
            vectors: Embedding vektörleri (shape: [n_vectors, embedding_dim])
            image_paths: Her vektöre karşılık gelen görsel yolları
        """
        if vectors.shape[0] != len(image_paths):
            raise ValueError("Vektör sayısı ile görsel yolu sayısı eşleşmiyor!")
        
        # Vektörleri float32'ye çevir (FAISS gereksinimi)
        vectors = vectors.astype(np.float32)
        
        # İndekse ekle
        self.index.add(vectors)
        self.image_paths.extend(image_paths)
        
        print(f"{len(image_paths)} görsel indekse eklendi. Toplam: {self.index.ntotal}")
    
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Sorgu vektörüne en yakın görselleri bulur.
        
        Args:
            query_vector: Sorgu embedding vektörü (shape: [embedding_dim])
            top_k: Döndürülecek sonuç sayısı
            
        Returns:
            (görsel_yolu, benzerlik_skoru) tuple'larının listesi
        """
        if self.index.ntotal == 0:
            print("Uyarı: İndeks boş!")
            return []
        
        # Query vektörünü uygun formata çevir
        query_vector = query_vector.astype(np.float32).reshape(1, -1)
        
        # FAISS ile arama yap
        # distances: cosine similarity skorları (1'e yakın = daha benzer)
        # indices: bulunan görsellerin indeksleri
        top_k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_vector, top_k)
        
        # Sonuçları formatla
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx >= 0:  # FAISS bulunamayan sonuçlar için -1 döndürür
                results.append((self.image_paths[idx], float(score)))
        
        return results
    
    def save(self, index_path: str, metadata_path: str):
        """
        İndeksi ve metadata'yı diske kaydeder.
        
        Args:
            index_path: FAISS indeksinin kaydedileceği yol (.faiss)
            metadata_path: Metadata'nın kaydedileceği yol (.pkl)
        """
        # FAISS indeksini kaydet
        faiss.write_index(self.index, index_path)
        
        # Metadata'yı kaydet (görsel yolları)
        metadata = {
            'image_paths': self.image_paths,
            'embedding_dim': self.embedding_dim
        }
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"İndeks kaydedildi: {index_path}")
        print(f"Metadata kaydedildi: {metadata_path}")
    
    @classmethod
    def load(cls, index_path: str, metadata_path: str) -> 'VectorDB':
        """
        Kaydedilmiş indeksi ve metadata'yı yükler.
        
        Args:
            index_path: FAISS indeksinin yolu (.faiss)
            metadata_path: Metadata'nın yolu (.pkl)
            
        Returns:
            Yüklenmiş VectorDB örneği
        """
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError("İndeks veya metadata dosyası bulunamadı!")
        
        # Metadata'yı yükle
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # VectorDB örneği oluştur
        vector_db = cls(metadata['embedding_dim'])
        
        # FAISS indeksini yükle
        vector_db.index = faiss.read_index(index_path)
        vector_db.image_paths = metadata['image_paths']
        
        print(f"İndeks yüklendi: {index_path}")
        print(f"Toplam {vector_db.index.ntotal} görsel")
        
        return vector_db
    
    def get_stats(self) -> dict:
        """İndeks hakkında istatistikler döndürür."""
        return {
            'total_vectors': self.index.ntotal,
            'embedding_dim': self.embedding_dim,
            'total_images': len(self.image_paths)
        }
