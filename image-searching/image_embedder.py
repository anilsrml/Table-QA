"""
OpenCLIP kullanarak görsel ve metin embedding'leri üreten modül.
"""
import torch
import open_clip
from PIL import Image
import numpy as np
from typing import List, Union


class ImageEmbedder:
    """OpenCLIP modelini kullanarak görsel ve metin embedding'leri üretir."""
    
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai"):
        """
        ImageEmbedder sınıfını başlatır.
        
        Args:
            model_name: OpenCLIP model adı (varsayılan: ViT-B-32)
            pretrained: Pretrained model kaynağı (varsayılan: openai)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Cihaz: {self.device}")
        
        # OpenCLIP modelini ve önişlemciyi yükle
        print(f"Model yükleniyor: {model_name} ({pretrained})...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Tokenizer'ı yükle
        self.tokenizer = open_clip.get_tokenizer(model_name)
        print("Model başarıyla yüklendi!")
    
    def encode_image(self, image_path: str) -> np.ndarray:
        """
        Bir görseli embedding vektörüne dönüştürür.
        
        Args:
            image_path: Görsel dosyasının yolu
            
        Returns:
            Normalize edilmiş embedding vektörü (numpy array)
        """
        # Görseli yükle ve önişle
        image = Image.open(image_path).convert("RGB")
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Embedding üret
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            # Normalize et (cosine similarity için)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy()[0]
    
    def encode_images_batch(self, image_paths: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Birden fazla görseli batch halinde embedding vektörlerine dönüştürür.
        
        Args:
            image_paths: Görsel dosyalarının yolları listesi
            batch_size: Batch boyutu (varsayılan: 32)
            
        Returns:
            Normalize edilmiş embedding vektörleri (numpy array, shape: [n_images, embedding_dim])
        """
        all_features = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            
            # Batch'teki görselleri yükle
            for img_path in batch_paths:
                try:
                    image = Image.open(img_path).convert("RGB")
                    batch_images.append(self.preprocess(image))
                except Exception as e:
                    print(f"Uyarı: {img_path} yüklenirken hata: {e}")
                    continue
            
            if not batch_images:
                continue
            
            # Batch tensor oluştur
            batch_tensor = torch.stack(batch_images).to(self.device)
            
            # Embedding üret
            with torch.no_grad():
                image_features = self.model.encode_image(batch_tensor)
                # Normalize et
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            all_features.append(image_features.cpu().numpy())
        
        return np.vstack(all_features) if all_features else np.array([])
    
    def encode_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Bir veya birden fazla metni embedding vektörüne dönüştürür.
        
        Args:
            text: Metin veya metinler listesi
            
        Returns:
            Normalize edilmiş embedding vektörü(leri) (numpy array)
        """
        # Tek metin ise liste yap
        if isinstance(text, str):
            text = [text]
        
        # Metni tokenize et
        text_tokens = self.tokenizer(text).to(self.device)
        
        # Embedding üret
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            # Normalize et
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        embeddings = text_features.cpu().numpy()
        
        # Tek metin ise tek vektör döndür
        return embeddings[0] if len(text) == 1 else embeddings
    
    def get_embedding_dim(self) -> int:
        """Embedding vektörünün boyutunu döndürür."""
        dummy_text = self.tokenizer(["test"]).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(dummy_text)
        return features.shape[-1]
