"""
Text-to-Image Arama Sistemi - Ana Komut Satırı Arayüzü

Kullanım:
    1. Görselleri indeksleme:
       python main.py index --image-dir ./images --output-dir ./indices
    
    2. Metin ile arama yapma:
       python main.py search --query "a red car" --index-dir ./indices --top-k 5
"""
import argparse
import sys
import os
from image_search import ImageSearchEngine


def cmd_index(args):
    """Görselleri indeksleme komutu."""
    print("=" * 60)
    print("TEXT-TO-IMAGE ARAMA SİSTEMİ - İNDEKSLEME")
    print("=" * 60)
    
    # Görsel dizininin varlığını kontrol et
    if not os.path.exists(args.image_dir):
        print(f"❌ Hata: {args.image_dir} dizini bulunamadı!")
        sys.exit(1)
    
    # Arama motorunu başlat
    print(f"\nModel: {args.model}")
    engine = ImageSearchEngine(model_name=args.model, pretrained=args.pretrained)
    
    # Görselleri indeksle
    engine.index_images(args.image_dir)
    
    # İndeksi kaydet
    if args.output_dir:
        engine.save_index(args.output_dir)
        print(f"\n✓ İndeks kaydedildi: {args.output_dir}")
    
    # İstatistikleri göster
    stats = engine.get_stats()
    print("\n" + "=" * 60)
    print("İNDEKSLEME TAMAMLANDI")
    print("=" * 60)
    print(f"Toplam görsel: {stats['total_images']}")
    print(f"Embedding boyutu: {stats['embedding_dim']}")
    print("=" * 60)


def cmd_search(args):
    """Metin ile arama komutu."""
    print("=" * 60)
    print("TEXT-TO-IMAGE ARAMA SİSTEMİ - ARAMA")
    print("=" * 60)
    
    # İndeks dizininin varlığını kontrol et
    if not os.path.exists(args.index_dir):
        print(f"❌ Hata: {args.index_dir} dizini bulunamadı!")
        print("Önce 'index' komutu ile görselleri indekslemelisiniz.")
        sys.exit(1)
    
    # Arama motorunu başlat
    print(f"\nModel: {args.model}")
    engine = ImageSearchEngine(model_name=args.model, pretrained=args.pretrained)
    
    # İndeksi yükle
    engine.load_index(args.index_dir)
    
    # Arama yap
    print(f"\nSorgu: '{args.query}'")
    print(f"Top-K: {args.top_k}")
    print("\nArama yapılıyor...\n")
    
    results = engine.search(args.query, top_k=args.top_k)
    
    # Sonuçları göster
    print("=" * 60)
    print("ARAMA SONUÇLARI")
    print("=" * 60)
    
    if not results:
        print("Hiç sonuç bulunamadı!")
    else:
        for i, (image_path, score) in enumerate(results, 1):
            print(f"\n{i}. Benzerlik: {score:.4f}")
            print(f"   Yol: {image_path}")
    
    print("\n" + "=" * 60)


def main():
    """Ana fonksiyon - argümanları parse eder ve ilgili komutu çalıştırır."""
    parser = argparse.ArgumentParser(
        description="Text-to-Image Arama Sistemi",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Alt komutlar (subparsers)
    subparsers = parser.add_subparsers(dest='command', help='Komutlar')
    subparsers.required = True
    
    # INDEX komutu
    parser_index = subparsers.add_parser('index', help='Görselleri indeksle')
    parser_index.add_argument(
        '--image-dir',
        type=str,
        required=True,
        help='Görsellerin bulunduğu dizin'
    )
    parser_index.add_argument(
        '--output-dir',
        type=str,
        default='./indices',
        help='İndeksin kaydedileceği dizin (varsayılan: ./indices)'
    )
    parser_index.add_argument(
        '--model',
        type=str,
        default='ViT-B-32',
        help='OpenCLIP model adı (varsayılan: ViT-B-32)'
    )
    parser_index.add_argument(
        '--pretrained',
        type=str,
        default='openai',
        help='Pretrained model kaynağı (varsayılan: openai)'
    )
    parser_index.set_defaults(func=cmd_index)
    
    # SEARCH komutu
    parser_search = subparsers.add_parser('search', help='Metin ile görsel ara')
    parser_search.add_argument(
        '--query',
        type=str,
        required=True,
        help='Arama sorgusu (doğal dil)'
    )
    parser_search.add_argument(
        '--index-dir',
        type=str,
        default='./indices',
        help='İndeksin bulunduğu dizin (varsayılan: ./indices)'
    )
    parser_search.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Döndürülecek sonuç sayısı (varsayılan: 5)'
    )
    parser_search.add_argument(
        '--model',
        type=str,
        default='ViT-B-32',
        help='OpenCLIP model adı (varsayılan: ViT-B-32)'
    )
    parser_search.add_argument(
        '--pretrained',
        type=str,
        default='openai',
        help='Pretrained model kaynağı (varsayılan: openai)'
    )
    parser_search.set_defaults(func=cmd_search)
    
    # Argümanları parse et
    args = parser.parse_args()
    
    # İlgili komutu çalıştır
    args.func(args)


if __name__ == '__main__':
    main()
