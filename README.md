# 🏀 NBA Fantezi Asistanı - RAG Chatbot

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?style=for-the-badge&logo=streamlit)](https://rag-retrieval-augmented-generation-zhp2pyuybghd7jfyrpwrjs.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**Akbank GenAI Bootcamp** için geliştirilmiş, NBA oyuncu istatistiklerini analiz eden akıllı sohbet robotu.

---

## 📋 İçindekiler

- [Proje Hakkında](#-proje-hakkında)
- [Özellikler](#-özellikler)
- [Teknolojiler](#-teknolojiler)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [Proje Mimarisi](#-proje-mimarisi)
- [Veri Seti](#-veri-seti)
- [Demo](#-demo)
- [Katkıda Bulunma](#-katkıda-bulunma)

---

## 🎯 Proje Hakkında

Bu proje, **Retrieval Augmented Generation (RAG)** mimarisini kullanarak NBA oyuncu istatistiklerini sorgulayabileceğiniz akıllı bir chatbot'tur. Kullanıcılar doğal dilde sorular sorabilir ve sistem, vektör veritabanından ilgili bilgileri çekerek Gemini AI ile akıllı cevaplar üretir.

### Problem
NBA fantasy oyuncuları ve takipçileri, oyuncu performanslarını analiz etmek için karmaşık veri setlerini manuel olarak incelemek zorunda kalıyor.

### Çözüm
RAG tabanlı chatbot ile kullanıcılar, doğal dilde sorular sorarak anında detaylı oyuncu istatistiklerine ulaşabiliyor.

---

## ✨ Özellikler

- 🤖 **Doğal Dil İşleme**: Türkçe ve İngilizce sorular anlayabilir
- 🔍 **Akıllı Arama**: FAISS vektör veritabanı ile hızlı ve doğru sonuçlar
- 📊 **Detaylı İstatistikler**: Sayı, ribaund, asist, blok ve daha fazlası
- 🎯 **Bağlam Odaklı Cevaplar**: Sadece mevcut verilere dayalı güvenilir bilgiler
- 💾 **Kaynak Gösterimi**: Her cevabın dayandığı kaynaklara erişim
- ⚡ **Gerçek Zamanlı**: Anlık sorgulama ve cevaplama
- 🌐 **Web Tabanlı**: Her yerden erişilebilir Streamlit arayüzü

---

## 🛠 Teknolojiler

### Core Technologies
- **LLM**: Google Gemini 1.5 Flash
- **Framework**: LangChain
- **Vector Database**: FAISS
- **Embeddings**: HuggingFace Sentence Transformers
- **UI**: Streamlit
- **Data Processing**: Pandas

### Model Detayları
- **Embedding Model**: `paraphrase-multilingual-MiniLM-L12-v2` (Türkçe destekli)
- **LLM Model**: `gemini-1.5-flash-latest`
- **Vector Store**: FAISS (Facebook AI Similarity Search)

---

## 📦 Kurulum

### Gereksinimler
- Python 3.8 veya üzeri
- Google API Key ([Buradan alın](https://aistudio.google.com/apikey))

### Adım 1: Repository'yi Klonlayın
```bash
git clone https://github.com/muzaffer2/RAG-Retrieval-Augmented-Generation.git
cd RAG-Retrieval-Augmented-Generation
```

### Adım 2: Sanal Ortam Oluşturun (Opsiyonel ama Önerilen)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate  # Windows
```

### Adım 3: Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt
```

### Adım 4: Veri Setini Hazırlayın
`nba_fantasy_dataset.csv` dosyasını proje ana dizinine yerleştirin.

### Adım 5: Uygulamayı Çalıştırın
```bash
streamlit run project.py
```

Uygulama otomatik olarak tarayıcınızda `http://localhost:8501` adresinde açılacaktır.

---

## 🚀 Kullanım

### 1. API Key Girişi
Sol taraftaki sidebar'dan Google API anahtarınızı girin.

### 2. Soru Sorma
Metin kutusuna sorunuzu yazın. Örnek sorular:

```
- "Jayson Tatum en son maçında kaç sayı attı?"
- "LeBron James'in ribaund istatistikleri nedir?"
- "En yüksek sayıyı hangi oyuncu attı?"
- "Boston Celtics oyuncularının performansı nasıl?"
```

### 3. Sonuçları İnceleme
- Ana cevabı okuyun
- "Kullanılan Kaynaklar" bölümünden detayları görün
- Farklı sorularla denemeye devam edin

---

## 🏗 Proje Mimarisi

### RAG Pipeline

```
┌─────────────────┐
│  Kullanıcı      │
│  Sorusu         │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│  Embedding (HuggingFace)    │
│  Çok Dilli Model            │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  FAISS Vector Search        │
│  Top-K İlgili Belgeler      │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  LangChain QA Chain         │
│  Bağlam + Soru              │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Gemini 1.5 Flash           │
│  Cevap Üretimi              │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Türkçe Detaylı Cevap       │
└─────────────────────────────┘
```

### Dosya Yapısı

```
RAG-Retrieval-Augmented-Generation/
│
├── project.py                 # Ana uygulama dosyası
├── requirements.txt           # Python bağımlılıkları
├── nba_fantasy_dataset.csv   # NBA oyuncu istatistikleri
├── README.md                  # Proje dokümantasyonu
└── .gitignore                # Git ignore kuralları
```

---

## 📊 Veri Seti

### Kaynak
NBA Fantasy istatistikleri CSV formatında saklanmaktadır.

### Veri Yapısı
Veri seti şu sütunları içerir:
- `Player`: Oyuncu adı
- `Tm`: Takım kodu
- `Data`: Maç tarihi
- `Opp`: Rakip takım
- `MP`: Oyun süresi (dakika)
- `PTS`: Sayı
- `TRB`: Toplam ribaund
- `AST`: Asist
- `STL`: Top çalma
- `BLK`: Blok
- `FG%`: Saha içi isabet yüzdesi

### Örnek Veri
```csv
Player;Tm;Data;Opp;MP;PTS;TRB;AST;STL;BLK;FG%
Jayson Tatum;BOS;2024-01-15;LAL;36;32;8;5;2;1;0.485
```

---

## 🎥 Demo

### Live Demo
👉 [Uygulamayı Deneyin](https://rag-retrieval-augmented-generation-zhp2pyuybghd7jfyrpwrjs.streamlit.app/)

### Ekran Görüntüleri

#### Ana Ekran
![Ana Ekran](https://via.placeholder.com/800x400?text=NBA+Fantezi+Asistani+Ana+Ekran)

#### Soru-Cevap Örneği
![Soru Cevap](https://via.placeholder.com/800x400?text=Soru+Cevap+Ornegi)

---

## 🔧 Gelişmiş Ayarlar

### Embedding Modelini Değiştirme
```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
```

### LLM Parametrelerini Ayarlama
```python
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest", 
    temperature=0.1,  # 0-1 arası (düşük = daha tutarlı)
    google_api_key=api_key
)
```

### Similarity Search Sonuç Sayısı
```python
docs = vector_store.similarity_search(user_question, k=5)  # k değerini ayarlayın
```

---

## 📈 Performans

- **Vektör İndeksleme**: ~1-2 dakika (ilk seferde)
- **Sorgu Süresi**: ~2-5 saniye
- **Doğruluk**: Veri setine bağlı olarak yüksek
- **Desteklenen Diller**: Türkçe, İngilizce

---

## 🐛 Bilinen Sorunlar & Çözümler

### Problem: "Quota exceeded" hatası
**Çözüm**: Yeni Google API key alın veya lokal embedding kullanın.

### Problem: Veri dosyası bulunamadı
**Çözüm**: `nba_fantasy_dataset.csv` dosyasının proje kök dizininde olduğundan emin olun.

### Problem: Yavaş cevap üretimi
**Çözüm**: `k` değerini düşürün (örn: k=3) veya daha hızlı model kullanın.

---

## 🤝 Katkıda Bulunma

Katkılarınızı bekliyoruz! Lütfen şu adımları izleyin:

1. Fork'layın
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Commit'leyin (`git commit -m 'Add some AmazingFeature'`)
4. Push'layın (`git push origin feature/AmazingFeature`)
5. Pull Request açın

---

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

---

## 👨‍💻 Geliştirici

**Muzaffer**
- GitHub: [@muzaffer2](https://github.com/muzaffer2)
- Proje: [RAG-Retrieval-Augmented-Generation](https://github.com/muzaffer2/RAG-Retrieval-Augmented-Generation)

---

## 🙏 Teşekkürler

- **Akbank GenAI Bootcamp** - Proje desteği için
- **Google Gemini** - LLM API'si için
- **HuggingFace** - Embedding modelleri için
- **LangChain** - RAG framework'ü için
- **Streamlit** - UI framework'ü için

---

## 📚 Kaynaklar

- [Gemini API Docs](https://ai.google.dev/gemini-api/docs)
- [LangChain Documentation](https://python.langchain.com/)
- [FAISS Documentation](https://faiss.ai/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [RAG Nedir?](https://research.ibm.com/blog/retrieval-augmented-generation-RAG)

---

<div align="center">
  <p>⭐ Projeyi beğendiyseniz yıldız vermeyi unutmayın!</p>
  <p>Made with ❤️ for Akbank GenAI Bootcamp</p>
</div>