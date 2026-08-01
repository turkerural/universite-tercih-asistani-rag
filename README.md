# 🎓 Üniversite Bölüm Asistanı

Türkiye'deki üniversiteler ve tüm lisans bölümleri hakkında, **internet gerekmeden**, tamamen yerel olarak çalışan bir RAG (Retrieval-Augmented Generation) tabanlı yapay zekâ asistanı.

Bu proje, **Microsoft Foundry Local** kullanılarak sıfırdan geliştirilmiş bir öğrenme stajı çıktısıdır — LangChain gibi hazır kütüphaneler kullanılmadan, embedding, vektör arama, hibrit skorlama, query routing, reranking ve prompt mühendisliği adımlarının her biri elle inşa edilmiştir.

---

## ✨ Özellikler

- **%100 offline çalışma** — internet bağlantısı gerektirmez, tüm modeller ve veri yerel makinede
- **2025 YKS verisi** — taban puan, sıralama, kontenjan, burs durumu, eğitim dili, akademik kadro gibi ayrıntılı bilgiler
- **Genel üniversite bilgisi** — kuruluş tarihi, kurucu, yerleşke/kampüs konumları
- **Kod seviyesinde doğruluk garantileri** — üniversite/program adı karışmasını ve halüsinasyonu önleyen deterministik filtreler
- **Query routing** — soru tipine (sayısal veri mi, genel bilgi mi, ikisi birden mi) göre farklı retrieval stratejisi
- **Karma (çok parçalı) soru desteği** — "X nerede, ne zaman kuruldu ve taban puanı kaç?" gibi soruları, hem genel hem sayısal kaynaktan garantili beslenerek cevaplar
- **Streamlit arayüzü** — glassmorphism tasarımlı, koyu/açık tema destekli sohbet arayüzü

---

## 🧠 Mimari — Nasıl Çalışıyor

```
Kullanıcı Sorusu
       │
       ▼
[1] Ön işleme — Türkçe normalize (â/î/û dahil), kısaltma genişletme
       │
       ▼
[2] Vektör arama — sqlite-vec ile e5-large embedding üzerinden cosine benzerliği
       │
       ▼
[3] Hibrit skorlama — embedding (%45) + anahtar kelime (%10) + özel isim (%45)
       │
       ▼
[4] Query routing — soru SAYISAL / GENEL / KARMA / BELİRSİZ olarak sınıflandırılır
       │
       ▼
[5] Deterministik filtreler ve SQL garantileri
    • Üniversite adı çözümleme (isim-alt-kümesi ve tireli isim karışıklığını önler)
    • Program adı filtresi (yanlış bölümün karışmasını önler, varyantları
      üniversite bazında ayrı değerlendirir)
    • Hedefli YÖK Atlas + GENEL bilgi dosyası SQL garantisi (embedding
      aramasının bulamadığı doğru chunk'ı garantiye alır)
    • GENEL/kavramsal sorularda YÖK Atlas verisi tamamen dışlanır
      (kelime kökü halüsinasyonunu — örn. 'kontenjan'ı 'yerleşke' sanma —
      kökten önler)
    • KARMA sorularda çeşitlilik garantili reranking (tek bir kaynak türü
      diğerini tamamen dışlayamaz)
       │
       ▼
[6] Reranking — cross-encoder ile son 5 chunk seçimi
       │
       ▼
[7] LLM (Phi-4-mini, Foundry Local üzerinden) — context'e dayanarak cevap üretir
```

### Neden bu kadar çok filtre var?

Yerel olarak çalışabilen küçük dil modelleri (Phi-4-mini gibi), context'te birden fazla benzer bilgi olduğunda bunları karıştırabiliyor ve elinde olmayan bilgiyi **uydurabiliyor** (halüsinasyon). Bu proje boyunca, **prompt talimatlarına güvenmek yerine kod seviyesinde deterministik kontroller** kullanmanın çok daha güvenilir olduğu defalarca doğrulandı — kritik doğruluk gerektiren noktalarda Python tarafında kesin kurallar uygulanıyor, LLM sadece son adımda, temiz ve garantili bir context ile doğal cümle kuruyor.

---

## 🛠️ Teknoloji Yığını

| Bileşen | Teknoloji | Çalıştığı Yer |
|---|---|---|
| Dil modeli (LLM) | Phi-4-mini-instruct-cuda-gpu (Microsoft Foundry Local) | GPU |
| Embedding modeli | intfloat/multilingual-e5-large | CPU |
| Reranker | cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 | CPU |
| Vektör veritabanı | SQLite + sqlite-vec eklentisi | — |
| Arayüz | Streamlit | — |

> **Not:** Embedding ve reranker modelleri bilinçli olarak CPU'da çalıştırılıyor. 8 GB VRAM'lik GPU'larda üç modelin aynı anda GPU'yu paylaşması bellek taşmasına (`OnnxRuntimeGenAIException`) yol açıyordu; embedding/reranker'ı CPU'ya taşımak bu sorunu kalıcı olarak çözdü.

---

## 📁 Proje Yapısı

```
├── app.py                        # Sohbet arayüzü (Streamlit, glassmorphism tasarım)
├── rag_asistan.py                # Retrieval + generation mantığının tamamı (RAG çekirdeği)
├── embed_ve_kaydet.py            # Ham veriyi chunk'layıp embedding'lerini hesaplayarak veritabanına yazar
├── logger.py                     # Soru/cevap geçmişini kaydeden yardımcı modül
├── data/                         # Ham metin verisi (üniversite sayfaları + YÖK Atlas dökümü)
├── test_sorulari_100*.md         # Veri setinden türetilmiş 100'er soruluk test setleri
└── universite_asistani.db        # SQLite veritabanı (embed_ve_kaydet.py çalıştırılınca oluşur, repoya dahil değil)
```

> Veri, Wikipedia ve YÖK Atlas'ın herkese açık API'leri kullanılarak toplanmıştır; veri toplama scriptleri bu repoda yer almıyor, sadece toplanmış çıktı (`data/`) paylaşılıyor.

---

## 🚀 Kurulum

### Gereksinimler
- Python 3.10+
- [Microsoft Foundry Local](https://aka.ms/foundry-local) kurulu olmalı
- CUDA destekli bir GPU (LLM için önerilir; embedding/reranker zaten CPU'da çalışıyor)

### Adımlar

```bash
# Bağımlılıkları kur
pip install -r requirements.txt

# Foundry Local'ı kur ve modeli indir
winget install Microsoft.FoundryLocal
foundry model run Phi-4-mini-instruct-cuda-gpu

# Veritabanını oluştur (data/ klasöründeki .txt dosyalarını işler)
python embed_ve_kaydet.py

# Arayüzü başlat
streamlit run app.py
```

---

## 📊 Veri Kaynakları

- **YÖK Atlas 2025** — tüm lisans programlarının taban puan, sıralama, kontenjan, burs ve akademik kadro bilgileri; `embed_ve_kaydet.py` bu veriyi ham cümle yerine `Üniversite: | Fakülte: | Program: | Taban Puan: | Sıralama: ...` şeklinde açıkça etiketlenmiş alanlara dönüştürerek işler.
- **Üniversite genel bilgi sayfaları** — kuruluş, tarihçe, yerleşke bilgileri
- **Bölüm tanım sayfaları** — "Bilgisayar Mühendisliği nedir?" gibi kavramsal sorulara cevap veren genel bilgi metinleri

> Veri setinde başlangıçta bulunan mevzuat (yönetmelik) verisi, embed süresini/cevap tutarlılığını olumsuz etkilediği için kaldırılmıştır.

---

## ⚠️ Bilinen Sınırlamalar

- **Aynı isimli farklı kampüsler** (örn. ODTÜ'nün Ankara ve Kıbrıs kampüsleri) bazen karışabiliyor
- **Query routing sezgisel bir yöntem** — anahtar kelime tabanlı sınıflandırma kesin değil; hiç görülmemiş ifade kalıpları yanlış sınıflanabilir
- **Kaynak veri bazen eksik** — bazı Wikipedia tabanlı bölüm dosyalarında listeler yarım kalmış olabilir (retrieval/kod sorunu değil, ham veri eksikliği)
- **Liste tipi sorular** ("X üniversitesinde hangi bölümler var?") her zaman eksiksiz cevaplanamayabiliyor

## ✅ Çözülen Kritik Hatalar (Geliştirme Süreci)

- Üniversite isim-alt-kümesi karışması (Ankara / Ankara Yıldırım Beyazıt) — deterministik çözümleme ile
- Tireyle birleşik isim çakışması (İstanbul Üniversitesi-Cerrahpaşa)
- Program adı karışması (aynı üniversitenin farklı mühendislik dalları)
- "Yerleşke" ile "Yerleşen Öğrenci Sayısı" kelime kökü halüsinasyonu
- Kavramsal ("nedir") sorulara YÖK Atlas verisinin sızması
- Çok parçalı (karma) sorularda tek kaynak türünün diğerini dışlaması
- GPU bellek taşması (embedding/reranker CPU'ya taşınarak)

---

## 📌 Proje Hakkında

Bu proje, Rumeli Üniversitesi'nin 40 günlük Microsoft staj programı kapsamında geliştirilmiştir (Gün 1-20: RAG tabanlı offline soru-cevap asistanı).