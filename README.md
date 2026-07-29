# 🎓 Üniversite Bölüm Asistanı

Türkiye'deki üniversiteler ve tüm lisans bölümleri hakkında, **internet gerekmeden**, tamamen yerel olarak çalışan bir RAG (Retrieval-Augmented Generation) tabanlı yapay zekâ asistanı.

Bu proje, **Microsoft Foundry Local** kullanılarak sıfırdan geliştirilmiş bir öğrenme stajı çıktısıdır — LangChain gibi hazır kütüphaneler kullanılmadan, embedding, vektör arama, hibrit skorlama, query routing, reranking ve prompt mühendisliği adımlarının her biri elle inşa edilmiştir.

---

## ✨ Özellikler

- **%100 offline çalışma** — internet bağlantısı gerektirmez, tüm modeller ve veri yerel makinede
- **2025 YKS verisi** — taban puan, sıralama, kontenjan, burs durumu (Burslu/Ücretli/%50 İndirimli), eğitim dili, akademik kadro gibi ayrıntılı bilgiler
- **Genel üniversite bilgisi** — kuruluş tarihi, yerleşke/kampüs konumları, akademik birimler
- **Kod seviyesinde doğruluk garantileri** — üniversite/program adı karışmasını önleyen deterministik filtreler; prompt talimatlarına değil, Python mantığına dayanan güvenlik katmanları
- **Query routing** — soru tipine (sayısal veri mi, genel bilgi mi) göre farklı retrieval stratejisi, halüsinasyon riskini azaltır
- **Streamlit arayüzü** — glassmorphism (buzlu cam) tasarımlı, koyu/açık tema destekli sohbet arayüzü, canlı donanım (RAM/CPU/GPU) izleme paneli

---

## 🧠 Mimari — Nasıl Çalışıyor

```
Kullanıcı Sorusu
       │
       ▼
[1] Ön işleme — Türkçe normalize (turkce_normalize, turkce_kucuk_harf),
    kısaltma genişletme (İTÜ → İstanbul Teknik Üniversitesi vb.)
       │
       ▼
[2] Vektör arama — sqlite-vec ile e5-large embedding'leri üzerinden
    cosine benzerliğine göre ilk adaylar bulunur
       │
       ▼
[3] Hibrit skorlama — embedding benzerliği (%45) + anahtar kelime
    eşleşmesi (%10) + özel isim eşleşmesi (%45)
       │
       ▼
[4] Query routing (soru_tipini_belirle) — soru SAYISAL / GENEL /
    BELİRSİZ olarak sınıflandırılır
       │
       ▼
[5] Deterministik filtreler
    • Üniversite adı çözümleme (hedef_universite_coz) — isim-alt-kümesi
      karışıklığını (örn. "Ankara Üniversitesi" vs "Ankara Yıldırım
      Beyazıt Üniversitesi") önler
    • Program adı filtresi (program_adina_gore_filtrele) — yanlış
      bölümün karışmasını önler, aynı üniversitenin burslu/ücretli
      varyantlarını üniversite bazında ayrı değerlendirir
    • Hedefli ek veri çekme (hedef_universiteye_ait_ek_adaylar) — hedef
      üniversite netse, o üniversitenin verisini doğrudan SQL ile garantiye alır
    • GENEL sorularda YÖK Atlas verisi tamamen dışlanır — "kaç
      yerleşkesi var" gibi sorularda modelin "Kontenjan"/"Yerleşen
      Öğrenci Sayısı" alanlarını "yerleşke" ile karıştırıp halüsinasyon
      yapmasını (örn. "168 yerleşke var") kökten engeller
       │
       ▼
[6] Reranking (cross_encoder_ile_yeniden_sirala) — cross-encoder ile
    son 5 chunk seçilir
       │
       ▼
[7] LLM (Phi-4-mini, Foundry Local üzerinden) — context'e dayanarak,
    detaylı bir sistem promptu eşliğinde cevap üretir
```

### Neden bu kadar çok filtre var?

Yerel olarak çalışabilen küçük dil modelleri (Phi-4-mini gibi), context'te birden fazla benzer bilgi olduğunda bunları karıştırabiliyor. Bu proje boyunca, **prompt talimatlarına güvenmek yerine kod seviyesinde deterministik kontroller** kullanmanın çok daha güvenilir olduğu tekrar tekrar doğrulandı — bu yüzden kritik doğruluk gerektiren noktalarda Python tarafında kesin kurallar uygulanıyor, LLM sadece son adımda (temiz, filtrelenmiş context ile) doğal bir cevap üretiyor.

---

## 🛠️ Teknoloji Yığını

| Bileşen | Teknoloji | Çalıştığı Yer |
|---|---|---|
| Dil modeli (LLM) | Phi-4-mini-instruct-cuda-gpu (Microsoft Foundry Local) | GPU |
| Embedding modeli | intfloat/multilingual-e5-large | CPU |
| Reranker | cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 | CPU |
| Vektör veritabanı | SQLite + sqlite-vec eklentisi | — |
| Arayüz | Streamlit | — |

> **Not:** Embedding ve reranker modelleri bilinçli olarak CPU'da çalıştırılıyor. 8 GB VRAM'lik GPU'larda üç modelin (LLM + embedding + reranker) aynı anda GPU'yu paylaşması bellek taşmasına (`OnnxRuntimeGenAIException`) yol açıyordu; embedding/reranker'ı CPU'ya taşımak bu sorunu kalıcı olarak çözdü — GPU'yu sadece LLM kullanıyor.

---

## 📁 Proje Yapısı

```
├── app.py                 # Sohbet arayüzü (Streamlit, glassmorphism tasarım)
├── rag_asistan.py         # Retrieval + generation mantığının tamamı (RAG çekirdeği)
├── embed_ve_kaydet.py     # Ham veriyi chunk'layıp embedding'lerini hesaplayarak veritabanına yazar
├── logger.py              # Soru/cevap geçmişini kaydeden yardımcı modül
├── data/                  # Ham metin verisi (üniversite sayfaları + YÖK Atlas dökümü)
└── universite_asistani.db # SQLite veritabanı (embed_ve_kaydet.py çalıştırılınca oluşur, repoya dahil değil)
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

- **YÖK Atlas 2025** — tüm lisans programlarının taban puan, sıralama, kontenjan, burs ve akademik kadro bilgileri; `embed_ve_kaydet.py` bu veriyi ham cümle yerine `Üniversite: | Fakülte: | Program: | Taban Puan: | Sıralama: ...` şeklinde açıkça etiketlenmiş alanlara dönüştürerek işler (`yokatlas_satirini_yapilandir`) — bu, taban puan ile sıralama gibi birbirine yakın iki sayının modelde karışmasını önler.
- **Üniversite genel bilgi sayfaları** — kuruluş, tarihçe, yerleşke bilgileri
- **Bölüm tanım sayfaları** — "Bilgisayar Mühendisliği nedir?" gibi kavramsal sorulara cevap veren genel bilgi metinleri

> Veri setinde başlangıçta bulunan mevzuat (yönetmelik) verisi, toplam veri boyutunun büyük bir kısmını oluşturduğu ve embed süresini/cevap tutarlılığını olumsuz etkilediği için kaldırılmıştır.

---

## ⚠️ Bilinen Sınırlamalar

- **Aynı isimli farklı kampüsler** (örn. ODTÜ'nün Ankara ve Kıbrıs kampüsleri) bazen karışabiliyor
- **Query routing sezgisel bir yöntem** — anahtar kelime tabanlı sınıflandırma (`SAYISAL_SORU_KELIMELERI` / `GENEL_SORU_KELIMELERI`) kesin değil; karma sorular ("X Üniversitesi Y bölümü hangi kampüste?") yanlış sınıflanabilir
- **Retrieval her zaman doğru chunk'ı bulamayabiliyor** — bazı genel bilgi soruları (örn. "X Üniversitesi'ni kim kurdu?") veri setinde cevap olmasına rağmen bazen "bilgi bulunmuyor" cevabı alabiliyor; sistem bu durumda **halüsinasyon yapmak yerine reddetmeyi** tercih edecek şekilde tasarlandı
- **Liste tipi sorular** ("X üniversitesinde hangi bölümler var?") her zaman eksiksiz cevaplanamayabiliyor
- **Model rastgeleliği** — düşük de olsa bir `temperature` (0.2) kullanıldığı için aynı soru farklı ifadelerle sorulduğunda küçük farklılıklar olabiliyor

---

## 🗺️ Sonraki Adımlar

- [ ] Query routing sınıflandırıcısının kelime listesi yerine daha sağlam bir yöntemle (örn. küçük bir sınıflandırıcı) yapılması
- [ ] Genel bilgi sorularında retrieval kalitesinin artırılması (bazı doğru cevapların context'e hiç girmemesi sorunu)
- [ ] Liste tipi sorular için ayrı bir retrieval stratejisi
- [ ] YÖK'ün resmi istatistik portalından (öğrenci sayısı, akademik kadro) ek veri entegrasyonu

---

## 📌 Proje Hakkında

Bu proje, Rumeli Üniversitesi'nin 40 günlük Microsoft staj programı kapsamında geliştirilmiştir (Gün 1-20: RAG tabanlı offline soru-cevap asistanı).
