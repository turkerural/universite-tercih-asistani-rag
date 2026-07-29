import os
import re
import glob
import sqlite3
import json
import sqlite_vec
from sentence_transformers import SentenceTransformer


def dosya_adindan_baslik_cikar(dosya_adi):
    baslik = dosya_adi.replace(".txt", "")
    baslik = baslik.replace("_", " ")
    return baslik


def metni_chunkla(metin, min_uzunluk=50, max_uzunluk=400):
    """Temel paragraf/cümle birleştirme mantığı (bölüm içeriğini parçalar)."""
    paragraflar = [p.strip() for p in metin.split("\n") if p.strip()]
    parcalar = []
    for p in paragraflar:
        if len(p) > max_uzunluk:
            cumleler = re.split(r'(?<=[.!?])\s+', p)
            parcalar.extend(cumleler)
        else:
            parcalar.append(p)

    chunklar = []
    mevcut_chunk = ""
    for parca in parcalar:
        if len(mevcut_chunk) + len(parca) < max_uzunluk:
            mevcut_chunk += " " + parca
        else:
            if mevcut_chunk.strip():
                if len(mevcut_chunk.strip()) < min_uzunluk and chunklar:
                    chunklar[-1] += " " + mevcut_chunk.strip()
                else:
                    chunklar.append(mevcut_chunk.strip())
            mevcut_chunk = parca
    if mevcut_chunk.strip():
        if len(mevcut_chunk.strip()) < min_uzunluk and chunklar:
            chunklar[-1] += " " + mevcut_chunk.strip()
        else:
            chunklar.append(mevcut_chunk.strip())
    return chunklar


def genel_dosya_chunkla(metin, baslik_adi, min_uzunluk=50, max_uzunluk=400):
    """Metni basitçe paragraf/cümle bazlı chunklar, bölüm başlığı
    tespiti veya etiketleme yapmaz."""
    return metni_chunkla(metin, min_uzunluk, max_uzunluk)


def satir_satir_chunkla(metin):
    return [p.strip() for p in metin.split("\n") if p.strip()]


def yokatlas_satirini_yapilandir(satir):
    """YÖK Atlas satırındaki serbest metni, taban puan/sıralama/kontenjan gibi
    alanları AÇIKÇA etiketleyerek yeniden yazar. Amaç: 'taban puan' ve
    'sıralama' gibi birbirine yakın konumdaki iki farklı sayının, LLM
    tarafından karıştırılmasını önlemek — her sayı artık kendi etiketiyle
    birlikte geliyor.

    Regex herhangi bir alanı bulamazsa o alan atlanır (chunk kaybolmaz),
    ve orijinal satır her zaman sonunda saklanır — böylece regex'in
    yakalayamadığı ek bilgiler (akademik kadro, şehir vb.) de kaybolmaz."""

    def bul(desen, varsayilan=None):
        eslesme = re.search(desen, satir)
        return eslesme.group(1).strip() if eslesme else varsayilan

    universite = bul(r'^(.*?\([^)]*(?:üniversite|Üniversite)[^)]*\))\s+bünyesindeki')
    fakulte = bul(r'bünyesindeki\s+(.*?)\s+altında yer alan')
    program = bul(r'altında yer alan\s+(.*?)\s+programı,')
    taban_puan = bul(r'([\d.]+)\s+taban puan')
    siralama = bul(r'([\d.]+)\.\s*sıralama')
    kontenjan = bul(r'kontenjanı\s+(\d+)\s+kişidir')
    yerlesen = bul(r'(\d+)\s+kişi yerleşmiştir')
    egitim_suresi = bul(r'eğitim süresi\s+(\d+)\s+yıldır')
    egitim_dili = bul(r'eğitim dili\s+(.+?)dir\.')
    puan_turu = bul(r'Program\s+(.+?)\s+puan türü ile')
    ogrenim_turu = bul(r'Öğrenim türü\s+(.+?)dir\.')

    if not taban_puan or not siralama:
        return satir

    alanlar = []
    if universite:
        alanlar.append(f"Üniversite: {universite}")
    if fakulte:
        alanlar.append(f"Fakülte: {fakulte}")
    if program:
        alanlar.append(f"Program: {program}")
    alanlar.append(f"Taban Puan: {taban_puan}")
    alanlar.append(f"Sıralama: {siralama}")
    if kontenjan:
        alanlar.append(f"Kontenjan: {kontenjan}")
    if yerlesen:
        alanlar.append(f"Yerleşen Öğrenci Sayısı: {yerlesen}")
    if egitim_suresi:
        alanlar.append(f"Eğitim Süresi: {egitim_suresi} yıl")
    if egitim_dili:
        alanlar.append(f"Eğitim Dili: {egitim_dili}")
    if puan_turu:
        alanlar.append(f"Puan Türü: {puan_turu}")
    if ogrenim_turu:
        alanlar.append(f"Öğrenim Türü: {ogrenim_turu}")

    yapilandirilmis = " | ".join(alanlar)
    return f"{yapilandirilmis}\n\nOrijinal metin: {satir}"


OZEL_CHUNK_DOSYALARI = [
    "yokatlas_tum_bolumler_2025.txt",
]


print("Embedding modeli yükleniyor (e5-large)...")
model = SentenceTransformer('intfloat/multilingual-e5-large')

conn = sqlite3.connect("universite_asistani.db")
conn.enable_load_extension(True)
sqlite_vec.load(conn)
conn.enable_load_extension(False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kaynak_dosya TEXT,
    metin TEXT,
    embedding TEXT
)
""")
cursor.execute("""
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
    embedding FLOAT[1024] distance_metric=cosine
)
""")
conn.commit()

txt_dosyalari = glob.glob("data/*.txt")
toplam_chunk = 0

for dosya_yolu in txt_dosyalari:
    dosya_adi = os.path.basename(dosya_yolu)
    with open(dosya_yolu, "r", encoding="utf-8") as f:
        icerik = f.read()

    baslik_adi = dosya_adindan_baslik_cikar(dosya_adi)

    if dosya_adi in OZEL_CHUNK_DOSYALARI:
        chunklar = [yokatlas_satirini_yapilandir(satir) for satir in satir_satir_chunkla(icerik)]
    else:
        chunklar = genel_dosya_chunkla(icerik, baslik_adi)

    print(f"{dosya_adi}: {len(chunklar)} parçaya bölündü")

    for chunk in chunklar:
        embedding = model.encode("passage: " + chunk).tolist()
        embedding_json = json.dumps(embedding)

        cursor.execute(
            "INSERT INTO chunks (kaynak_dosya, metin, embedding) VALUES (?, ?, ?)",
            (dosya_adi, chunk, embedding_json)
        )
        yeni_id = cursor.lastrowid

        cursor.execute(
            "INSERT INTO chunks_vec (rowid, embedding) VALUES (?, ?)",
            (yeni_id, embedding_json)
        )
        toplam_chunk += 1

conn.commit()
conn.close()

print(f"\n✅ Tamamlandı! Toplam {toplam_chunk} parça, universite_asistani.db dosyasına kaydedildi.")