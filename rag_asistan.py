import sqlite3
import json
import re
import numpy as np
import torch
import sqlite_vec
from sentence_transformers import SentenceTransformer, CrossEncoder
from foundry_local_sdk import Configuration, FoundryLocalManager


from logger import log_kaydet

print("Embedding modeli yükleniyor (e5-large)...")
embed_model = SentenceTransformer('intfloat/multilingual-e5-large', device='cpu')

print("Reranker modeli yükleniyor...")
reranker = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1', device='cpu')

print("Foundry Local'a bağlanılıyor...")
config = Configuration(app_name="universite_asistani")
FoundryLocalManager.initialize(config)
manager = FoundryLocalManager.instance

print("GPU hızlandırma bileşenleri kontrol ediliyor...")
result = manager.download_and_register_eps()
print(f"EP kayıt durumu -> Başarılı: {result.success}, Durum: {result.status}")

MODEL_ID = "Phi-4-mini-instruct-cuda-gpu:5"
print(f"Model yükleniyor: {MODEL_ID}")
model = manager.catalog.get_model_variant(MODEL_ID)
model.download()
model.load()

chat_client = model.get_chat_client()
chat_client.settings.temperature = 0.2
chat_client.settings.top_p = 0.9
chat_client.settings.max_tokens = 500

KULLAN_SQLITE_VEC = True
KULLAN_RERANKING = True


def turkce_normalize(metin):
    metin = metin.translate(str.maketrans({"İ": "i", "I": "ı"})).lower()
    donusum = str.maketrans("çğıöşü", "cgiosu")
    return metin.translate(donusum)


def turkce_kucuk_harf(metin):
    """Python'ın standart .lower() fonksiyonu, Türkçe'ye özgü noktalı büyük
    'İ' harfini YANLIŞ küçültür: 'İTÜ'.lower() -> 'i̇tü' (görünmez bir
    U+0307 karakteri ekler). Bu da 'itü' in 'i̇tü' kontrolünün SESSİZCE
    başarısız olmasına, dolayısıyla kısaltmanın hiç açılmamasına yol açar.
    Bu fonksiyon, 'İ' ve 'I' harflerini Türkçe kurallarına göre önce
    dönüştürüp sonra küçültür, bu sorunu önler."""
    donusum = str.maketrans({"İ": "i", "I": "ı"})
    return metin.translate(donusum).lower()


def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


KISALTMALAR = {
    "itü": "istanbul teknik üniversitesi",
    "odtü": "orta doğu teknik üniversitesi",
    "boğaziçi": "boğaziçi üniversitesi",
    "ytü": "yıldız teknik üniversitesi",
    "gtü": "gebze teknik üniversitesi",
    "ktü": "karadeniz teknik üniversitesi",
    "koç": "koç üniversitesi",
    "sabancı": "sabancı üniversitesi",
    "bilkent": "ihsan doğramacı bilkent üniversitesi",
}


def kisaltma_genislet(soru):
    soru_kucuk = turkce_kucuk_harf(soru)
    for kisa, uzun in KISALTMALAR.items():
        if kisa in soru_kucuk:
            soru_kucuk = soru_kucuk.replace(kisa, uzun)
    return soru_kucuk


def ozel_isimleri_bul(soru_orijinal):
    """Sorudaki 'özel isim' adaylarını (üniversite/bölüm adı gibi önemli
    terimleri) tespit eder.

    ÖNEMLİ: Bu fonksiyon önceden büyük harfle başlayan kelime olup
    olmamasına göre iki FARKLI moda geçiyordu (biri sadece büyük harfli
    kelimeleri, diğeri 4+ harfli TÜM kelimeleri seçiyordu) — bu da aynı
    soru sadece farklı büyük/küçük harfle yazıldığında (örn. 'Hemşirelik'
    vs 'hemşirelik') hibrit skorun ve dolayısıyla retrieval sonucunun
    tamamen değişmesine yol açıyordu. Artık kullanıcının yazım şekline
    bakılmaksızın HER ZAMAN aynı, tutarlı kelime kümesi kullanılıyor:
    büyük harfle başlayan kelimeler VE 4 harften uzun kelimelerin BİRLEŞİMİ."""
    kelimeler = [kelime.strip(".,?!():") for kelime in soru_orijinal.split()]

    buyuk_harfli = {
        turkce_normalize(k) for k in kelimeler if k[0:1].isupper() and len(k) > 2
    }
    uzun_kelimeler = {
        turkce_normalize(k) for k in kelimeler if len(k) > 4
    }

    return buyuk_harfli | uzun_kelimeler


def program_govdesi(program_adi):
    """'Bilgisayar Mühendisliği (Burslu)' -> 'bilgisayar mühendisliği'
    Sondaki parantezli ekleri (Burslu, İngilizce, %50 İndirimli vb.) atar."""
    temel = re.sub(r"\s*\([^)]*\)\s*", " ", program_adi).strip()
    return turkce_normalize(temel)


VARYANT_ANAHTAR_KELIMELERI = [
    "burslu", "ücretli", "indirimli", "ingilizce", "uolp", "kktc",
    "fransızca", "almanca", "arapça", "italyanca", "ispanyolca",
]


def program_adi_duz_mu(program_adi_ham):
    """'Hukuk' -> True, 'Hukuk (Burslu)' veya 'Hukuk (UOLP-...) (Ücretli)'
    -> False. Yani hiç parantez içeren ek yoksa 'düz/varsayılan' programdır."""
    return "(" not in program_adi_ham


def program_adina_gore_filtrele(soru, adaylar):
    """Adaylar arasından, YÖK Atlas kayıtlarında 'Program:' adı sorunun
    içinde TAM olarak (kelime kelime) geçenleri seçer. Hiç eşleşme yoksa
    adayları OLDUĞU GİBİ döner (filtre uygulanmaz).

    ÖNEMLİ: Wikipedia/genel bilgi dosyalarından (yokatlas OLMAYAN) gelen
    adaylar HİÇBİR ZAMAN elenmiyor — sadece YÖK Atlas alt kümesi
    filtreleniyor. Aksi halde 'Bilgisayar mühendisliği nedir?' gibi
    KAVRAMSAL bir soru, program adını içerdiği için yanlışlıkla 'sayısal
    bilgi sorusu' sanılıp, tanımı anlatan Wikipedia chunk'ı atılıp sadece
    onlarca üniversitenin ham puan kaydı kalıyordu — bu da soruyu
    cevapsız bırakıyordu.

    EK KURAL: Aynı üniversitenin AYNI temel programının birden fazla
    varyantı (Burslu, Ücretli, UOLP, İngilizce vb.) eşleşirse VE kullanıcı
    sorusunda bu varyantlardan hiçbirini AÇIKÇA belirtmemişse, o üniversite
    için sadece parantezsiz 'düz/varsayılan' varyantı tutuyoruz.

    ÖNEMLİ: Bu tercih HER ÜNİVERSİTE İÇİN AYRI AYRI uygulanır — tüm
    adaylar genelinde DEĞİL. Aksi halde, örneğin Boğaziçi'nin TEK kaydı
    '(İngilizce)' etiketliyken, başka bir üniversitenin (örn. Balıkesir)
    parantezsiz bir kaydı varsa, 'tüm havuzda tek düz kayıt Balıkesir'
    diye yanlışlıkla SADECE Balıkesir'i tutup Boğaziçi'yi (ve diğer tüm
    üniversiteleri) tamamen eleyebiliyordu — bu, gerçek bir hataydı ve
    'Boğaziçi Üniversitesi Bilgisayar Mühendisliği' gibi sorularda yanlış
    üniversitenin gösterilmesine yol açıyordu."""
    soru_normalize = turkce_normalize(soru)
    eslesenler = []
    yokatlas_disi_adaylar = []
    for skor, kaynak, metin in adaylar:
        if kaynak != "yokatlas_tum_bolumler_2025.txt":
            yokatlas_disi_adaylar.append((skor, kaynak, metin))
            continue
        eslesme = re.search(r"Program:\s*(.+?)\s*\|", metin)
        if not eslesme:
            continue
        program_adi_ham = eslesme.group(1)
        govde = program_govdesi(program_adi_ham)
        if govde and govde in soru_normalize:
            uni_eslesme = re.search(r"Üniversite:\s*(.+?)\s*\(", metin)
            uni_adi = uni_eslesme.group(1).strip() if uni_eslesme else None
            eslesenler.append((skor, kaynak, metin, program_adi_ham, uni_adi))

    if not eslesenler:
        return adaylar

    varyant_belirtilmis = any(k in soru_normalize for k in VARYANT_ANAHTAR_KELIMELERI)
    if not varyant_belirtilmis:
        gruplu = {}
        for e in eslesenler:
            gruplu.setdefault(e[4], []).append(e)
        yeni_eslesenler = []
        for uni_adi, grup in gruplu.items():
            duz_olanlar = [e for e in grup if program_adi_duz_mu(e[3])]
            if len(duz_olanlar) == 1:
                yeni_eslesenler.extend(duz_olanlar)
            else:
                yeni_eslesenler.extend(grup)
        eslesenler = yeni_eslesenler

    yokatlas_sonuc = [(skor, kaynak, metin) for skor, kaynak, metin, _, _ in eslesenler]
    return yokatlas_sonuc + yokatlas_disi_adaylar


def _universite_govdesi(ad_normalize):
    """'ankara yildirim beyazit universitesi' -> ['ankara','yildirim','beyazit']
    (sondaki 'üniversite/üniversitesi' kelimesini atar).

    ÖNEMLİ: Tireyle birleşik isimlerde (örn. gerçek bir üniversite olan
    'İstanbul Üniversitesi-Cerrahpaşa') tire önce boşluğa çevrilir. Aksi
    halde 'üniversitesi-cerrahpasa' tek kelime sayılır ve bu kelime de
    'universite' ile BAŞLADIĞI için yanlışlıkla TAMAMEN silinir — bu da bu
    üniversitenin gövdesini de ['istanbul'] yapıp düz 'İstanbul Üniversitesi'
    ile ÇAKIŞMASINA (ve dolayısıyla çözümlemenin 'belirsiz' sayılıp hiç
    filtre uygulanmamasına) yol açıyordu."""
    kelimeler = ad_normalize.replace("-", " ").split()
    if kelimeler and kelimeler[-1] in ("universite", "universitesi"):
        kelimeler = kelimeler[:-1]
    return kelimeler


NORMALIZE_TO_ORIJINAL = {}


def tum_universite_adlarini_getir():
    """Veritabanındaki TÜM bilinen üniversite isimlerini (normalize edilmiş)
    döner — hem yokatlas'taki 'Üniversite: X (şehir, tür)' alanlarından,
    hem de '_Üniversitesi.txt' türü genel bilgi dosyalarının adlarından.
    Bu, program başlarken BİR KEZ çalıştırılır. Ayrıca NORMALIZE_TO_ORIJINAL
    sözlüğünü de doldurur (normalize edilmiş isim -> DB'de geçen orijinal
    Türkçe karakterli hali), böylece daha sonra hedef üniversite netleşince
    veritabanında doğrudan SQL LIKE ile arayabiliriz."""
    conn = sqlite3.connect("universite_asistani.db")
    cursor = conn.cursor()
    adlar = set()

    cursor.execute(
        "SELECT metin FROM chunks WHERE kaynak_dosya = 'yokatlas_tum_bolumler_2025.txt'"
    )
    for (metin,) in cursor.fetchall():
        eslesme = re.search(r"Üniversite:\s*(.+?)\s*\(", metin)
        if eslesme:
            orijinal = eslesme.group(1).strip()
            norm = turkce_normalize(orijinal)
            adlar.add(norm)
            NORMALIZE_TO_ORIJINAL.setdefault(norm, orijinal)

    cursor.execute(
        "SELECT DISTINCT kaynak_dosya FROM chunks "
        "WHERE kaynak_dosya != 'yokatlas_tum_bolumler_2025.txt'"
    )
    for (kaynak,) in cursor.fetchall():
        dn = kaynak.lower()
        if "üniversite" in dn or "universite" in dn:
            okunur = kaynak.replace(".txt", "").replace("_", " ")
            adlar.add(turkce_normalize(okunur))

    conn.close()
    return adlar


def hedef_universite_coz(soru_genisletilmis, tum_adlar):
    """Sorudan (kısaltmalar zaten açılmış haliyle) hedeflenen üniversiteyi
    çıkarır. Yalnızca TEK ve NET bir eşleşme varsa ismi döner; hiç eşleşme
    yoksa ya da BİRDEN FAZLA üniversite eşleşiyorsa (belirsizse) None döner
    — bu durumda normal (filtresiz) akışa devam edilir, çünkü emin değiliz.

    ÖNEMLİ: "üniversite" kelimesinden hemen önceki metnin TAMAMINI tek bir
    isim sanmıyoruz — çünkü kullanıcı "bilgisayar mühendisliği nişantaşı
    üniversitesi" gibi, BÖLÜM ADINI üniversite adından ÖNCE de yazabiliyor.
    Bunun yerine, "üniversite" kelimesine bitişik kelimelerden başlayarak
    azalan pencere boyutlarıyla (4, 3, 2, 1 kelime) deneme yapıyoruz — ilk
    NET (tek) eşleşmeyi bulduğumuz anda onu kullanıyoruz. Böylece hem
    "Ankara Üniversitesi" (1 kelime) hem "Yıldız Teknik Üniversitesi"
    (2 kelime) hem de önüne başka kelimeler eklenmiş sorular doğru
    çözülüyor."""
    eslesme = re.search(r"(.+?)\s+[üÜ]niversite", soru_genisletilmis)
    if not eslesme:
        return None

    tum_kelimeler = turkce_normalize(eslesme.group(1).strip()).split()
    if not tum_kelimeler:
        return None

    for pencere in range(min(4, len(tum_kelimeler)), 0, -1):
        hedef_kelimeler = tum_kelimeler[-pencere:]
        adaylar = [
            ad for ad in tum_adlar
            if len(_universite_govdesi(ad)) >= pencere
            and _universite_govdesi(ad)[-pencere:] == hedef_kelimeler
        ]
        if len(adaylar) == 1:
            return adaylar[0]

    return None


def chunk_universite_adi(kaynak, metin):
    """Bir chunk'ın ait olduğu üniversitenin normalize edilmiş adını döner."""
    if kaynak == "yokatlas_tum_bolumler_2025.txt":
        eslesme = re.search(r"Üniversite:\s*(.+?)\s*\(", metin)
        return turkce_normalize(eslesme.group(1).strip()) if eslesme else None
    dn = kaynak.lower()
    if "üniversite" in dn or "universite" in dn:
        return turkce_normalize(kaynak.replace(".txt", "").replace("_", " "))
    return None


print("Bilinen üniversite isimleri yükleniyor...")
TUM_UNIVERSITE_ADLARI = tum_universite_adlarini_getir()
print(f"{len(TUM_UNIVERSITE_ADLARI)} farklı üniversite ismi tespit edildi.")


SAYISAL_SORU_KELIMELERI = [
    "taban puan", "sıralama", "kontenjan", "burslu", "ücretli", "indirimli",
    "puan türü", "kaç kişi", "yerleşen", "eğitim süresi", "hangi bölüm",
    "hangi program", "puan kaç", "sırala",
]

GENEL_SORU_KELIMELERI = [
    "nerede", "ne zaman", "kuruldu", "yerleşke", "kampüs", "tarihçe",
    "kim kurdu", "rektör", "kaç fakülte", "akademik birim", "kuruluş",
]


def soru_tipini_belirle(soru_kucuk):
    """Soruyu SAYISAL, GENEL veya BELIRSIZ olarak sınıflandırır.
    SAYISAL: taban puan/sıralama/kontenjan gibi YÖK Atlas verisi istiyor.
    GENEL: yerleşke/kuruluş tarihi gibi Wikipedia tarzı bilgi istiyor.
    BELIRSIZ: hiçbiri net değil — güvenli tarafta kalıp ikisine de (daha
    ölçülü şekilde) bakıyoruz."""
    sayisal_mi = any(k in soru_kucuk for k in SAYISAL_SORU_KELIMELERI)
    genel_mi = any(k in soru_kucuk for k in GENEL_SORU_KELIMELERI)

    if sayisal_mi and not genel_mi:
        return "SAYISAL"
    if genel_mi and not sayisal_mi:
        return "GENEL"
    return "BELIRSIZ"


def hedef_universiteye_ait_ek_adaylar(hedef_universite, sinir=30):
    """Hedef üniversite netleştiğinde, o üniversiteye ait YÖK Atlas
    kayıtlarını DOĞRUDAN veritabanından (semantik aramadan bağımsız) çeker.

    NEDEN GEREKLİ: Hibrit skor formülü özel isim eşleşmesine (%45 ağırlık)
    çok önem veriyor — yani 'Boğaziçi' geçen HERHANGİ bir chunk (hangi
    bölüm olursa olsun) yüksek puan alıp ilk-15'e girebiliyor. Bu da bazen
    gerçek aranan bölümün (örn. Bilgisayar Mühendisliği) ilk-15'in DIŞINDA
    kalmasına, dolayısıyla hiç bulunamamasına yol açıyordu.

    GÜVENLİK: Bu fonksiyon SADECE hedef üniversite kesin olarak
    çözülebildiğinde çağrılıyor — yani genel/kavramsal sorularda (örn.
    'Bilgisayar mühendisliği nedir?') hiç devreye girmiyor, dolayısıyla
    dün yaşadığımız 'genel havuzu büyütünce döngü hatası geri geldi'
    sorununu tekrarlamıyor: havuzu HER sorguda büyütmek yerine, sadece
    gerekli olan spesifik durumda, hedefli bir ekleme yapıyoruz."""
    orijinal_isim = NORMALIZE_TO_ORIJINAL.get(hedef_universite)
    if not orijinal_isim:
        return []

    conn = sqlite3.connect("universite_asistani.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT kaynak_dosya, metin FROM chunks "
        "WHERE kaynak_dosya = 'yokatlas_tum_bolumler_2025.txt' AND metin LIKE ? "
        "LIMIT ?",
        (f"Üniversite: {orijinal_isim} (%", sinir),
    )
    satirlar = cursor.fetchall()
    conn.close()
    return [(1.0, kaynak, metin) for kaynak, metin in satirlar]


def cross_encoder_ile_yeniden_sirala(soru, adaylar, k=5):
    """adaylar: [(hibrit_skor, kaynak, metin), ...] listesi.
    Cross-encoder ile her adayı soruyla BİRLİKTE değerlendirip yeniden sıralar."""
    if not adaylar:
        return []

    ciftler = [[soru, metin] for _, _, metin in adaylar]
    ham_skorlar = reranker.predict(ciftler)

    normal_skorlar = torch.sigmoid(torch.tensor(ham_skorlar)).numpy()

    birlesik = list(zip(normal_skorlar, adaylar))
    birlesik.sort(key=lambda x: x[0], reverse=True)

    return [(float(ns), kaynak, metin) for ns, (_, kaynak, metin) in birlesik[:k]]


def get_top_chunks_eski(soru, k=15):
    soru_orijinal = soru.strip()
    soru_kucuk = kisaltma_genislet(soru_orijinal.lower())
    soru_embedding = embed_model.encode("query: " + soru_kucuk).tolist()
    ozel_isimler = ozel_isimleri_bul(soru_orijinal)

    conn = sqlite3.connect("universite_asistani.db")
    cursor = conn.cursor()
    cursor.execute("SELECT kaynak_dosya, metin, embedding FROM chunks")
    tum_chunklar = cursor.fetchall()
    conn.close()

    soru_kelimeleri = set(soru_kucuk.split())

    sonuclar = []
    for kaynak_dosya, metin, embedding_json in tum_chunklar:
        chunk_embedding = json.loads(embedding_json)
        benzerlik = cosine_similarity(soru_embedding, chunk_embedding)
        metin_norm = turkce_normalize(metin)
        metin_kelimeleri = set(metin.lower().split())
        ortak = len(soru_kelimeleri & metin_kelimeleri)
        anahtar_skoru = ortak / max(len(soru_kelimeleri), 1)
        eslesme = sum(1 for isim in ozel_isimler if isim in metin_norm)
        ozel_isim_orani = eslesme / max(len(ozel_isimler), 1) if ozel_isimler else 0
        hibrit_skor = (0.45 * benzerlik) + (0.1 * anahtar_skoru) + (0.45 * ozel_isim_orani)
        sonuclar.append((hibrit_skor, kaynak_dosya, metin))

    sonuclar.sort(key=lambda x: x[0], reverse=True)
    return sonuclar[:k]


def get_top_chunks_sqlite_vec(soru, k=15, vektor_havuz_boyutu=30):
    soru_orijinal = soru.strip()
    soru_kucuk = kisaltma_genislet(soru_orijinal.lower())
    soru_embedding = embed_model.encode("query: " + soru_kucuk).tolist()
    soru_embedding_json = json.dumps(soru_embedding)

    conn = sqlite3.connect("universite_asistani.db")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT chunks.kaynak_dosya, chunks.metin, chunks_vec.distance
        FROM chunks_vec
        JOIN chunks ON chunks.id = chunks_vec.rowid
        WHERE chunks_vec.embedding MATCH ? AND k = ?
        ORDER BY chunks_vec.distance
    """, (soru_embedding_json, vektor_havuz_boyutu))
    on_secim = cursor.fetchall()
    conn.close()

    ozel_isimler = ozel_isimleri_bul(soru_orijinal)
    soru_kelimeleri = set(soru_kucuk.split())

    sonuclar = []
    for kaynak, metin, distance in on_secim:
        benzerlik = 1 - (distance / 2)
        metin_norm = turkce_normalize(metin)
        metin_kelimeleri = set(metin.lower().split())
        ortak = len(soru_kelimeleri & metin_kelimeleri)
        anahtar_skoru = ortak / max(len(soru_kelimeleri), 1)
        eslesme = sum(1 for isim in ozel_isimler if isim in metin_norm)
        ozel_isim_orani = eslesme / max(len(ozel_isimler), 1) if ozel_isimler else 0
        hibrit_skor = (0.45 * benzerlik) + (0.1 * anahtar_skoru) + (0.45 * ozel_isim_orani)
        sonuclar.append((hibrit_skor, kaynak, metin))

    sonuclar.sort(key=lambda x: x[0], reverse=True)
    return sonuclar[:k]


def get_top_chunks(soru, k=5):
    aday_sayisi = 10 if KULLAN_RERANKING else k

    if KULLAN_SQLITE_VEC:
        adaylar = get_top_chunks_sqlite_vec(soru, aday_sayisi)
    else:
        adaylar = get_top_chunks_eski(soru, aday_sayisi)

    soru_kucuk_routing = turkce_kucuk_harf(soru)
    soru_tipi = soru_tipini_belirle(soru_kucuk_routing)

    if soru_tipi == "GENEL":
        adaylar = [
            (skor, kaynak, metin) for skor, kaynak, metin in adaylar
            if kaynak != "yokatlas_tum_bolumler_2025.txt"
        ]

    soru_genisletilmis_erken = kisaltma_genislet(soru)
    hedef_universite_erken = hedef_universite_coz(soru_genisletilmis_erken, TUM_UNIVERSITE_ADLARI)
    if hedef_universite_erken and soru_tipi != "GENEL":
        ek_sinir = 15 if soru_tipi == "SAYISAL" else 5
        ek_adaylar = hedef_universiteye_ait_ek_adaylar(hedef_universite_erken, sinir=ek_sinir)
        mevcut_metinler = {metin for _, _, metin in adaylar}
        for skor, kaynak, metin in ek_adaylar:
            if metin not in mevcut_metinler:
                adaylar.append((skor, kaynak, metin))
                mevcut_metinler.add(metin)

    adaylar = program_adina_gore_filtrele(soru, adaylar)

    if KULLAN_RERANKING:
        return cross_encoder_ile_yeniden_sirala(soru, adaylar, k)
    else:
        return adaylar[:k]


BENZERLIK_ESIGI = 0.40

def answer_query(soru):
    top_chunklar = get_top_chunks(soru, k=5)

    if not top_chunklar or top_chunklar[0][0] < BENZERLIK_ESIGI:
        cevap = "Bu konuda bilgi tabanımda yeterli bilgi bulunmuyor."
        log_kaydet(soru, cevap, top_chunklar)
        return {
            "cevap": cevap,
            "kaynaklar": top_chunklar,
        }

    soru_genisletilmis = kisaltma_genislet(soru)

    hedef_universite = hedef_universite_coz(soru_genisletilmis, TUM_UNIVERSITE_ADLARI)
    if hedef_universite:
        filtrelenmis = [
            (skor, kaynak, metin) for skor, kaynak, metin in top_chunklar
            if chunk_universite_adi(kaynak, metin) == hedef_universite
        ]
        if not filtrelenmis:
            cevap = "Bu üniversite hakkında veri tabanımda bilgi bulunmuyor."
            log_kaydet(soru, cevap, top_chunklar)
            return {
                "cevap": cevap,
                "kaynaklar": top_chunklar,
            }
        top_chunklar = filtrelenmis

    baglam = "\n\n".join([
        f"[Kaynak: {kaynak}]\n{metin}"
        for _, kaynak, metin in top_chunklar
    ])

    sistem_promptu = (
        "Sen üniversite bölümleri hakkında bilgi veren bir asistansın. "
        "SADECE aşağıdaki bağlamda AÇIKÇA GEÇEN üniversite için cevap ver. "
        "Kullanıcının sorduğu üniversite adı (kısaltmalar açılmış haliyle) "
        "bağlamdaki metinde geçmiyorsa, başka bir üniversitenin verisini "
        "KESİNLİKLE kullanma ve 'Bu üniversite hakkında veri tabanımda bilgi "
        "bulunmuyor' de. Farklı üniversitelerin verilerini asla karıştırma "
        "veya kendiliğinden karşılaştırma yapma. Kişisel tavsiye verme, "
        "sadece objektif bilgi sun. Soruyu tekrar etme, doğrudan ve kısa "
        "cevap ver. Bağlamda yeterli bilgi yoksa 'Bu konuda yeterli bilgim "
        "yok' de. Bağlamda AÇIKÇA yazmayan hiçbir ilişkiyi veya detayı "
        "(örneğin bir programın hangi fakülteye bağlı olduğunu, bir "
        "programın var olup olmadığını) kendi tahmininle tamamlama veya "
        "diğer programlardan yola çıkarak varsayma; sadece bağlamda birebir "
        "yazan bilgiyi kullan. ÇOK ÖNEMLİ: Kullanıcının sorduğu üniversite "
        "adı, bağlamdaki bir üniversitenin isminin SADECE BİR PARÇASI ise "
        "(örneğin kullanıcı 'Ankara Üniversitesi' derken bağlamda 'Ankara "
        "Yıldırım Beyazıt Üniversitesi' geçiyorsa, ya da 'Marmara "
        "Üniversitesi' derken bağlamda başka bir 'Marmara ...Üniversitesi' "
        "geçiyorsa), bunlar KESİNLİKLE FARKLI üniversitelerdir ve "
        "birbirinin yerine ASLA kullanılamaz. Üniversite adı bağlamda TAM "
        "VE EKSİKSİZ (kelime kelime birebir) geçmiyorsa, o üniversite "
        "hakkında 'veri tabanımda bilgi bulunmuyor' de. Bağlamda aynı "
        "bölüm/programın BİRDEN FAZLA varyantı varsa (örn. 'Burslu', "
        "'%50 İndirimli', 'Ücretli' gibi farklı burs seçenekleri, ya da "
        "Türkçe/İngilizce gibi farklı eğitim dili seçenekleri), cevabında "
        "SADECE BİRİNİ seçip diğerlerini atlama — bağlamda geçen HER "
        "varyantı, kendi taban puanı/sıralaması/kontenjanıyla birlikte "
        "ayrı ayrı belirt. BU SATIR-SATIR FORMAT KURALI SADECE kullanıcı "
        "AÇIKÇA taban puan/sıralama/kontenjan SAYISI istiyorsa geçerlidir. "
        "Böyle bir durumda: cevabını serbest bir cümle olarak KURMA, bunun "
        "yerine bağlamdaki HER ilgili satırı, o satırdaki 'Program:', "
        "'Taban Puan:' ve 'Sıralama:' alanlarını DEĞİŞTİRMEDEN, birebir "
        "kopyalayarak, aşağıdaki gibi ayrı satırlar halinde listele:\n"
        "- <Program adı>: Taban Puan <X>, Sıralama <Y>\n"
        "Bir satırdaki sayıyı ASLA başka bir satırın program adıyla "
        "birleştirme — her sayı SADECE aynı bağlam satırında birlikte "
        "geçtiği program adıyla eşleşmelidir. Sadece tek bir ilgili "
        "program/satır varsa, normal kısa bir cümleyle cevap verebilirsin. "
        "KULLANICI SAYI İSTEMİYORSA (örneğin 'burslu mu?', 'var mı?', "
        "'hangi dilde?' gibi evet/hayır veya nitel bir soru soruyorsa, ya "
        "da soru kavramsal bir tanım/karşılaştırmaysa), bu satır-satır "
        "formatı HİÇ KULLANMA — bağlamda 'Taban Puan:'/'Sıralama:' alanı "
        "görsen bile onları cevabına dahil etme; sadece SORULAN şeye "
        "(örn. burs durumu, varlık, dil) normal, akıcı bir cümleyle, "
        "bağlamdaki ilgili bilgiye dayanarak cevap ver. Kesinlikle sayı "
        "uydurma ve bağlamda gerçekten yazan bilgiyi kullanmadan "
        "'bilgi yok' deme."
    )

    kullanici_promptu = f"Bağlam:\n{baglam}\n\nSoru: {soru_genisletilmis}"

    response = chat_client.complete_chat(
        [
            {"role": "system", "content": sistem_promptu},
            {"role": "user", "content": kullanici_promptu}
        ]
    )

    cevap = response.choices[0].message.content.strip()
    log_kaydet(soru, cevap, top_chunklar)

    return {
        "cevap": cevap,
        "kaynaklar": top_chunklar,
    }


if __name__ == "__main__":
    test_sorulari = [
        "Rumeli Üniversitesi yüzde 50 ücretli Bilgisayar Mühendisliği taban puanı kaç?",
    ]

    for soru in test_sorulari:
        print(f"\n{'=' * 70}")
        print(f"Soru: {soru}")
        sonuc = answer_query(soru)
        print(f"Cevap: {sonuc['cevap']}")
        print("Kaynaklar:")
        for skor, kaynak, metin in sonuc["kaynaklar"]:
            print(f"  skor: {skor:.3f} | {metin[:100]}")

    model.unload()