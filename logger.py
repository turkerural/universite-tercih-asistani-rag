import json
import datetime


def log_kaydet(soru, cevap, top_chunklar, dosya_adi="sorgu_loglari.jsonl"):
    """Her soru-cevabı bir log dosyasına (JSONL formatında) kaydeder."""
    en_iyi_skor = top_chunklar[0][0] if top_chunklar else 0.0
    kayit = {
        "zaman": datetime.datetime.now().isoformat(timespec="seconds"),
        "soru": soru,
        "cevap": cevap,
        "en_iyi_skor": round(en_iyi_skor, 3),
        "kaynak_sayisi": len(top_chunklar),
    }
    with open(dosya_adi, "a", encoding="utf-8") as f:
        f.write(json.dumps(kayit, ensure_ascii=False) + "\n")


def loglari_oku(dosya_adi="sorgu_loglari.jsonl"):
    """Kaydedilen tüm logları okuyup liste olarak döndürür - analiz için kullanışlı."""
    try:
        with open(dosya_adi, encoding="utf-8") as f:
            return [json.loads(satir) for satir in f if satir.strip()]
    except FileNotFoundError:
        return []