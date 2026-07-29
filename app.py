import streamlit as st
import psutil
import subprocess
import os
import random
from rag_asistan import answer_query

st.set_page_config(page_title="Üniversite Bölüm Asistanı", page_icon="🎓", layout="wide")


if "tema" not in st.session_state:
    st.session_state.tema = "koyu"

KOYU = {
    "bg_0": "#0A0A10",
    "bg_1": "#121017",
    "glow_1": "rgba(212, 175, 106, 0.14)",
    "glow_2": "rgba(95, 224, 208, 0.07)",
    "cam": "rgba(255, 255, 255, 0.045)",
    "cam_border": "rgba(212, 175, 106, 0.20)",
    "cam_border_hover": "rgba(212, 175, 106, 0.55)",
    "notr_border": "rgba(255, 255, 255, 0.09)",
    "metin_1": "#F3EFE8",
    "metin_2": "#A39CAE",
    "metin_3": "#736C7E",
    "altin": "#D9B36A",
    "altin_parlak": "#EFCB8B",
    "teal": "#5FE0D0",
    "kirmizi": "#E08585",
    "sidebar_bg": "rgba(18, 16, 23, 0.55)",
}

ACIK = {
    "bg_0": "#F3EEE3",
    "bg_1": "#FBF8F1",
    "glow_1": "rgba(166, 117, 46, 0.10)",
    "glow_2": "rgba(31, 156, 140, 0.06)",
    "cam": "rgba(255, 255, 255, 0.55)",
    "cam_border": "rgba(166, 117, 46, 0.24)",
    "cam_border_hover": "rgba(166, 117, 46, 0.6)",
    "notr_border": "rgba(90, 74, 46, 0.14)",
    "metin_1": "#231E17",
    "metin_2": "#6B6254",
    "metin_3": "#948C7C",
    "altin": "#A6752E",
    "altin_parlak": "#8A5D1F",
    "teal": "#188577",
    "kirmizi": "#B04343",
    "sidebar_bg": "rgba(251, 248, 241, 0.6)",
}

T = KOYU if st.session_state.tema == "koyu" else ACIK

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600;700&display=swap');

    html, body {{ background: {T['bg_0']} !important; }}
    header[data-testid="stHeader"] {{ background: transparent !important; }}
    header[data-testid="stHeader"] * {{ color: {T['metin_2']} !important; }}
    div[data-testid="stAppViewContainer"] {{ background: transparent !important; }}
    div[data-testid="stMainBlockContainer"] {{ background: transparent !important; }}
    div[data-testid="stBottomBlockContainer"] {{ background: transparent !important; }}
    div[data-testid="stDecoration"] {{ background: transparent !important; }}

    div[data-testid="stChatInputContainer"] {{
        background: transparent !important;
        border-top: 1px solid {T['notr_border']} !important;
    }}

    .stApp {{
        background:
            radial-gradient(ellipse 900px 500px at 12% -10%, {T['glow_1']}, transparent 60%),
            radial-gradient(ellipse 700px 500px at 100% 15%, {T['glow_2']}, transparent 55%),
            linear-gradient(180deg, {T['bg_0']} 0%, {T['bg_1']} 100%) !important;
        color: {T['metin_1']} !important;
    }}

    section[data-testid="stSidebar"] {{
        background: {T['sidebar_bg']} !important;
        backdrop-filter: blur(24px) saturate(140%);
        -webkit-backdrop-filter: blur(24px) saturate(140%);
        border-right: 1px solid {T['cam_border']};
    }}
    section[data-testid="stSidebar"] * {{ color: {T['metin_2']} !important; }}
    section[data-testid="stSidebar"] h3 {{
        color: {T['altin']} !important;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        font-weight: 600;
    }}

    html, body, [class*="css"] {{ font-family: 'IBM Plex Sans', sans-serif; }}

    .cam {{
        background: {T['cam']};
        border: 1px solid {T['cam_border']};
        backdrop-filter: blur(20px) saturate(160%);
        -webkit-backdrop-filter: blur(20px) saturate(160%);
        box-shadow: 0 8px 32px rgba(0,0,0,0.18), inset 0 1px 0 rgba(255,255,255,0.06);
        transition: border-color 0.25s ease;
    }}

    .kimlik-karti {{
        position: relative;
        border-radius: 16px !important;
        padding: 0;
        margin-bottom: 26px;
        overflow: hidden;
    }}
    .kimlik-ust {{ padding: 32px 40px 26px 40px; position: relative; }}
    .muhur {{
        position: absolute;
        top: 24px; right: 34px;
        width: 54px; height: 54px;
        border-radius: 50%;
        border: 1.5px dashed {T['altin']};
        opacity: 0.75;
        display: flex; align-items: center; justify-content: center;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.56rem;
        color: {T['altin']};
        text-align: center;
        line-height: 1.25;
        letter-spacing: 0.02em;
    }}
    .kimlik-baslik {{
        font-family: 'Fraunces', serif;
        font-size: 2.1rem;
        font-weight: 700;
        color: {T['metin_1']};
        margin: 0 0 8px 0;
        line-height: 1.1;
        max-width: 400px;
    }}
    .kimlik-sub {{
        color: {T['metin_2']};
        font-size: 0.92rem;
        margin: 0;
        max-width: 420px;
        line-height: 1.5;
    }}
    .perfore {{
        border-top: 1.5px dashed {T['cam_border']};
        position: relative;
    }}
    .perfore::before, .perfore::after {{
        content: '';
        position: absolute;
        top: -8px;
        width: 16px; height: 16px;
        border-radius: 50%;
        background: {T['bg_0']};
        box-shadow: inset 0 0 0 1px {T['cam_border']};
    }}
    .perfore::before {{ left: -8px; }}
    .perfore::after {{ right: -8px; }}
    .kimlik-alt {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 16px 40px;
        flex-wrap: wrap;
        gap: 16px;
    }}
    .stat-satiri {{ display: flex; gap: 28px; flex-wrap: wrap; }}
    .stat-item {{ display: flex; flex-direction: column; }}
    .stat-num {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.1rem;
        font-weight: 600;
        background: linear-gradient(135deg, {T['altin_parlak']}, {T['altin']});
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .stat-label {{
        font-size: 0.6rem;
        color: {T['metin_3']};
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}
    .barkod {{ display: flex; gap: 2px; align-items: flex-end; height: 26px; opacity: 0.5; }}
    .barkod span {{ width: 2px; background: {T['metin_2']}; }}

    .skor-badge {{
        display: inline-flex;
        align-items: center;
        font-family: 'IBM Plex Mono', monospace;
        padding: 4px 11px;
        border-radius: 3px;
        font-size: 0.74rem;
        font-weight: 600;
        margin-right: 6px;
        border: 1px solid transparent;
    }}
    .skor-yuksek {{ background: rgba(95, 224, 208, 0.12); color: {T['teal']}; border-color: rgba(95, 224, 208, 0.3); }}
    .skor-orta {{ background: rgba(217, 179, 106, 0.12); color: {T['altin']}; border-color: rgba(217, 179, 106, 0.3); }}
    .skor-dusuk {{ background: rgba(224, 133, 133, 0.12); color: {T['kirmizi']}; border-color: rgba(224, 133, 133, 0.3); }}

    .kaynak-kutusu {{
        background: {T['cam']};
        border: 1px solid {T['cam_border']};
        border-left: 3px solid {T['altin']};
        backdrop-filter: blur(14px);
        padding: 13px 17px;
        border-radius: 4px;
        margin-bottom: 10px;
        font-size: 0.85rem;
        line-height: 1.55;
        color: {T['metin_2']};
    }}
    .kaynak-baslik {{
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        color: {T['altin']};
        font-size: 0.75rem;
    }}

    .veri-etiketi {{
        display: inline-block;
        background: {T['cam']};
        border: 1px solid {T['cam_border']};
        color: {T['metin_2']};
        padding: 5px 13px;
        border-radius: 3px;
        font-size: 0.72rem;
        font-weight: 500;
        margin: 3px 4px 3px 0;
    }}
    .motor-etiketi {{
        display: inline-block;
        background: rgba(95, 224, 208, 0.08);
        border: 1px solid rgba(95, 224, 208, 0.28);
        color: {T['teal']};
        font-family: 'IBM Plex Mono', monospace;
        padding: 4px 11px;
        border-radius: 3px;
        font-size: 0.68rem;
        font-weight: 600;
        margin: 3px 4px 3px 0;
    }}

    div[data-testid="stChatMessage"] {{
        background: {T['cam']} !important;
        border: 1px solid {T['cam_border']} !important;
        border-left: 3px solid {T['altin']} !important;
        backdrop-filter: blur(20px) saturate(160%);
        -webkit-backdrop-filter: blur(20px) saturate(160%);
        border-radius: 4px !important;
        box-shadow: 0 6px 24px rgba(0,0,0,0.14);
    }}
    div[data-testid="stChatMessage"] p {{ color: {T['metin_1']} !important; }}

    div[data-testid="stMetric"] {{
        background: {T['cam']};
        border: 1px solid {T['cam_border']};
        border-radius: 8px;
        padding: 10px 14px;
        backdrop-filter: blur(14px);
    }}
    div[data-testid="stMetricValue"] {{
        color: {T['metin_1']} !important;
        font-family: 'IBM Plex Mono', monospace;
    }}
    div[data-testid="stMetricLabel"] {{ color: {T['metin_3']} !important; }}
    .streamlit-expanderHeader {{ color: {T['metin_2']} !important; }}
    div[data-testid="stExpander"] {{
        background: {T['cam']} !important;
        border: 1px solid {T['cam_border']} !important;
        border-radius: 8px !important;
        backdrop-filter: blur(16px);
    }}

    section[data-testid="stSidebar"] button {{
        background: {T['cam']} !important;
        border: 1px solid {T['cam_border']} !important;
        color: {T['metin_2']} !important;
        text-align: left !important;
        border-radius: 4px !important;
        font-size: 0.82rem !important;
        backdrop-filter: blur(10px);
        transition: all 0.2s ease !important;
    }}
    section[data-testid="stSidebar"] button:hover {{
        border-color: {T['cam_border_hover']} !important;
        color: {T['altin_parlak']} !important;
    }}

    div[data-testid="stChatInput"] {{
        background: {T['cam']} !important;
        border: 1px solid {T['cam_border']} !important;
        border-radius: 6px !important;
        backdrop-filter: blur(20px);
    }}
    div[data-testid="stChatInput"] textarea {{
        background: transparent !important;
        color: {T['metin_1']} !important;
        border: none !important;
    }}
    div[data-testid="stChatInput"] textarea::placeholder {{ color: {T['metin_3']} !important; }}

    hr {{ border-color: {T['notr_border']} !important; }}

    div[data-testid="stToggle"] label p {{
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.72rem !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }}
</style>
""", unsafe_allow_html=True)


def barkod_html():
    random.seed(42)
    cubuklar = "".join(f'<span style="height:{random.randint(10,26)}px"></span>' for _ in range(30))
    return f'<div class="barkod">{cubuklar}</div>'


def skor_rozeti(skor):
    if skor >= 0.7:
        sinif, etiket = "skor-yuksek", "yüksek eşleşme"
    elif skor >= 0.5:
        sinif, etiket = "skor-orta", "orta eşleşme"
    else:
        sinif, etiket = "skor-dusuk", "zayıf eşleşme"
    return f'<span class="skor-badge {sinif}">{etiket} · {skor:.2f}</span>'


def gpu_bilgisi_al():
    try:
        sonuc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2
        )
        if sonuc.returncode == 0:
            kullanilan, toplam, kullanim = sonuc.stdout.strip().split(", ")
            return {"kullanilan_mb": float(kullanilan), "toplam_mb": float(toplam), "kullanim_yuzde": float(kullanim)}
    except Exception:
        return None
    return None


def donanim_paneli():
    with st.sidebar:
        acik_mi = st.toggle("Açık tema", value=(st.session_state.tema == "acik"))
        yeni_tema = "acik" if acik_mi else "koyu"
        if yeni_tema != st.session_state.tema:
            st.session_state.tema = yeni_tema
            st.rerun()

        st.markdown("### Donanım")
        process = psutil.Process(os.getpid())
        ram_mb = process.memory_info().rss / (1024 * 1024)
        col1, col2 = st.columns(2)
        col1.metric("RAM", f"{ram_mb:.0f} MB")
        col2.metric("CPU", f"%{psutil.cpu_percent(interval=0.3)}")

        gpu = gpu_bilgisi_al()
        if gpu:
            st.metric("GPU Bellek", f"{gpu['kullanilan_mb']:.0f} / {gpu['toplam_mb']:.0f} MB", f"%{gpu['kullanim_yuzde']} kullanım")
        else:
            st.caption("GPU bilgisi alınamadı")

        if st.button("Yenile", use_container_width=True):
            st.rerun()

        st.divider()
        st.markdown("### Motor")
        st.markdown('<span class="motor-etiketi">phi-4-mini</span><span class="motor-etiketi">e5-large</span><span class="motor-etiketi">sqlite-vec</span>', unsafe_allow_html=True)

        st.divider()
        st.markdown("### Veri kapsamı")
        st.markdown("""
<span class="veri-etiketi">2025 YKS verisi</span>
<span class="veri-etiketi">Taban puan / sıralama</span>
<span class="veri-etiketi">Burs durumu</span>
<span class="veri-etiketi">Akademik kadro</span>
<span class="veri-etiketi">228 üniversite</span>
        """, unsafe_allow_html=True)

        st.divider()
        st.markdown("### Örnek sorular")
        ornekler = [
            "Boğaziçi Üniversitesi Bilgisayar Mühendisliği taban puanı kaç?",
            "İzmir'deki devlet üniversitelerinde Tıp Fakültesi hangileri?",
            "Koç Üniversitesi Bilgisayar Mühendisliği burslu mu?",
            "Galatasaray Üniversitesi Hukuk Fakültesi'nde kaç profesör var?",
            "Bilgisayar mühendisliği nedir?",
        ]
        for ornek in ornekler:
            if st.button(ornek, use_container_width=True, key=f"ornek_{ornek}"):
                st.session_state.secilen_ornek = ornek


donanim_paneli()

st.markdown(f"""
<div class="cam kimlik-karti">
    <div class="kimlik-ust">
        <div class="muhur">2025<br>YKS</div>
        <p class="kimlik-baslik">Üniversite Bölüm Asistanı</p>
        <p class="kimlik-sub">228 üniversite ve tüm lisans bölümleri hakkında, internet gerekmeden cevap veren yapay zekâ.</p>
    </div>
    <div class="perfore"></div>
    <div class="kimlik-alt">
        <div class="stat-satiri">
            <div class="stat-item"><span class="stat-num">9.000+</span><span class="stat-label">Bilgi parçası</span></div>
            <div class="stat-item"><span class="stat-num">228</span><span class="stat-label">Üniversite</span></div>
            <div class="stat-item"><span class="stat-num">2025</span><span class="stat-label">Güncel veri</span></div>
            <div class="stat-item"><span class="stat-num">%100</span><span class="stat-label">Offline</span></div>
        </div>
        {barkod_html()}
    </div>
</div>
""", unsafe_allow_html=True)

if "gecmis" not in st.session_state:
    st.session_state.gecmis = []
if "secilen_ornek" not in st.session_state:
    st.session_state.secilen_ornek = ""

for gecmis_soru, sonuc in st.session_state.gecmis:
    with st.chat_message("user", avatar="🙋"):
        st.write(gecmis_soru)
    with st.chat_message("assistant", avatar="🎓"):
        st.write(sonuc["cevap"])
        with st.expander("Kullanılan kaynaklar"):
            for skor, kaynak, metin in sonuc["kaynaklar"]:
                st.markdown(skor_rozeti(skor), unsafe_allow_html=True)
                st.markdown(f'<div class="kaynak-kutusu"><span class="kaynak-baslik">{kaynak}</span><br>{metin[:500]}...</div>', unsafe_allow_html=True)

varsayilan_soru = st.session_state.secilen_ornek
soru = st.chat_input("Üniversite veya bölüm hakkında bir şey sor...")

if soru or varsayilan_soru:
    aktif_soru = soru if soru else varsayilan_soru
    st.session_state.secilen_ornek = ""

    with st.chat_message("user", avatar="🙋"):
        st.write(aktif_soru)

    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner("Düşünüyorum..."):
            sonuc = answer_query(aktif_soru)
        st.write(sonuc["cevap"])
        with st.expander("Kullanılan kaynaklar"):
            for skor, kaynak, metin in sonuc["kaynaklar"]:
                st.markdown(skor_rozeti(skor), unsafe_allow_html=True)
                st.markdown(f'<div class="kaynak-kutusu"><span class="kaynak-baslik">{kaynak}</span><br>{metin[:1000]}...</div>', unsafe_allow_html=True)

    st.session_state.gecmis.append((aktif_soru, sonuc))
    st.rerun()