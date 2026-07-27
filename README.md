# AgentArdil

Triyaj tabanlı **hibrit agent sistemi**: gelen her isteği önce sınıflandırıp üç
yürütme şeridinden en uygununa yönlendirir — böylece basit sorular anında cevaplanır,
karmaşık işler ise gereken kadar akıl yürütmeyle çözülür. Sesle konuşarak da
kullanılabilir, çok alanlı bir araç setine sahiptir ve her koşunun ne yaptığını
adım adım loglar.

- **Backend:** Python · LangGraph · FastAPI
- **Frontend:** React 19 · Vite
- **LLM:** sağlayıcı-bağımsız (OpenAI-uyumlu endpoint; varsayılan `gpt-4.1`)
- **Ses tanıma (STT):** Groq Whisper (`whisper-large-v3`)
- **Sesli çıktı (TTS):** Coqui XTTS-v2 — Türkçe TTS, yerel/CUDA, süreç içi (opsiyonel)

---

## İçindekiler

- [Neden hibrit?](#neden-hibrit)
- [Mimari](#mimari)
- [Dizin yapısı](#dizin-yapısı)
- [Kurulum](#kurulum)
- [Çalıştırma](#çalıştırma)
- [Nasıl çalışır?](#nasıl-çalışır)
- [Loglama ve gözlemlenebilirlik](#loglama-ve-gözlemlenebilirlik)
- [Ses tanıma (STT)](#ses-tanıma-stt)
- [Sesli çıktı (TTS)](#sesli-çıktı-tts)
- [Ortam değişkenleri](#ortam-değişkenleri)
- [Tasarım kararları](#tasarım-kararları)

---

## Neden hibrit?

Tek tip bir agent her işi aynı yöntemle çözmeye çalışır: "nasılsın" sorusuna bile
plan kurar, ya da çok adımlı bir işi tek hamlede halletmeye çalışıp yarıda kalır.
AgentArdil bunun yerine önce **triyaj** yapar ve işi yapısına göre yönlendirir:

| Şerit | Ne zaman | Nasıl çalışır |
|-------|----------|----------------|
| **direct** | Araç gerektirmeyen sohbet/tanım soruları | Triyaj cevabı **aynı yanıtta** üretir (ekstra tur yok) |
| **react** | Araç gerekiyor ama yol baştan belli değil, ya da iş kısa | Her turda gözleme bakarak sonraki eylemi seçer |
| **plan_execute** | Çok adımlı, adımları baştan çıkarılabilen işler | Önce tam plan, sonra sırayla yürütme, gerekirse yeniden planlama |

Ayrım işin **konusuyla değil yapısıyla** ilgilidir: "Adımları şimdiden yazabiliyor
muyum?" sorusu şeridi belirler.

---

## Mimari

```mermaid
flowchart TD
    IN([Kullanıcı isteği<br/>metin ya da ses]) --> TRIAGE{TRİYAJ<br/>LLM sınıflandırıcı}

    TRIAGE -->|lane: direct| DIRECT([Cevap triyajda üretildi])
    TRIAGE -->|lane: react<br/>+ ön-seçili araçlar| REACT[ReAct şeridi<br/>gözle → eylem → gözle]
    TRIAGE -->|lane: plan_execute<br/>+ ön-seçili araçlar| PE[Plan-Execute şeridi]

    subgraph PE_SUB [Plan-Execute]
        P[planner] --> E[executor] --> R[replanner]
        R -->|devam| E
        R -->|bitti| PEEND([cevap])
    end
    PE --> PE_SUB

    REACT --> TOOLS[(Araç kataloğu)]
    E --> TOOLS

    DIRECT --> OUT([Nihai cevap + üretilen dosyalar])
    REACT --> OUT
    PEEND --> OUT
```

Triyaj **zengin** bir karar döndürür: sadece şeridi değil, o iş için **ön-seçilmiş
araç alt kümesini** de. Böylece alt katmanlar 20+ aracın tamamını değil, yalnızca
gerekebilecek 2-3 aracı görür — context küçülür, yanlış araç seçme ihtimali düşer.

Her şerit dışarıya **aynı arayüzü** sunar (`{input, history}` alır,
`{response, tool_calls, artifacts}` döner); bu yüzden triyaj katmanı şeritlerin
içini bilmek zorunda değildir.

---

## Dizin yapısı

```
AgentArdil/
├── run_api.py              # API sunucusunu başlatır (uvicorn)
├── main.py                 # CLI: şeritleri/triyajı terminalden çalıştır
├── pyproject.toml
├── requirements.txt        # tüm bağımlılıklar (sonda opsiyonel TTS bölümü)
├── .env.example            # ortam değişkenleri şablonu
│
├── src/agent_core/
│   ├── router.py           # Sistemin tek giriş kapısı: triyaj + şeritler
│   ├── config.py           # Sağlayıcı-bağımsız LLM kurulumu (.env'den)
│   ├── tools.py            # Araç registry'si + LangChain köprüsü
│   ├── history.py          # Session (sohbet) hafızası
│   ├── messages.py         # Mesajlardan araç çağrısı/dosya çıkarımı (ortak)
│   ├── tracing.py          # Koşu izleme → session log dosyaları
│   ├── stt.py              # Ses tanıma (Groq Whisper)
│   ├── tts.py              # Sesli çıktı: metni parçalayıp XTTS-v2 ile seslendirir (süreç içi)
│   ├── api.py              # FastAPI: frontend'in konuştuğu HTTP arayüzü
│   │
│   ├── triage/             # Yönlendirme katmanı (schemas/prompts/nodes)
│   ├── plan_execute/       # Plan-Execute şeridi (+ kısa süreli hafıza)
│   └── react/              # ReAct şeridi
│
└── Frontend/               # React 19 + Vite sohbet arayüzü
    ├── src/api.js          # Backend istemcisi
    └── src/components/      # Sidebar, ChatHeader, ChatMessage, ChatInput
```

---

## Kurulum

### Gereksinimler
- Python ≥ 3.11
- Node.js ≥ 18
- [`uv`](https://github.com/astral-sh/uv) (önerilir) ya da `pip`

### 1) Backend

```bash
# proje kökünde
uv venv --python 3.11
uv pip install -e .

# ortam değişkenleri
cp .env.example .env
# .env'i açıp LLM_API_KEY, GROQ_API_KEY (ve istersen TAVILY_API_KEY) doldur
```

> **Not (Windows/OneDrive):** Proje OneDrive altındaysa `uv` komutlarını
> `--link-mode=copy` ile çalıştır (OneDrive sabit bağlantıya izin vermez):
> `uv pip install --link-mode=copy -e .`

> **Opsiyonel — yerel TTS (XTTS-v2):** Sesli çıktı için gereken paketler
> `requirements.txt`'in sonundaki **opsiyonel TTS bölümündedir** (torch+CUDA ~2.5 GB).
> İstemiyorsan o bölümü silebilirsin. Ayrı venv/süreç gerekmez; ayrıntı için
> [Sesli çıktı (TTS)](#sesli-çıktı-tts) bölümüne bak.

### 2) Frontend

```bash
cd Frontend
npm install
```

---

## Çalıştırma

Sistem **iki süreçten** oluşur (sesli çıktı ayrı bir süreç GEREKTİRMEZ — model
backend'in içinde çalışır):

| Süreç | Port | Nasıl başlar |
|-------|:----:|--------------|
| **Backend API** (uvicorn) | `8000` | `python run_api.py` |
| **Frontend** (Vite) | `5173` | `Frontend/`'de `npm run dev` |

### 1) Backend (Terminal 1 — proje kökü)

Önce venv'i aktive et (Windows cmd: `.venv\Scripts\activate`), sonra:

```bash
python run_api.py            # http://127.0.0.1:8000
# geliştirme (dosya değişince yeniden başlar):
python run_api.py --reload
```

Sesli çıktı açıksa XTTS modeli **ilk sesli istekte** aynı süreçte yüklenir (o ilk
cevap birkaç saniye bekler; sonrakiler hızlı). Sesli mod kapalıyken hiç yüklenmez.

### 2) Frontend (Terminal 2 — `Frontend/`)

```bash
npm run dev                  # http://localhost:5173
```

Tarayıcıda **frontend adresini** aç (`http://localhost:5173`) — backend adresini
(`:8000`) değil. Vite, `/api/*` isteklerini otomatik olarak backend'e yönlendirir
(proxy), böylece CORS derdi olmaz ve ajanın ürettiği görseller/sesler doğrudan çalışır.

> Kök adres `http://127.0.0.1:8000/` "Not Found" döndürür — bu **normaldir**, API
> `/api/*` altında yaşar. Sağlık kontrolü: `http://127.0.0.1:8000/api/health`.

### Kullanım

- Alt taraftaki giriş kutusuna yaz ya da 🎤 ile konuş (STT metne çevirir, gönderirsin).
- 🔊 **Sesli çıktı** anahtarını açarsan cevap seslendirilir (bkz. [Sesli çıktı](#sesli-çıktı-tts)).
- Sağ alttaki dil seçici STT dilini belirler.

### CLI (backend'i tek başına denemek için)

```bash
python main.py "Apple'ın güncel fiyatı nedir?"          # triyaj karar verir
python main.py --lane react "AAPL ile MSFT'yi karşılaştır"
python main.py --lane plan_execute --tools get_balance_sheet,plot_chart "MSFT bilançosu ve grafiği"
```

---

## Nasıl çalışır?

### Triyaj
Küçük/hızlı bir LLM, isteği **structured output** ile sınıflandırır. `lane` katı
bir enum'dur (geçersiz şerit üretilemez). `direct` şeridinde cevabı da aynı yanıtta
döndürür — sohbet mesajları için ikinci bir tur beklenmez.

### Plan-Execute şeridi
`planner → executor → replanner` döngüsü. **Kısa süreli hafıza** her adımın tam
kaydını tutar: yapılan araç çağrıları, **ham çıktıları** ve üretilen dosyalar.
(Yalnızca özet saklamak, replanner'ın tamamlanmış bir adımı "yapılmamış" sanıp
yeniden planlamasına — sonsuz döngüye — yol açıyordu; ham çıktı bunu çözer.)

### ReAct şeridi
Önden plan kurmayan tek bir döngü: elindeki bilgiye bak → bir araç çağır → çıktısını
oku → sonraki eyleme karar ver. LangChain'in hazır tool-calling ajanı kullanılır.

### Araçlar
`tools.py`'daki registry'ye bir fonksiyon eklemek yeni bir yetenek kazandırır;
agent kodu değişmez. Mevcut set finans ağırlıklıdır (yfinance: fiyat, temel veriler,
rasyolar, gelir tablosu, bilanço, nakit akışı, analist, haber, teknik göstergeler,
karşılaştırma) + `calculator`, `web_search` (Tavily), `plot_chart`, `visualize_data`.
Prompt'lar **alan bağımsızdır** — yeni alanlarda araç eklendikçe prompt'lara
dokunmak gerekmez.

Her araç çağrısında zorunlu bir `reason` alanı vardır: model o aracı **neden**
seçtiğini yazar. Araç mantığına girmez, yalnızca loga (DÜŞÜNCE) geçer.

### Session hafızası
Aynı sohbetteki önceki turlar (kullanıcı mesajları + ajan cevapları) triyaja ve
şeritlere aktarılır; böylece **"onu 1 yıllık yap"** gibi devam istekleri çözülür.
Hafıza yalnızca konuşma metnidir (ara süreç/araç çıktıları girmez).

---

## Loglama ve gözlemlenebilirlik

Frontend'e yalnızca **nihai cevap** döner; ara sürecin tamamı sunucuda
`logs/sessions/<session_id>.log` dosyasına **anlık** yazılır (koşu sürerken
`tail -f` ile izlenebilir). Her koşu için:

- Triyaj kararı (şerit, alan, seçilen araçlar, gerekçe)
- Her LLM çağrısı: **girdi mesajları, çıktı, model, token (girdi/çıktı), süre**
- ReAct/Plan-Execute döngüsü: **DÜŞÜNCE → EYLEM → GÖZLEM** çerçevesiyle
- Araç çağrıları, çıktıları, üretilen dosyalar
- Özet: toplam süre, LLM çağrısı sayısı, toplam token, aşama başına dağılım

Örnek (kısaltılmış):
```
[  2,658 ms]   ┌─ DÖNGÜ 1 · gpt-4.1 · 1.54 sn · 688 token (girdi 637 / çıktı 51)
                │ DÜŞÜNCE: TSLA hakkında en güncel gelişmeleri öğrenmek istiyorum.
                │ EYLEM  : get_company_news(TSLA)
                │ GÖZLEM : get_company_news → 1.75 sn · 518 karakter
                └────────────────────────────────────────
```

---

## Ses tanıma (STT)

Frontend'deki mikrofon butonu tarayıcıda (MediaRecorder) ses alır, `/api/transcribe`
ucuna yükler; backend Groq Whisper ile metne çevirip döndürür. Metin **giriş
kutusuna yazılır** (otomatik gönderilmez) — gözden geçirip gönderirsin.

Giriş kutusunun üstünde bir **cihaz seçici** ve **canlı seviye göstergesi** vardır:
birden çok mikrofon varsa (ör. dahili + Bluetooth kulaklık) doğru olanı seçebilir ve
mikrofonun sesi duyup duymadığını anında görebilirsin.

---

## Sesli çıktı (TTS)

Metni sese çevirmek için **Coqui XTTS-v2** kullanılır — Türkçe dahil 17 dil, yerel
GPU'da (CUDA) çalışır, **58 yerleşik konuşmacı** ve ses klonlama destekler.

Arayüzde giriş kutusunun üstündeki 🔊 **Sesli çıktı** anahtarı açıkken, ajanın
cevabı **metinle birlikte** seslendirilir. Uzun cevaplar cümle sınırında
**parçalanır** ve arayüz parçaları **sırayla oynatır** — biri biterken diğeri
hazırdır (aşamalı oynatma; kullanıcı ▶'ye basınca başlar).

### Mimari (ayrı süreç/venv YOK)

XTTS-v2 ana venv'le uyumlu (numpy 2.x) olduğundan Chatterbox'ta gereken izole
venv/mikroservis mimarisine gerek kalmadı. Model, backend'in **kendi sürecinde**
(`src/agent_core/tts.py`) ilk sesli istekte **tembel yüklenir**, sonra bellekte
sıcak kalır. Sesli mod kapalıyken torch/TTS hiç import edilmez.

```
Frontend (🔊)  ──►  API :8000  ──►  tts.py (parçala → XTTS-v2, süreç içi)
      ▲                                        │
      └────────  wav parçaları (sırayla oynatılır)  ◄────┘
```

### Kurulum (aynı venv'e ek)

Sesli çıktı paketleri `requirements.txt`'in sonundaki opsiyonel TTS bölümündedir
(CUDA'lı torch dahil, pytorch index'i dosyada tanımlı). Tümünü kurmak için:

```bash
uv pip install --link-mode=copy -r requirements.txt
```

Model (`xtts_v2`, ~1.8 GB) ilk çalıştırmada iner, sonra cache'ten yüklenir. Lisans
onayı `COQUI_TOS_AGREED=1` olarak `tts.py` içinde otomatik ayarlanır.

> **Not:** requirements'ta `transformers<5` sabitlenir — coqui-tts bunu sürümsüz
> istiyor ama transformers 5.x'te XTTS'in kullandığı `isin_mps_friendly` kaldırıldı,
> o yüzden 4.x gerekiyor.

### Ses ayarları

`.env`'den (değişince sunucuyu yeniden başlat):

| Değişken | Varsayılan | Etki |
|----------|:----------:|------|
| `TTS_LANGUAGE` | `tr` | Sentez dili (17 dil) |
| `TTS_SPEAKER` | `Claribel Dervla` | Yerleşik konuşmacı (58 seçenek: Daisy Studious, Gracie Wise…) |
| `TTS_SPEAKER_WAV` | — | Ses **klonlama** için kısa referans wav yolu (verilirse `TTS_SPEAKER` yok sayılır) |
| `TTS_CHUNK_TOKENS` | `120` | Parça başına ~token bütçesi |
| `TTS_MAX_CHUNKS` | `24` | Toplam parça tavanı (aşamalı oynatma) |

Parça-sınırı klik/sessizliği için her parçaya hafif **baş/son sessizlik kırpma +
fade in/out** uygulanır (`TTS_FADE_MS`, `TTS_TRIM_SILENCE`).

> **Neden Chatterbox değil?** Önce Chatterbox denendi ama Türkçe'de otoregresif
> yapısı cümle sonlarında "junk/rambling" üretiyordu ve `numpy<2` gereksinimi ayrı
> bir izole venv+servis zorunlu kılıyordu. XTTS-v2 hem daha stabil hem ana venv'le
> uyumlu; mimari tek sürece indirildi.

Model: [coqui-tts (idiap)](https://github.com/idiap/coqui-ai-TTS) · lisans CPML (ticari değil).

---

## Ortam değişkenleri

Tümü `.env`'de (bkz. [`.env.example`](.env.example)):

| Değişken | Zorunlu | Açıklama |
|----------|:-------:|----------|
| `LLM_API_KEY` | ✅ | Ana model API anahtarı |
| `LLM_MODEL` | ✅ | Model adı (ör. `gpt-4.1`) |
| `LLM_BASE_URL` | ✅* | OpenAI-uyumlu endpoint (*varsayılan OpenAI ise gerekmez) |
| `GROQ_API_KEY` | ✅ | Ses tanıma (STT) için |
| `TAVILY_API_KEY` | — | `web_search` aracı için |
| `TRIAGE_MODEL` | — | Triyaj için ayrı/daha hızlı model (boşsa ana model) |
| `STT_MODEL` | — | Varsayılan `whisper-large-v3` |
| `STT_LANGUAGE` | — | Ör. `tr` (boşsa otomatik tespit) |
| `LLM_STRUCTURED_METHOD` | — | Sağlayıcı `json_schema` desteklemiyorsa `function_calling` |
| `TTS_LANGUAGE` | — | Sesli çıktı dili (varsayılan `tr`) |
| `TTS_SPEAKER` | — | XTTS yerleşik konuşmacı (varsayılan `Claribel Dervla`) |

> Sesli çıktının tüm knob'ları (konuşmacı, klonlama, parçalama) için bkz.
> [Sesli çıktı (TTS)](#sesli-çıktı-tts) ve `.env.example`.

---

## Tasarım kararları

- **Yanlış triyaj için telafi/fallback mekanizması yok** (bilinçli): karar netliği
  ve sadelik önceliklidir. Şeritler arası eskalasyon yoktur.
- **Triyaj cömert araç seçer:** eksik kalan bir aracı alt katman göremeyeceği için,
  gerekebilecek araçlar da listeye eklenir; zamanla loglara bakılarak daraltılır.
- **Session/mesajlar bellekte tutulur** — sunucu yeniden başlayınca sıfırlanır
  (kalıcılık henüz eklenmedi).
- **Prompt'lar alan bağımsızdır:** hiçbir prompt belirli bir araç adına ya da alana
  (finans vb.) demirlenmez; model kararı çalışma anındaki araç kataloğuna bakarak verir.

---

