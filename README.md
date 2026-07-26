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
- **Sesli çıktı (TTS):** Chatterbox Multilingual — Türkçe TTS, yerel/CUDA, izole serviste (opsiyonel)

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
├── run_api.py              # API sunucusunu başlatır (+ sesli çıktı servisini otomatik)
├── voice_service.py        # İZOLE Chatterbox TTS mikroservisi (ayrı venv'de koşar)
├── main.py                 # CLI: şeritleri/triyajı terminalden çalıştır
├── pyproject.toml
├── requirements.txt        # ana bağımlılıklar
├── requirements-voice.txt  # opsiyonel TTS (Chatterbox) — AYRI .venv-voice'a kurulur
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
│   ├── tts.py              # Sesli çıktı: metni parçalayıp voice_service'e HTTP ile yollar
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

> **Opsiyonel — yerel TTS (Chatterbox):** Sesli çıktı istiyorsan ayrı bir izole
> venv (`.venv-voice`) kurulur (torch+CUDA ~2.5 GB). Ana venv'e dokunmaz; kurulum
> adımları için [Sesli çıktı (TTS)](#sesli-çıktı-tts) bölümüne bak. STT/agent için
> gerekmez.

### 2) Frontend

```bash
cd Frontend
npm install
```

---

## Çalıştırma

Sistem en fazla **üç süreçten** oluşur. Sesli çıktı istemiyorsan ilk ikisi yeter:

| Süreç | Port | Ne zaman | Nasıl başlar |
|-------|:----:|----------|--------------|
| **Backend API** (uvicorn) | `8000` | Her zaman | `python run_api.py` |
| **Frontend** (Vite) | `5173` | Her zaman | `Frontend/`'de `npm run dev` |
| **Sesli çıktı servisi** (Chatterbox) | `8756` | Sadece TTS için | `run_api.py` **otomatik** başlatır (`.venv-voice` varsa) |

### 1) Backend (Terminal 1 — proje kökü)

Önce venv'i aktive et (Windows cmd: `.venv\Scripts\activate`), sonra:

```bash
python run_api.py            # http://127.0.0.1:8000
# geliştirme (dosya değişince yeniden başlar):
python run_api.py --reload
```

Açılışta `.venv-voice` bulunursa **sesli çıktı servisi otomatik başlatılır** (log:
`logs/voice_service.log`; model yüklenirken ilk sesli istek biraz bekler).
Sesli çıktıyı hiç istemiyorsan: `python run_api.py --no-voice`.

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

Metni sese çevirmek için **Chatterbox Multilingual** (Resemble AI) kullanılır —
Türkçe dahil 23 dil, yerel GPU'da (CUDA) çalışır, ses klonlama destekler.

Arayüzde giriş kutusunun üstündeki 🔊 **Sesli çıktı** anahtarı açıkken, ajanın
cevabı **metinle birlikte** seslendirilir. Uzun cevaplar cümle sınırında
**parçalanır** (parça başına ~token bütçesi, en çok birkaç düzine parça) ve arayüz
parçaları **sırayla oynatır** — biri biterken diğeri hazırdır (aşamalı oynatma).

### Neden ayrı bir venv/süreç?

Chatterbox `numpy<2` gerektiriyor; ana AgentArdil venv'i ise numpy 2.x kullanıyor
(scipy vb. ona göre derli). İkisi tek venv'de **binary olarak barışmıyor**. Bu
yüzden Chatterbox **kendi izole venv'inde** (`.venv-voice`) **ayrı bir süreç** olarak
koşar (`voice_service.py`, port `8756`); ana API ona HTTP ile konuşur
(`src/agent_core/tts.py`). Böylece ağır/çakışan TTS bağımlılıkları ana agent
sürecinden tamamen izole kalır ve sesli mod kapalıyken hiçbir maliyeti olmaz.

```
Frontend (🔊)  ──►  API :8000  ──►  tts.py  ──HTTP──►  voice_service :8756
                                   (parçalama)         (.venv-voice, Chatterbox)
      ▲                                                        │
      └──────────────  wav parçaları (sırayla oynatılır)  ◄────┘
```

### Kurulum (izole venv)

```bash
# 1) İzole venv (ana venv'e dokunmaz)
uv venv .venv-voice --python 3.11

# 2) CUDA'lı torch (RTX serisi -> cu124); OneDrive'da --link-mode=copy şart
uv pip install --link-mode=copy torch==2.6.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124 --python .venv-voice

# 3) Chatterbox + sabitler (bkz. requirements-voice.txt)
uv pip install --link-mode=copy -r requirements-voice.txt --python .venv-voice
```

Model ağırlıkları (`ResembleAI/chatterbox`, ~1-2 GB) ilk çalıştırmada Hugging
Face'ten iner, sonra cache'ten yüklenir.

> **Not:** `requirements-voice.txt`, `setuptools<81` sabitler — Chatterbox'ın ses
> filigranı (perth) `pkg_resources` kullanıyor ve setuptools 81+ bunu kaldırdı.

### Çalıştırma

`.venv-voice` mevcutsa `python run_api.py` **servisi otomatik başlatır** (ayrı
terminal gerekmez). İstersen elle de başlatabilirsin:

```bash
.venv-voice/Scripts/python voice_service.py     # http://127.0.0.1:8756
```

### Stabilite / ses ayarları

Sesin tonunu ya da kalitesini ayarlamak istersen `.env`'den (değişince servisi
yeniden başlat):

| Değişken | Varsayılan | Etki |
|----------|:----------:|------|
| `TTS_TEMPERATURE` | `0.5` | Çok düşük (0.2) run-on/junk üretir, yüksek dalgalandırır — 0.5 dengeli |
| `TTS_CFG_WEIGHT` | `0.7` | **Yüksek = metne daha sadık, cümle sonu junk'ı azaltır** (hâlâ varsa 1.0) |
| `TTS_EXAGGERATION` | `0.5` | Düşük = daha nötr/sakin ton |
| `TTS_LANGUAGE` | `tr` | Sentez dili |
| `TTS_AUDIO_PROMPT` | — | Ses klonlama için kısa referans wav yolu |
| `VOICE_AUTOSTART` | `1` | `run_api.py` servisi otomatik başlatsın mı |

#### Ayarlama gözlemleri (deneyerek bulundu)

Chatterbox **otoregresif** bir model; kalite büyük ölçüde üretim parametrelerine
bağlı. Türkçe'de yaşadıklarımız:

- **Cümle sonu "junk" (rambling):** Model bazen cümle bittikten sonra kısa bir
  gürültü/mırıldanma ekliyor. Asıl sebebi **çok düşük temperature** (ör. 0.2):
  otoregresif üretim fazla açgözlü olunca run-on yapıyor. **0.5** bunu belirgin
  azalttı. Ek olarak **`cfg_weight`'i 0.7'ye çıkarmak** (metne sadakat) junk'ı
  daha da düşürdü. Bu, Freya'daki (flow-matching) mantığın **tersi** — orada daha
  fazla adım stabilize ediyordu; burada aşırı düşük temperature *bozuyor*.
- **Parça-sınırı artefaktları:** Parçalar sırayla çalınırken dikişte klik/kuyruk
  duyulabiliyordu. `voice_service.py` her parçaya **baş/son sessizlik kırpma +
  kısa fade in/out** uyguluyor; ayrıca konuşmadan sonra **boşlukla ayrılmış kısa
  junk segmenti** kesiliyor (`TTS_TRIM_TAIL`). Uzun/rambling tipi junk'ı ise
  DSP güvenle kesemiyor (virgül duraklamasıyla karışıyor) — orada çözüm yukarıdaki
  üretim parametreleri.
- Junk **stokastik**: bazı üretimlerde hiç yok, bazılarında var. Parametreler
  sıklığını düşürür, %100 sıfırlamaz.

Model kaynağı: [ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox).

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
| `TTS_TEMPERATURE` | — | Sesli çıktı: düşük = daha stabil (varsayılan 0.6) |
| `VOICE_AUTOSTART` | — | `run_api.py` sesli çıktı servisini otomatik başlatsın mı (varsayılan açık) |

> Sesli çıktının tüm knob'ları (exaggeration, cfg_weight, dil, klonlama, port) için
> bkz. [Sesli çıktı (TTS)](#sesli-çıktı-tts) ve `.env.example`.

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

🤖 Bu proje [Claude Code](https://claude.com/claude-code) ile geliştirildi.
