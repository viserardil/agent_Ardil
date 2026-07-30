"""Türkçe metni SESLENDİRMEDEN ÖNCE okunur hale getirir (TTS ön-ucu).

Neden gerekli? XTTS'in kendi Türkçe temizleyicisi eksik: ondalık ayracı (virgül)
genişletilmiyor (kaynakta ``if lang != "tr"`` ile atlanmış), kısaltma sözlüğünde
sadece 3 giriş var (b./byk./dr.), birimler (km, kg, TL) ve akronimler (TBMM) hiç
işlenmiyor, cümle sonundaki nokta yüzünden "%8,7." → "yüzde sekiz yedinci" gibi
sıra sayısı hataları çıkıyor. Bunlar modele değil ÖN-UCA ait sorunlar; bu yüzden
katman modelden bağımsız: XTTS yerine başka bir motora geçilse de aynen çalışır.

Tek genel API: ``normalize(text)``. Bağımsız birimleri (sayı → yazı) ayrıca
kullanmak isteyen için ``number_to_words`` / ``ordinal_to_words`` da dışa açık.

Sıra önemlidir: tarih → saat → yüzde → para/birim → ondalık/binlik sayı →
sıra sayısı → kısaltma → akronim. Erken adımlar rakamları tükettiği için sonraki
kurallar yanlış eşleşemez.
"""

from __future__ import annotations

import os
import re

__all__ = ["normalize", "number_to_words", "ordinal_to_words", "has_spelled_acronym"]

# --- Sayı → yazı --------------------------------------------------------------
_ONES = ["", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"]
_TENS = ["", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"]
_SCALES = [(10**12, "trilyon"), (10**9, "milyar"), (10**6, "milyon"), (10**3, "bin")]

# Sıra sayısı, sayının SON kelimesine eklenir: "bin iki yüz otuz dört" → "...dördüncü".
_ORDINALS = {
    "sıfır": "sıfırıncı", "bir": "birinci", "iki": "ikinci", "üç": "üçüncü",
    "dört": "dördüncü", "beş": "beşinci", "altı": "altıncı", "yedi": "yedinci",
    "sekiz": "sekizinci", "dokuz": "dokuzuncu", "on": "onuncu", "yirmi": "yirminci",
    "otuz": "otuzuncu", "kırk": "kırkıncı", "elli": "ellinci", "altmış": "altmışıncı",
    "yetmiş": "yetmişinci", "seksen": "sekseninci", "doksan": "doksanıncı",
    "yüz": "yüzüncü", "bin": "bininci", "milyon": "milyonuncu",
    "milyar": "milyarıncı", "trilyon": "trilyonuncu",
}


def _under_thousand(n: int) -> str:
    parts = []
    hundreds, rest = divmod(n, 100)
    if hundreds:
        parts.append("yüz" if hundreds == 1 else f"{_ONES[hundreds]} yüz")
    tens, ones = divmod(rest, 10)
    if tens:
        parts.append(_TENS[tens])
    if ones:
        parts.append(_ONES[ones])
    return " ".join(parts)


def number_to_words(n: int) -> str:
    """Tam sayıyı Türkçe okunuşuna çevirir (1234 → 'bin iki yüz otuz dört')."""
    if n < 0:
        return "eksi " + number_to_words(-n)
    if n == 0:
        return "sıfır"
    parts = []
    for value, name in _SCALES:
        q, n = divmod(n, value)
        if q:
            # "bir bin" denmez ama "bir milyon" denir.
            prefix = "" if (value == 1000 and q == 1) else _under_thousand(q) + " "
            parts.append(f"{prefix}{name}")
    if n:
        parts.append(_under_thousand(n))
    return " ".join(parts)


def ordinal_to_words(n: int) -> str:
    """Sıra sayısı (3 → 'üçüncü', 1234 → 'bin iki yüz otuz dördüncü')."""
    words = number_to_words(n).split()
    words[-1] = _ORDINALS.get(words[-1], words[-1] + "inci")
    return " ".join(words)


# Ayraç KAYNAKTAKİ gibi seslendirilir: "33,91" → virgül, "33.91" → nokta.
# (Dil bilgisi kuralı hepsine "virgül" der ama finans/borsa konuşmasında nokta ile
# yazılan değer "nokta" diye okunuyor; yazımı korumak daha az şaşırtıcı.)
_DECIMAL_WORDS = {",": "virgül", ".": "nokta"}


def _decimal_to_words(int_part: int, frac: str, sep: str = ",") -> str:
    """Ondalık: '12,5' → 'on iki virgül beş', '33.91' → 'otuz üç nokta doksan bir'.

    Baştaki sıfırlar anlamlı olduğu için (0,05 ≠ 0,5) kesir kısmı sıfırla
    başlıyorsa rakam rakam okunur; değilse tek sayı olarak.
    """
    if frac.startswith("0"):
        tail = " ".join(_ONES[int(d)] if d != "0" else "sıfır" for d in frac)
    else:
        tail = number_to_words(int(frac))
    return f"{number_to_words(int_part)} {_DECIMAL_WORDS.get(sep, 'virgül')} {tail}"


# --- Sözlükler ----------------------------------------------------------------
_MONTHS = ["", "ocak", "şubat", "mart", "nisan", "mayıs", "haziran",
           "temmuz", "ağustos", "eylül", "ekim", "kasım", "aralık"]

# Nokta ile biten kısaltmalar (küçük/büyük harf duyarsız).
_ABBREVIATIONS = {
    "vb": "ve benzeri", "vs": "vesaire", "örn": "örneğin", "bkz": "bakınız",
    "dr": "doktor", "prof": "profesör", "doç": "doçent", "av": "avukat",
    "sn": "sayın", "mah": "mahalle", "cad": "cadde", "sok": "sokak",
    "no": "numara", "tel": "telefon", "yy": "yüzyıl", "yrd": "yardımcı",
    "müh": "mühendis", "öğr": "öğretim", "gör": "görevlisi", "bkn": "bakınız",
}

# Sayıdan SONRA gelen birimler. Kesme işaretli ekler korunur: "dk'da" → "dakikada".
_UNITS = {
    "km": "kilometre", "cm": "santimetre", "mm": "milimetre", "kg": "kilogram",
    "gr": "gram", "mg": "miligram", "lt": "litre", "ml": "mililitre",
    "sn": "saniye", "dk": "dakika", "sa": "saat", "mb": "megabayt",
    "gb": "gigabayt", "tb": "terabayt", "kb": "kilobayt", "hz": "hertz",
    "khz": "kilohertz", "mhz": "megahertz", "ghz": "gigahertz",
    "tl": "lira", "usd": "dolar", "eur": "euro", "gbp": "sterlin",
    "m²": "metrekare", "km²": "kilometrekare", "m³": "metreküp",
}

_CURRENCY_SYMBOLS = {"₺": "lira", "$": "dolar", "€": "euro", "£": "sterlin"}

# Harf harf okunması gereken TÜRKÇE akronimler. Sesli harf içerenler (TÜİK,
# ASELSAN) zaten kelime gibi okunduğu için listeye alınmaz; sadece belirsizler.
_ACRONYMS = {
    "ABD": "a be de", "AB": "a be", "KDV": "ke de ve", "ÖTV": "ö te ve",
    "THY": "te he ye", "TRT": "te re te", "AŞ": "a şe", "KKB": "ke ke be",
}

# İNGİLİZCE kısaltmalar İngilizce harf adlarıyla okunur: NVDA "en vi di ey",
# AMD "ey em di" — Türkçe harf adlarıyla ("ne ve de a") yanlış duyuluyor.
# Buraya yalnızca gerçekten HARF HARF söylenenler girer; "Fed", "Nasdaq" gibi
# kelime olarak okunanlar girmez.
_EN_ACRONYMS = {
    "AAPL", "MSFT", "AMD", "NVDA", "GOOG", "GOOGL", "AMZN", "META", "TSLA",
    "NFLX", "INTC", "QCOM", "AVGO", "ORCL", "CRM", "IBM", "JPM", "BABA",
    "ETF", "IPO", "CEO", "CFO", "CTO", "GPU", "CPU", "API", "AI", "USB",
    "SEC", "NYSE", "SPX", "NDX", "EPS", "ROI", "ROE", "IT", "HR", "PDF", "URL",
}

# İngiliz alfabesinin harf adları, Türkçe yazımla (XTTS Türkçe okuduğu için).
_EN_LETTERS = {
    "a": "ey", "b": "bi", "c": "si", "d": "di", "e": "i", "f": "ef", "g": "ci",
    "h": "eyç", "i": "ay", "j": "cey", "k": "key", "l": "el", "m": "em",
    "n": "en", "o": "ov", "p": "pi", "q": "kiyu", "r": "ar", "s": "es",
    "t": "ti", "u": "yu", "v": "vi", "w": "dabılyu", "x": "eks", "y": "vay",
    "z": "zi",
}

# Kelime olarak okunan ama İngilizce telaffuzu olan markalar: Türkçe yazımıyla
# okununca bozuluyor (NVIDIA → "Nühidya"). Türkçe fonetik karşılığını yazıyoruz.
# Yalnızca varsayılan okunuşu YANLIŞ olanlar listede (Tesla, Meta, Intel doğru
# okunduğu için yok).
# Her karşılık sentezlenip STT ile geri okunarak doğrulandı (marka adı geri
# geliyor mu diye). "Amazon" listede YOK: Türkçe yazımıyla zaten doğru okunuyor,
# "emazın" ise "Emaz'ın" duyuluyordu.
_BRANDS = {
    "nvidia": "envidya", "apple": "epıl", "google": "gugıl",
    "microsoft": "maykrosoft", "netflix": "netfliks", "nasdaq": "nazdak",
    "iphone": "ayfon", "youtube": "yutub",
    "broadcom": "brodkom", "qualcomm": "kualkom",
}

# Hecelenen akronimlerde harfler arasına konan ayraç. Boşlukla ayrılınca XTTS
# harfleri birbirine yapıştırıp çok hızlı okuyor; virgül belirgin bir duraklama
# koyuyor. Ölçüm (4 tekrar ort., yalın "te be me me"): boşluk 1.42 s, virgül
# 2.58 s, nokta 4.88 s. Nokta harf başına ~1.2 s ile fazla kesik kaldığı için
# varsayılan virgül; TTS_ACRONYM_SEP ile değiştirilebilir (ör. ". ").
_ACRONYM_SEP = os.getenv("TTS_ACRONYM_SEP", ", ")

_LETTERS = {
    "a": "a", "b": "be", "c": "ce", "ç": "çe", "d": "de", "e": "e", "f": "fe",
    "g": "ge", "ğ": "yumuşak ge", "h": "he", "ı": "ı", "i": "i", "j": "je",
    "k": "ke", "l": "le", "m": "me", "n": "ne", "o": "o", "ö": "ö", "p": "pe",
    "q": "kü", "r": "re", "s": "se", "ş": "şe", "t": "te", "u": "u", "ü": "ü",
    "v": "ve", "w": "çift ve", "x": "iks", "y": "ye", "z": "ze",
}

_VOWELS = set("aeıioöuüAEIİOÖUÜ")
_BACK, _FRONT = set("aıou"), set("eiöü")
_UNVOICED = set("fstkçşhp")

# Kesme işaretli ekin YENİDEN üretilmesi. "USD'ye" → "dolar" + "ye" = "dolarye"
# olmaz; kök değiştiği için ek de değişmeli ("dolara"). Yazılı eki türüne göre
# sınıflandırıp yeni kökün ünlü uyumuna göre baştan kuruyoruz. Sınıflandıramazsak
# ek olduğu gibi bırakılır (yanlış tahmin etmektense dokunmamak yeğ).
_SUFFIX_CLASSES = {
    "loc": ("de", "da", "te", "ta"),
    "abl": ("den", "dan", "ten", "tan"),
    "dat": ("e", "a", "ye", "ya"),
    "acc": ("i", "ı", "u", "ü", "yi", "yı", "yu", "yü"),
    "gen": ("in", "ın", "un", "ün", "nin", "nın", "nun", "nün"),
    "plu": ("ler", "lar"),
    "wth": ("li", "lı", "lu", "lü"),
    "lik": ("lik", "lık", "luk", "lük"),
}
_SUFFIX_LOOKUP = {form: cls for cls, forms in _SUFFIX_CLASSES.items() for form in forms}


def _last_vowel(word: str) -> str:
    for ch in reversed(word.lower()):
        if ch in _BACK or ch in _FRONT:
            return ch
    return "a"


def _harmonize(suffix: str, stem: str) -> str:
    """Eki yeni kökün ünlü/ünsüz uyumuna göre yeniden üretir."""
    cls = _SUFFIX_LOOKUP.get(suffix.lower())
    if cls is None:
        return suffix
    v = _last_vowel(stem)
    two = "a" if v in _BACK else "e"                       # 2'li uyum (a/e)
    four = {"a": "ı", "ı": "ı", "o": "u", "u": "u",        # 4'lü uyum (ı/i/u/ü)
            "e": "i", "i": "i", "ö": "ü", "ü": "ü"}[v]
    last = stem[-1].lower()
    hard = "t" if last in _UNVOICED else "d"               # ünsüz benzeşmesi
    vowel_final = last in _VOWELS
    if cls == "loc":
        return f"{hard}{two}"
    if cls == "abl":
        return f"{hard}{two}n"
    if cls == "dat":
        return f"y{two}" if vowel_final else two
    if cls == "acc":
        return f"y{four}" if vowel_final else four
    if cls == "gen":
        return f"n{four}n" if vowel_final else f"{four}n"
    if cls == "plu":
        return f"l{two}r"
    if cls == "wth":
        return f"l{four}"
    return f"l{four}k"  # lik


# --- Kurallar -----------------------------------------------------------------
_RE_DATE = re.compile(r"(?<!\d)(\d{1,2})[./](\d{1,2})[./](\d{4})(?!\d)")
_RE_TIME = re.compile(r"(?<!\d)(\d{1,2}):(\d{2})(?!\d)('\w+)?")

# Sayı çekirdeği — hem Türkçe (1.234,56) hem İngilizce (1,234.56) biçimi.
# Tam sayı kısmı 3'erli GRUPLARDAN oluşuyorsa (1.234 / 1,234,567) ayraç binliktir
# ve atılır; geri kalan tek ayraç ondalıktır ve "virgül" diye okunur. Böylece
# araç çıktısından gelen ham float'lar (33.91) doğru okunur.
# (?!\d) şart: onsuz "34.2567" hatalı biçimde "34.256" + "7" diye bölünüyor.
_NUM_CORE = (r"(\d{1,3}(?:\.\d{3})+(?!\d)|\d{1,3}(?:,\d{3})+(?!\d)|\d+)"
             r"(?:([.,])(\d+))?")
# Kesme işaretli ek her üçünde de yakalanır ("2025'te", "%42,5'ten", "$40'a");
# yakalanmazsa kesme işareti metinde kalıp "beş'ten" gibi okunuyor.
_SUFFIX = r"(?:['’](\w+))?"
_RE_PERCENT = re.compile(r"%\s*" + _NUM_CORE + _SUFFIX)
# Simge sayıdan önce ("$40") ya da sonra ("100₺") gelebilir; ikisi de yakalanır.
_RE_CURRENCY_SYM = re.compile(r"([₺$€£])\s*" + _NUM_CORE + _SUFFIX)
_RE_CURRENCY_POST = re.compile(_NUM_CORE + r"\s*([₺$€£])" + _SUFFIX)
_RE_NUMBER = re.compile(r"(?<![\w.,])" + _NUM_CORE + _SUFFIX)
# Sıra sayısı: "3. katta" evet; "…%8,7." (cümle sonu) HAYIR — bu yüzden ardından
# boşluk + küçük harf şartı var. Cümle sonundaki nokta küçük harfle devam etmez.
_RE_ORDINAL_DOT = re.compile(r"(?<![\d,.])(\d{1,4})\.(?=\s+[a-zçğıöşü])")
_RE_ORDINAL_SUF = re.compile(r"(?<!\d)(\d+)['’]?(inci|ıncı|uncu|üncü|nci|ncı|ncu|ncü)\b",
                             re.IGNORECASE)
_RE_UNIT = re.compile(
    r"\b(" + "|".join(sorted(map(re.escape, _UNITS), key=len, reverse=True)) + r")"
    r"(?:['’](\w+))?(?![\wçğıöşüÇĞİÖŞÜ])", re.IGNORECASE)
_RE_ABBREV = re.compile(
    r"\b(" + "|".join(sorted(_ABBREVIATIONS, key=len, reverse=True)) + r")\.",
    re.IGNORECASE)
_RE_ACRONYM = re.compile(r"\b([A-ZÇĞİÖŞÜ]{2,6})(?:['’](\w+))?\b")
_RE_BRAND = re.compile(
    r"\b(" + "|".join(sorted(map(re.escape, _BRANDS), key=len, reverse=True)) + r")"
    r"(?:['’](\w+))?(?![\wçğıöşüÇĞİÖŞÜ])", re.IGNORECASE)
_RE_WS = re.compile(r"\s{2,}")


def _n(int_digits: str, sep: str | None = None, frac: str | None = None) -> str:
    """Sayı çekirdeğini okunuşa çevirir; binlik ayraçları atılır.

    Ayraç kararı regex'te verilir: 3'erli gruplar (1.234 / 1,234,567) binlik,
    geriye kalan tek ayraç ondalıktır ve yazıldığı gibi seslendirilir
    (33.91 → "otuz üç nokta doksan bir", 33,91 → "…virgül…").
    """
    value = int(int_digits.replace(".", "").replace(",", ""))
    # Salt sıfırdan oluşan kesir okunmaz: "1.500,00 TL" → "bin beş yüz lira".
    if not frac or not frac.strip("0"):
        return number_to_words(value)
    return _decimal_to_words(value, frac, sep or ",")


def _sub_date(m: re.Match) -> str:
    day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not 1 <= month <= 12 or not 1 <= day <= 31:
        return m.group(0)
    return f"{number_to_words(day)} {_MONTHS[month]} {number_to_words(year)}"


def _sub_time(m: re.Match) -> str:
    hour, minute = int(m.group(1)), int(m.group(2))
    if hour > 23 or minute > 59:
        return m.group(0)
    minute_words = "" if minute == 0 else " " + number_to_words(minute)
    suffix = m.group(3)
    if suffix:  # "14:30'da" → "on dört otuzda"
        return f"{number_to_words(hour)}{minute_words}{suffix[1:]}"
    return f"{number_to_words(hour)}{minute_words}"


def _sub_number(m: re.Match) -> str:
    words = _n(m.group(1), m.group(2), m.group(3))
    suffix = m.group(4)
    return f"{words}{suffix}" if suffix else words  # "2025'te" → "…beşte"


def _sub_percent(m: re.Match) -> str:
    # Ek sayının okunuşuna takılır ("%42,5'ten" → "…beşten"); okunuş değişmediği
    # için yazılı ek olduğu gibi kullanılır.
    words = "yüzde " + _n(m.group(1), m.group(2), m.group(3))
    suffix = m.group(4)
    return f"{words}{suffix}" if suffix else words


def _sub_currency(m: re.Match) -> str:
    # Burada ek PARA BİRİMİNE takılır ("€50'ye" → "elli euroya"), kök değiştiği
    # için yeniden üretilir.
    unit = _CURRENCY_SYMBOLS[m.group(1)]
    words = f"{_n(m.group(2), m.group(3), m.group(4))} {unit}"
    suffix = m.group(5)
    return f"{words}{_harmonize(suffix, unit)}" if suffix else words


def _sub_currency_post(m: re.Match) -> str:
    """Simge sayıdan sonra: '100₺'den' → 'yüz liradan'."""
    unit = _CURRENCY_SYMBOLS[m.group(4)]
    words = f"{_n(m.group(1), m.group(2), m.group(3))} {unit}"
    suffix = m.group(5)
    return f"{words}{_harmonize(suffix, unit)}" if suffix else words


def _sub_unit(m: re.Match) -> str:
    unit = _UNITS[m.group(1).lower()]
    suffix = m.group(2)
    return f"{unit}{_harmonize(suffix, unit)}" if suffix else unit


# Türkçede hece sonunda gelebilen ünsüz çiftleri (üst, kart, bank…). Bunlarla
# biten akronimler kelime gibi okunur (BIST → "bist"); "pl" gibi Türkçe olmayan
# yığınlar hecelenir (AAPL → "a a pe le").
_VALID_CODAS = {"st", "rt", "nt", "nk", "rk", "lt", "lk", "rs", "ns", "rç",
                "rp", "sk", "şk", "ft", "ht"}


def _is_pronounceable(word: str) -> bool:
    """Akronim kelime gibi okunabilir mi (TÜİK/BIST evet, TBMM/AAPL hayır)?

    Ölçüt: ünlü içerecek, hiçbir yerde 3+ ardışık ünsüz olmayacak ve sondaki
    ünsüz yığını Türkçede hece sonu olabilecek bir çift olacak.
    """
    if not any(ch in _VOWELS for ch in word):
        return False
    runs, cur = [], 0
    for ch in word:
        if ch in _VOWELS:
            runs.append(cur)
            cur = 0
        else:
            cur += 1
    runs.append(cur)
    if max(runs) > 2:
        return False
    if runs[-1] == 2:
        return word[-2:].lower() in _VALID_CODAS
    return True


def _sub_brand(m: re.Match) -> str:
    """İngilizce marka adını Türkçe fonetiğiyle yazar: Nvidia → envidya."""
    stem = _BRANDS[m.group(1).lower()]
    suffix = m.group(2)
    return f"{stem}{_harmonize(suffix, stem)}" if suffix else stem


def _spell_acronym(word: str) -> tuple[str, str]:
    """Bir akronimi harfe/kelimeye çevirir. (seslendirme, uyum-için-son-parça) döner.

    İkinci eleman ek uyumlaması için kullanılır: hecelenmişse SON harf, kelime
    gibi okunuyorsa kelimenin KENDİSİ (bkz. ``_harmonize``).
    """
    spelled = _ACRONYMS.get(word)
    if word in _EN_ACRONYMS:  # İngilizce kısaltma: İngilizce harf adlarıyla
        letters = [_EN_LETTERS[ch.lower()] if ch.lower() in _EN_LETTERS else ch
                   for ch in word]
    elif spelled is None:
        if _is_pronounceable(word):
            return word, word
        letters = [_LETTERS[ch.lower()] if ch.lower() in _LETTERS else ch
                   for ch in word]
    else:
        letters = spelled.split()  # sözlükteki karşılık da aynı ayraçla birleşir
    return _ACRONYM_SEP.join(letters), letters[-1]


def _sub_acronym(m: re.Match) -> str:
    word, suffix = m.group(1), m.group(2)
    spelled, last = _spell_acronym(word)
    # Ek son HARFİN okunuşuna göre uyumlanır ("TBMM'de" → "…me, mede").
    return f"{spelled}{_harmonize(suffix, last)}" if suffix else spelled


# Bitişik iki akronim/oran kısaltması ("F/K", "PD/DD"): finansta çok yaygın ama
# \b sınırı "/" karakterinde bittiği için genel akronim kuralı her iki tarafı
# AYRI yakalayıp aradaki "/" karakterini olduğu gibi bırakıyordu. Burada "/" yı
# "bölü" diye okuyup her iki tarafı da kendi akronim mantığıyla (Türkçe/İngilizce
# hecelenmiş ya da kelime gibi) çeviriyoruz: "F/K" → "fe bölü ke", "PD/DD" →
# "pe, de bölü de, de".
_RE_ACRONYM_RATIO = re.compile(r"\b([A-ZÇĞİÖŞÜ]{1,6})/([A-ZÇĞİÖŞÜ]{1,6})\b")


def _sub_acronym_ratio(m: re.Match) -> str:
    left, _ = _spell_acronym(m.group(1))
    right, _ = _spell_acronym(m.group(2))
    return f"{left} bölü {right}"


# Hecelenmiş kısaltma imzası: ayraçla bağlı EN AZ İKİ harf adı ("a, me, de").
# tts.py bunu, o parçadaki duraksamaları kısaltmak için kullanıyor — heceleme
# olmayan parçaların doğal duraklarına dokunulmasın diye.
_LETTER_NAMES = sorted(set(_LETTERS.values()) | set(_EN_LETTERS.values()),
                       key=len, reverse=True)
_SPELLED_RE = re.compile(
    r"(?<![\wçğıöşü])(?:{n})(?:{s}(?:{n}))+".format(
        n="|".join(map(re.escape, _LETTER_NAMES)),
        s=(re.escape(_ACRONYM_SEP.strip()) + r"\s*") if _ACRONYM_SEP.strip() else r"\s+",
    )
)


def has_spelled_acronym(text: str) -> bool:
    """Metinde harf harf okunan bir kısaltma var mı?"""
    return bool(_SPELLED_RE.search(text))


def normalize(text: str) -> str:
    """Türkçe metni seslendirmeye uygun hale getirir (rakam/kısaltma → kelime)."""
    if not text:
        return text
    text = _RE_ABBREV.sub(lambda m: _ABBREVIATIONS[m.group(1).lower()], text)
    text = _RE_DATE.sub(_sub_date, text)
    text = _RE_TIME.sub(_sub_time, text)
    text = _RE_PERCENT.sub(_sub_percent, text)
    text = _RE_CURRENCY_SYM.sub(_sub_currency, text)
    text = _RE_CURRENCY_POST.sub(_sub_currency_post, text)
    text = _RE_ORDINAL_SUF.sub(lambda m: ordinal_to_words(int(m.group(1))), text)
    text = _RE_ORDINAL_DOT.sub(lambda m: ordinal_to_words(int(m.group(1))), text)
    text = _RE_NUMBER.sub(_sub_number, text)
    text = _RE_UNIT.sub(_sub_unit, text)
    text = _RE_BRAND.sub(_sub_brand, text)  # akronimden ÖNCE: "NVIDIA" da yakalanır
    # Oran kısaltmaları ("F/K", "PD/DD") tekil akronim kuralından ÖNCE: yoksa "/"
    # bir kelime sınırı sayılıp taraflar ayrı yakalanır, "/" karakteri kalır.
    text = _RE_ACRONYM_RATIO.sub(_sub_acronym_ratio, text)
    text = _RE_ACRONYM.sub(_sub_acronym, text)
    return _RE_WS.sub(" ", text).strip()
