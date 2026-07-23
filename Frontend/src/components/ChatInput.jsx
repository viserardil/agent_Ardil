import React, { useState, useRef, useEffect } from 'react';
import { Paperclip, Mic, Square, Loader2, ArrowUp, Languages } from 'lucide-react';
import { api } from '../api';

// Ses tanıma dilleri. "auto" = Whisper kendi tespit eder. Yeni dil eklemek için
// {code, label} eklemen yeterli — kod ISO 639-1 iki harfli (tr, es, en, de...).
const LANGUAGES = [
  { code: 'auto', label: 'Otomatik algıla' },
  { code: 'tr', label: 'Türkçe' },
  { code: 'en', label: 'English' },
  { code: 'es', label: 'Español' },
  { code: 'de', label: 'Deutsch' },
  { code: 'fr', label: 'Français' },
  { code: 'it', label: 'Italiano' },
  { code: 'pt', label: 'Português' },
  { code: 'ru', label: 'Русский' },
  { code: 'ar', label: 'العربية' },
  { code: 'zh', label: '中文' },
  { code: 'ja', label: '日本語' }
];

export function ChatInput({ onSendMessage, suggestionChips, disabled, runSummary }) {
  const [text, setText] = useState('');
  const [recording, setRecording] = useState(false);
  const [transcribing, setTranscribing] = useState(false);
  const [micError, setMicError] = useState('');

  // Ses girişi cihaz seçimi + canlı seviye. "Altyazı M.K." halüsinasyonunun sebebi
  // tarayıcının SESSİZ bir varsayılan mikrofon seçmesiydi (ör. bağlı Bluetooth
  // kulaklığın çalışmayan mikrofonu). Kullanıcı doğru cihazı seçebilsin ve seviyeyi
  // canlı görsün diye bunlar eklendi.
  const [devices, setDevices] = useState([]);
  const [deviceId, setDeviceId] = useState('');
  const [level, setLevel] = useState(0);

  // Ses tanıma dili — arayüzden anında değişir, sunucu restart'ı gerekmez.
  // Seçim localStorage'da tutulur (sayfa yenilense de korunur). Varsayılan: Türkçe.
  const [lang, setLang] = useState(() => localStorage.getItem('ardil_stt_lang') || 'tr');
  const changeLang = (code) => {
    setLang(code);
    try {
      localStorage.setItem('ardil_stt_lang', code);
    } catch {
      // localStorage kapalıysa (gizli sekme vb.) sessiz geç
    }
  };

  const recorderRef = useRef(null);
  const chunksRef = useRef([]);
  const audioCtxRef = useRef(null);
  const rafRef = useRef(null);
  const sourceRef = useRef(null);

  const defaultChips = [
    'AAPL güncel fiyatı nedir?',
    'AAPL ile MSFT’yi karşılaştır',
    'MSFT’nin son 6 aylık fiyat grafiğini çiz'
  ];
  const chips = suggestionChips || defaultChips;

  useEffect(() => () => cleanupMeter(), []);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!text.trim() || disabled) return;
    onSendMessage(text);
    setText('');
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleChipClick = (chipText) => {
    if (disabled) return;
    onSendMessage(chipText);
  };

  // --- Sesli giriş -----------------------------------------------------------
  const pickMime = () => {
    const prefs = ['audio/webm', 'audio/mp4', 'audio/ogg'];
    for (const m of prefs) {
      if (window.MediaRecorder?.isTypeSupported?.(m)) return m;
    }
    return '';
  };

  // İzin verildikten SONRA cihaz etiketleri okunabiliyor; listeyi o an tazeliyoruz.
  const refreshDevices = async () => {
    try {
      const all = await navigator.mediaDevices.enumerateDevices();
      setDevices(all.filter((d) => d.kind === 'audioinput'));
    } catch {
      // enumerate başarısızsa sessiz geç; seçici görünmez, varsayılan kullanılır
    }
  };

  // Canlı seviye ölçer: seçili mikrofon gerçekten sinyal alıyor mu göstergesi.
  const startMeter = (stream) => {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    const ctx = new AudioCtx();
    const source = ctx.createMediaStreamSource(stream);
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 512;
    source.connect(analyser);
    audioCtxRef.current = ctx;
    sourceRef.current = source;

    const data = new Uint8Array(analyser.fftSize);
    const tick = () => {
      analyser.getByteTimeDomainData(data);
      let sum = 0;
      for (let i = 0; i < data.length; i++) {
        const x = (data[i] - 128) / 128;
        sum += x * x;
      }
      setLevel(Math.sqrt(sum / data.length));
      rafRef.current = requestAnimationFrame(tick);
    };
    tick();
  };

  const cleanupMeter = () => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    try {
      sourceRef.current?.disconnect();
    } catch {
      // zaten kapalıysa yok say
    }
    sourceRef.current = null;
    if (audioCtxRef.current && audioCtxRef.current.state !== 'closed') {
      audioCtxRef.current.close();
    }
    audioCtxRef.current = null;
    setLevel(0);
  };

  const startRecording = async () => {
    setMicError('');
    if (!navigator.mediaDevices?.getUserMedia) {
      setMicError('Tarayıcı mikrofon kaydını desteklemiyor.');
      return;
    }
    try {
      const constraints = {
        audio: deviceId ? { deviceId: { exact: deviceId } } : true
      };
      const stream = await navigator.mediaDevices.getUserMedia(constraints);

      // Hangi cihaz seçildi? (teşhis için) ve cihaz listesini güncelle.
      const track = stream.getAudioTracks()[0];
      if (track && !deviceId) {
        const settings = track.getSettings?.();
        if (settings?.deviceId) setDeviceId(settings.deviceId);
      }
      refreshDevices();
      startMeter(stream);

      const mime = pickMime();
      const recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
      chunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        cleanupMeter();
        const type = mime || 'audio/webm';
        const ext = type.includes('mp4') ? 'mp4' : type.includes('ogg') ? 'ogg' : 'webm';
        const blob = new Blob(chunksRef.current, { type });

        setTranscribing(true);
        try {
          const { text: transcript } = await api.transcribe(blob, `audio.${ext}`, lang);
          if (transcript) {
            setText((prev) => (prev.trim() ? prev.trim() + ' ' : '') + transcript);
          } else {
            setMicError('Ses anlaşılamadı, tekrar dener misiniz?');
          }
        } catch (err) {
          setMicError(`Ses metne çevrilemedi: ${err.message}`);
        } finally {
          setTranscribing(false);
        }
      };

      recorder.start();
      recorderRef.current = recorder;
      setRecording(true);
    } catch (err) {
      setMicError(`Mikrofona erişilemedi: ${err.message}`);
      cleanupMeter();
    }
  };

  const stopRecording = () => {
    if (recorderRef.current && recording) {
      recorderRef.current.stop();
      setRecording(false);
    }
  };

  const toggleMic = () => {
    if (disabled || transcribing) return;
    if (recording) stopRecording();
    else startRecording();
  };

  const micTitle = recording ? 'Kaydı durdur' : transcribing ? 'Metne çevriliyor…' : 'Sesli giriş';
  // Konuşurken seviye ~0 kalıyorsa cihaz sessiz demektir.
  const silentWhileRecording = recording && level < 0.01;

  return (
    <footer className="chat-footer">
      <style>{`@keyframes ardil-spin { to { transform: rotate(360deg) } }`}</style>

      {micError && (
        <div style={{ color: '#b91c1c', fontSize: 13, marginBottom: 8 }}>{micError}</div>
      )}

      {/* Ses tanıma dili (her zaman) + kayıtta canlı seviye + çok mikrofonda cihaz seçimi */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8, flexWrap: 'wrap' }}>
          <label
            style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: '#64748b' }}
            title="Ses tanıma dili"
          >
            <Languages size={14} />
            <select
              value={lang}
              onChange={(e) => changeLang(e.target.value)}
              disabled={recording}
              style={{
                fontSize: 12, padding: '4px 8px', borderRadius: 6,
                border: '1px solid #e2e8f0', background: '#fff', color: '#334155'
              }}
            >
              {LANGUAGES.map((l) => (
                <option key={l.code} value={l.code}>{l.label}</option>
              ))}
            </select>
          </label>

          {devices.length > 1 && (
            <select
              value={deviceId}
              onChange={(e) => setDeviceId(e.target.value)}
              disabled={recording}
              style={{
                fontSize: 12, padding: '4px 8px', borderRadius: 6,
                border: '1px solid #e2e8f0', background: '#fff', color: '#334155', maxWidth: 260
              }}
              title="Mikrofon cihazı"
            >
              {devices.map((d) => (
                <option key={d.deviceId} value={d.deviceId}>
                  {d.label || 'Mikrofon'}
                </option>
              ))}
            </select>
          )}

          {recording && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, flex: 1, minWidth: 160 }}>
              <div style={{ flex: 1, height: 6, background: '#e2e8f0', borderRadius: 999, overflow: 'hidden' }}>
                <div
                  style={{
                    height: '100%',
                    width: `${Math.min(100, level * 300)}%`,
                    background: silentWhileRecording ? '#f59e0b' : '#22c55e',
                    transition: 'width 80ms linear'
                  }}
                />
              </div>
              <span style={{ fontSize: 12, color: silentWhileRecording ? '#b45309' : '#16a34a', whiteSpace: 'nowrap' }}>
                {silentWhileRecording ? 'ses algılanmıyor' : 'dinliyorum'}
              </span>
            </div>
          )}
        </div>

      {/* Suggestion Chips */}
      <div className="suggestion-chips">
        {chips.map((chip, idx) => (
          <button key={idx} className="chip-btn" onClick={() => handleChipClick(chip)}>
            {chip}
          </button>
        ))}
      </div>

      {/* Main Input Field Container */}
      <form className="input-box-wrapper" onSubmit={handleSubmit}>
        <button type="button" className="input-action-btn" title="Attach file">
          <Paperclip size={18} />
        </button>

        <textarea
          className="chat-textarea"
          placeholder={
            recording
              ? 'Dinliyorum… bitince mikrofona tekrar bas'
              : transcribing
                ? 'Metne çevriliyor…'
                : disabled
                  ? 'Ajan çalışıyor…'
                  : 'Send a message to ardilAgent...'
          }
          rows={1}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
        />

        <button
          type="button"
          className="input-action-btn"
          title={micTitle}
          onClick={toggleMic}
          disabled={disabled || transcribing}
          style={recording ? { color: '#ef4444' } : undefined}
        >
          {transcribing ? (
            <Loader2 size={18} style={{ animation: 'ardil-spin 1s linear infinite' }} />
          ) : recording ? (
            <Square size={18} fill="#ef4444" />
          ) : (
            <Mic size={18} />
          )}
        </button>

        <button
          type="submit"
          className="send-btn"
          disabled={!text.trim() || disabled}
          title="Send message"
        >
          <ArrowUp size={18} />
        </button>
      </form>

      {/* Footer Info — token sayacı son koşudan gelir, sabit değer değil */}
      <div className="footer-disclaimer">
        <span>ardilAgent can make mistakes. Check important info.</span>
        <span>
          {runSummary
            ? `${runSummary.total_tokens?.toLocaleString('tr-TR')} token · ${runSummary.seconds?.toFixed(1)} sn`
            : '—'}
        </span>
      </div>
    </footer>
  );
}
