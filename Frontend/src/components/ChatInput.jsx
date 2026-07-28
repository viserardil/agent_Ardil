import React, { useState, useRef, useEffect } from 'react';
import { Paperclip, Mic, Square, ArrowUp, Languages, Volume2, VolumeX } from 'lucide-react';

// Ses tanıma dilleri (canlı STT şu an backend'de STT_STREAM_LANG ile ayarlanır;
// seçici ileride streaming'e de bağlanacak).
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

// AudioWorklet işlemcisi: mikrofon karelerini biriktirip ~100ms'lik (1600 örnek
// @16kHz) Float32 PCM parçaları olarak ana iş parçacığına yollar. Ayrı dosya
// yerine blob URL ile yüklenir (Vite paketlemesine takılmaz).
const WORKLET_CODE = `
class PCMProc extends AudioWorkletProcessor {
  constructor(){ super(); this.buf=[]; this.target=1600; }
  process(inputs){
    const ch = inputs[0] && inputs[0][0];
    if(ch){
      for(let i=0;i<ch.length;i++) this.buf.push(ch[i]);
      while(this.buf.length>=this.target){
        this.port.postMessage(Float32Array.from(this.buf.splice(0,this.target)));
      }
    }
    return true;
  }
}
registerProcessor('pcm-proc', PCMProc);
`;

export function ChatInput({ onSendMessage, suggestionChips, disabled, runSummary }) {
  const [text, setText] = useState('');
  const [recording, setRecording] = useState(false);
  const [micError, setMicError] = useState('');

  const [devices, setDevices] = useState([]);
  const [deviceId, setDeviceId] = useState('');
  const [level, setLevel] = useState(0);

  const [lang, setLang] = useState(() => localStorage.getItem('ardil_stt_lang') || 'tr');
  const changeLang = (code) => {
    setLang(code);
    try {
      localStorage.setItem('ardil_stt_lang', code);
    } catch {
      // localStorage kapalıysa sessiz geç
    }
  };

  // Sesli çıktı modu (XTTS).
  const [voiceOut, setVoiceOut] = useState(() => localStorage.getItem('ardil_tts_on') === '1');
  const toggleVoiceOut = () => {
    setVoiceOut((prev) => {
      const next = !prev;
      try {
        localStorage.setItem('ardil_tts_on', next ? '1' : '0');
      } catch {
        // localStorage kapalıysa sessiz geç
      }
      return next;
    });
  };

  // Canlı STT için kaynaklar
  const wsRef = useRef(null);
  const audioCtxRef = useRef(null);
  const streamRef = useRef(null);
  const nodeRef = useRef(null);
  const analyserRef = useRef(null);
  const rafRef = useRef(null);
  const baseTextRef = useRef('');   // streaming başlamadan önceki metin (üzerine eklenir)

  const defaultChips = [
    'AAPL güncel fiyatı nedir?',
    'AAPL ile MSFT’yi karşılaştır',
    'MSFT’nin son 6 aylık fiyat grafiğini çiz'
  ];
  const chips = suggestionChips || defaultChips;

  useEffect(() => () => stopLive(), []);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!text.trim() || disabled) return;
    if (recording) stopLive();
    onSendMessage(text, voiceOut);
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
    onSendMessage(chipText, voiceOut);
  };

  // --- Canlı (streaming) sesli giriş ----------------------------------------
  const refreshDevices = async () => {
    try {
      const all = await navigator.mediaDevices.enumerateDevices();
      setDevices(all.filter((d) => d.kind === 'audioinput'));
    } catch {
      // enumerate başarısızsa sessiz geç
    }
  };

  const startLive = async () => {
    setMicError('');
    if (!navigator.mediaDevices?.getUserMedia) {
      setMicError('Tarayıcı mikrofonu desteklemiyor.');
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: deviceId ? { deviceId: { exact: deviceId } } : true
      });
      streamRef.current = stream;
      const track = stream.getAudioTracks()[0];
      if (track && !deviceId) {
        const s = track.getSettings?.();
        if (s?.deviceId) setDeviceId(s.deviceId);
      }
      refreshDevices();

      // 16 kHz AudioContext — Whisper/silero bunu bekler (modern Chrome/Edge destekler).
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      const ctx = new AudioCtx({ sampleRate: 16000 });
      audioCtxRef.current = ctx;
      const src = ctx.createMediaStreamSource(stream);

      // Canlı seviye ölçer (mikrofon sinyal alıyor mu)
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 512;
      src.connect(analyser);
      analyserRef.current = analyser;
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

      // PCM akıtan worklet
      const blobUrl = URL.createObjectURL(new Blob([WORKLET_CODE], { type: 'application/javascript' }));
      await ctx.audioWorklet.addModule(blobUrl);
      URL.revokeObjectURL(blobUrl);
      const node = new AudioWorkletNode(ctx, 'pcm-proc');
      nodeRef.current = node;

      // WebSocket (/ws/stt — Vite proxy backend'e yönlendirir)
      const proto = location.protocol === 'https:' ? 'wss' : 'ws';
      const ws = new WebSocket(`${proto}://${location.host}/ws/stt`);
      ws.binaryType = 'arraybuffer';
      wsRef.current = ws;

      // Var olan metnin üzerine ekle
      baseTextRef.current = text.trim() ? text.trim() + ' ' : '';

      ws.onmessage = (e) => {
        let d;
        try {
          d = JSON.parse(e.data);
        } catch {
          return;
        }
        if (d.error) {
          setMicError('Canlı STT hatası: ' + d.error);
          return;
        }
        const c = d.committed || '';
        const i = d.interim || '';
        // committed (kesin) + interim (geçici) canlı olarak kutuya
        setText((baseTextRef.current + c + (i ? (c ? ' ' : '') + i : '')).trimStart());
      };
      ws.onerror = () => setMicError('Canlı STT bağlantı hatası — backend açık mı?');

      node.port.onmessage = (e) => {
        if (ws.readyState === 1) ws.send(e.data.buffer || e.data);
      };
      src.connect(node);
      node.connect(ctx.destination); // process çıktı yazmadığı için sessiz — echo yok

      setRecording(true);
    } catch (err) {
      setMicError('Mikrofona erişilemedi: ' + err.message);
      stopLive();
    }
  };

  const stopLive = () => {
    setRecording(false);
    try {
      wsRef.current?.close();
    } catch {
      // yok say
    }
    wsRef.current = null;
    try {
      if (nodeRef.current) nodeRef.current.port.onmessage = null;
      nodeRef.current?.disconnect();
    } catch {
      // yok say
    }
    nodeRef.current = null;
    try {
      analyserRef.current?.disconnect();
    } catch {
      // yok say
    }
    analyserRef.current = null;
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    try {
      streamRef.current?.getTracks().forEach((t) => t.stop());
    } catch {
      // yok say
    }
    streamRef.current = null;
    if (audioCtxRef.current && audioCtxRef.current.state !== 'closed') {
      try {
        audioCtxRef.current.close();
      } catch {
        // yok say
      }
    }
    audioCtxRef.current = null;
    setLevel(0);
  };

  const toggleMic = () => {
    if (disabled) return;
    if (recording) stopLive();
    else startLive();
  };

  const micTitle = recording ? 'Canlı dinlemeyi durdur' : 'Canlı sesli giriş';
  const silentWhileRecording = recording && level < 0.01;

  return (
    <footer className="chat-footer">
      <style>{`@keyframes ardil-spin { to { transform: rotate(360deg) } }`}</style>

      {micError && (
        <div style={{ color: '#b91c1c', fontSize: 13, marginBottom: 8 }}>{micError}</div>
      )}

      {/* Dil + sesli çıktı + cihaz + canlı seviye */}
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

        <button
          type="button"
          onClick={toggleVoiceOut}
          title={voiceOut ? 'Sesli çıktı açık — kapatmak için tıkla' : 'Sesli çıktı kapalı — açmak için tıkla'}
          aria-pressed={voiceOut}
          style={{
            display: 'flex', alignItems: 'center', gap: 6, fontSize: 12,
            padding: '4px 10px', borderRadius: 6, cursor: 'pointer',
            border: `1px solid ${voiceOut ? '#22c55e' : '#e2e8f0'}`,
            background: voiceOut ? '#f0fdf4' : '#fff',
            color: voiceOut ? '#16a34a' : '#64748b'
          }}
        >
          {voiceOut ? <Volume2 size={14} /> : <VolumeX size={14} />}
          <span>Sesli çıktı</span>
        </button>

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
              {silentWhileRecording ? 'ses algılanmıyor' : 'canlı dinliyorum…'}
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

      {/* Giriş kutusu */}
      <form className="input-box-wrapper" onSubmit={handleSubmit}>
        <button type="button" className="input-action-btn" title="Attach file">
          <Paperclip size={18} />
        </button>

        <textarea
          className="chat-textarea"
          placeholder={
            recording
              ? 'Canlı dinliyorum… konuş, yazı anında beliriyor'
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
          disabled={disabled}
          style={recording ? { color: '#ef4444' } : undefined}
        >
          {recording ? <Square size={18} fill="#ef4444" /> : <Mic size={18} />}
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

      {/* Footer Info */}
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
