import React, { useState, useRef, useEffect } from 'react';
import {
  Activity,
  LayoutGrid,
  Megaphone,
  CreditCard,
  Copy,
  Check,
  Play,
  Pause,
  Volume2
} from 'lucide-react';
import { Logo } from './Logo';

// Sesli çıktı oynatıcısı: backend cevabı parça parça (aşamalı) döndürür
// (run.audio_urls). Parçalar SIRAYLA çalınır — biri bitince otomatik diğerine
// geçilir; böylece uzun cevaplarda ilk parça hazır olur olmaz dinlenmeye başlanır.
// Tarayıcı otomatik oynatmayı engellerse kullanıcı oynat düğmesiyle başlatır.
function AudioSequence({ urls }) {
  const [idx, setIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const audioRef = useRef(null);
  // Sonraki parçaya OTOMATİK geçiş yalnızca kullanıcının başlattığı bir sıra
  // ilerlemesinde olsun. Mount/sayfa yenilemede otomatik çalma OLMAZ.
  const autoAdvance = useRef(false);

  // idx değişince: sadece kullanıcı zaten çalıyorken (sıra ilerlemesi) sonraki
  // parçayı çal. İlk render'da ve sayfa yeniden açıldığında OTOMATİK ÇALMAZ —
  // böylece tüm sesli mesajlar bir anda başlamaz; kullanıcı basınca başlar.
  useEffect(() => {
    if (!autoAdvance.current) return;
    autoAdvance.current = false;
    const el = audioRef.current;
    if (!el) return;
    el.currentTime = 0;
    el.play().then(() => setPlaying(true)).catch(() => setPlaying(false));
  }, [idx]);

  const handleEnded = () => {
    if (idx < urls.length - 1) {
      autoAdvance.current = true; // kullanıcı çalıyordu; sıradaki parçaya geç
      setIdx(idx + 1);
    } else {
      setPlaying(false); // hepsi bitti
    }
  };

  const toggle = () => {
    const el = audioRef.current;
    if (!el) return;
    if (playing) {
      el.pause();
      setPlaying(false);
      return;
    }
    // Sıra sonundayken ve bitmişse baştan başlat
    if (idx === urls.length - 1 && el.ended) {
      if (urls.length > 1) {
        autoAdvance.current = true;
        setIdx(0);
        return;
      }
      el.currentTime = 0;
    }
    el.play().then(() => setPlaying(true)).catch(() => setPlaying(false));
  };

  return (
    <div
      style={{
        display: 'flex', alignItems: 'center', gap: 8, marginTop: 12,
        padding: '6px 10px', borderRadius: 999, width: 'fit-content',
        background: '#f0fdf4', border: '1px solid #bbf7d0', color: '#16a34a'
      }}
    >
      <button
        type="button"
        onClick={toggle}
        title={playing ? 'Duraklat' : 'Oynat'}
        style={{
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          width: 26, height: 26, borderRadius: '50%', cursor: 'pointer',
          border: 'none', background: '#22c55e', color: '#fff'
        }}
      >
        {playing ? <Pause size={14} /> : <Play size={14} />}
      </button>
      <Volume2 size={14} />
      <span style={{ fontSize: 12, whiteSpace: 'nowrap' }}>
        Sesli cevap{urls.length > 1 ? ` · ${idx + 1}/${urls.length}` : ''}
      </span>
      <audio
        ref={audioRef}
        src={urls[idx]}
        onEnded={handleEnded}
        preload="auto"
        style={{ display: 'none' }}
      />
    </div>
  );
}

// Koşu özeti rozetleri: hangi şerit çalıştı, ne kadar sürdü, kaç token gitti.
// Ara sürecin tamamı sunucudaki log dosyasında; burada sadece rakamlar var.
function RunMeta({ run }) {
  if (!run) return null;

  const laneLabels = {
    direct: 'Doğrudan cevap',
    react: 'ReAct',
    plan_execute: 'Plan & Execute'
  };

  const items = [
    laneLabels[run.lane] || run.lane,
    `${run.seconds?.toFixed(1)} sn`,
    `${run.total_tokens?.toLocaleString('tr-TR')} token`,
    run.tool_calls > 0 ? `${run.tool_calls} araç` : null
  ].filter(Boolean);

  return (
    <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 10 }}>
      {items.map((item, idx) => (
        <span
          key={idx}
          style={{
            fontSize: 11,
            padding: '2px 8px',
            borderRadius: 999,
            background: '#f1f5f9',
            color: '#475569',
            border: '1px solid #e2e8f0'
          }}
        >
          {item}
        </span>
      ))}
    </div>
  );
}

export function ChatMessage({ message }) {
  const [copied, setCopied] = useState(false);

  if (message.role === 'user') {
    return (
      <div className="message-row user animate-fade-in">
        <div className="message-bubble-user">
          {message.text}
        </div>
      </div>
    );
  }

  const handleCopyCode = (codeText) => {
    navigator.clipboard.writeText(codeText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Ajanın ürettiği görseller (grafik vb.). Backend bunları /api/artifacts/... altında
  // servis ediyor; Vite proxy sayesinde göreli yol doğrudan çalışıyor.
  const artifactUrls = message.run?.artifact_urls || [];
  // Sesli çıktı modunda üretilen wav parçaları (varsa) — sırayla oynatılır.
  const audioUrls = message.run?.audio_urls || [];

  return (
    <div className="message-row ai animate-fade-in">
      <div className="message-avatar-ai" style={{ background: '#ffffff', border: '1px solid #e2e8f0' }}>
        <Logo size={20} />
      </div>

      <div className="message-content-ai">
        <div className="ai-header-name">ardilAgent</div>

        {/* Ajan çalışıyor: ara süreç gösterilmez, sadece durum */}
        {message.thinking && (
          <div className="ai-text" style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#64748b' }}>
            <style>
              {`@keyframes ardil-blink { 0%, 80%, 100% { opacity: .25 } 40% { opacity: 1 } }`}
            </style>
            <span>düşünüyor</span>
            {[0, 1, 2].map((i) => (
              <span
                key={i}
                style={{
                  width: 5,
                  height: 5,
                  borderRadius: '50%',
                  background: '#94a3b8',
                  animation: `ardil-blink 1.4s infinite`,
                  animationDelay: `${i * 0.2}s`
                }}
              />
            ))}
          </div>
        )}

        {/* Main AI Text */}
        {message.text && (
          <div className="ai-text" style={{ whiteSpace: 'pre-wrap' }}>
            {message.text}
          </div>
        )}

        {/* Sesli çıktı: cevap parçaları sırayla oynatılır */}
        {audioUrls.length > 0 && <AudioSequence urls={audioUrls} />}

        {/* Ajanın ürettiği görseller */}
        {artifactUrls.map((url) => (
          <a key={url} href={url} target="_blank" rel="noreferrer">
            <img
              src={url}
              alt="Ajanın ürettiği görsel"
              style={{
                display: 'block',
                maxWidth: '100%',
                marginTop: 12,
                borderRadius: 8,
                border: '1px solid #e2e8f0'
              }}
            />
          </a>
        ))}

        {/* Key Insights Callout Card */}
        {message.keyInsights && message.keyInsights.length > 0 && (
          <div className="insights-card">
            <div className="insights-header">
              <Activity size={16} />
              <span>Key Insights</span>
            </div>
            <ul className="insights-list">
              {message.keyInsights.map((insight, idx) => (
                <li key={idx}>{insight}</li>
              ))}
            </ul>
          </div>
        )}

        {/* Optional follow-up question text */}
        {message.followUpText && (
          <div className="ai-text">
            {message.followUpText}
          </div>
        )}

        {/* 3 Column Grid Cards */}
        {message.cards && message.cards.length > 0 && (
          <div className="roadmap-grid">
            {message.cards.map((card, idx) => {
              let CardIcon = LayoutGrid;
              if (card.iconType === 'marketing') CardIcon = Megaphone;
              if (card.iconType === 'pricing') CardIcon = CreditCard;

              return (
                <div key={idx} className="roadmap-card">
                  <div className="roadmap-icon-box">
                    <CardIcon size={18} />
                  </div>
                  <div className="roadmap-title">{card.title}</div>
                  <div className="roadmap-desc">{card.description}</div>
                </div>
              );
            })}
          </div>
        )}

        {/* Code Snippet Box */}
        {message.codeSnippet && (
          <div className="code-block-container">
            <div className="code-header">
              <span>yaml</span>
              <button
                className="copy-code-btn"
                onClick={() => handleCopyCode(message.codeSnippet)}
              >
                {copied ? <Check size={14} color="#22c55e" /> : <Copy size={14} />}
                <span>{copied ? 'Copied!' : ''}</span>
              </button>
            </div>
            <pre className="code-content">
              <code>{message.codeSnippet}</code>
            </pre>
          </div>
        )}

        <RunMeta run={message.run} />
      </div>
    </div>
  );
}
