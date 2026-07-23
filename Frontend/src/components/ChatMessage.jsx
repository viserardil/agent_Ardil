import React, { useState } from 'react';
import {
  Activity,
  LayoutGrid,
  Megaphone,
  CreditCard,
  Copy,
  Check
} from 'lucide-react';
import { Logo } from './Logo';

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
