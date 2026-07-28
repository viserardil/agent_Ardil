/**
 * Backend istemcisi.
 *
 * Backend yalnızca NİHAİ CEVABI döndürür — triyaj kararı, adımlar, LLM girdi/çıktıları
 * gibi ara süreç buraya akmaz; o, sunucuda logs/sessions/<id>.log dosyasına anlık
 * yazılır. Burada dönen `run` alanı sadece özet rakamlardır (şerit, süre, token).
 */

const BASE = '/api';

async function request(path, options = {}) {
  const response = await fetch(BASE + path, {
    headers: { 'Content-Type': 'application/json' },
    ...options
  });

  if (!response.ok) {
    let detail = '';
    try {
      const body = await response.json();
      detail = body.detail || '';
    } catch {
      // gövde JSON değilse sessizce geç; durum kodu yeterli bilgi
    }
    throw new Error(detail || `${response.status} ${response.statusText}`);
  }

  return response.json();
}

// Not: canlı (streaming) STT WebSocket ile çalışır (/ws/stt), bkz. ChatInput.jsx —
// burada bir REST çağrısı yoktur.

export const api = {
  health: () => request('/health'),
  listSessions: () => request('/sessions'),
  createSession: () => request('/sessions', { method: 'POST' }),
  getMessages: (sessionId) => request(`/sessions/${sessionId}/messages`),
  // voice=true iken backend cevabı XTTS ile seslendirip parça parça döner
  // (ai_message.run.audio_urls). Kapalıyken sesli çıktı hiç devreye girmez.
  sendMessage: (sessionId, text, voice = false) =>
    request(`/sessions/${sessionId}/messages`, {
      method: 'POST',
      body: JSON.stringify({ text, voice })
    })
};
