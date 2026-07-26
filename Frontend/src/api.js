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

// Ses kaydını metne çevirir (Groq Whisper). JSON değil multipart gönderilir:
// Content-Type başlığını ELLE KOYMA — tarayıcı, FormData için sınır (boundary)
// içeren doğru başlığı kendisi üretir. Ajanı çalıştırmaz, sadece metni döndürür.
async function transcribe(blob, filename = 'audio.webm', language) {
  const form = new FormData();
  form.append('file', blob, filename);
  if (language) form.append('language', language);

  const response = await fetch(BASE + '/transcribe', { method: 'POST', body: form });
  if (!response.ok) {
    let detail = '';
    try {
      detail = (await response.json()).detail || '';
    } catch {
      // gövde JSON değilse durum kodu yeterli
    }
    throw new Error(detail || `${response.status} ${response.statusText}`);
  }
  return response.json();
}

export const api = {
  health: () => request('/health'),
  listSessions: () => request('/sessions'),
  createSession: () => request('/sessions', { method: 'POST' }),
  getMessages: (sessionId) => request(`/sessions/${sessionId}/messages`),
  // voice=true iken backend cevabı Chatterbox ile seslendirip parça parça döner
  // (ai_message.run.audio_urls). Kapalıyken sesli çıktı hiç devreye girmez.
  sendMessage: (sessionId, text, voice = false) =>
    request(`/sessions/${sessionId}/messages`, {
      method: 'POST',
      body: JSON.stringify({ text, voice })
    }),
  transcribe
};
