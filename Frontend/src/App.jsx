import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Sidebar } from './components/Sidebar';
import { ChatHeader } from './components/ChatHeader';
import { ChatMessage } from './components/ChatMessage';
import { ChatInput } from './components/ChatInput';
import { api } from './api';

export function App() {
  const [conversations, setConversations] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [messagesMap, setMessagesMap] = useState({});
  const [filterQuery, setFilterQuery] = useState('');
  // Ajan çalışırken ara süreç UI'a akmaz (bilinçli karar) — sadece "düşünüyor"
  // gösterilir. Sürecin tamamı sunucuda log dosyasına yazılıyor.
  const [thinking, setThinking] = useState(false);
  const [backendStatus, setBackendStatus] = useState('connecting');
  const [error, setError] = useState('');

  const chatBodyRef = useRef(null);

  const currentMessages = messagesMap[activeId] || [];

  const filteredMessages = currentMessages.filter((msg) => {
    if (!filterQuery) return true;
    return msg.text?.toLowerCase().includes(filterQuery.toLowerCase());
  });

  // Açılışta backend'e bağlan: oturumları getir, hiç yoksa bir tane aç.
  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        await api.health();
        const sessions = await api.listSessions();
        if (cancelled) return;

        setBackendStatus('online');

        if (sessions.length === 0) {
          const created = await api.createSession();
          if (cancelled) return;
          setConversations([created]);
          setActiveId(created.id);
          setMessagesMap({ [created.id]: [] });
        } else {
          setConversations(sessions);
          setActiveId(sessions[sessions.length - 1].id);
        }
      } catch (err) {
        if (!cancelled) {
          setBackendStatus('offline');
          setError(`Backend'e bağlanılamadı: ${err.message}. "python run_api.py" çalışıyor mu?`);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  // Seçili oturumun mesajları henüz çekilmediyse getir.
  useEffect(() => {
    if (!activeId || messagesMap[activeId]) return;

    let cancelled = false;
    api
      .getMessages(activeId)
      .then((messages) => {
        if (!cancelled) setMessagesMap((prev) => ({ ...prev, [activeId]: messages }));
      })
      .catch((err) => {
        if (!cancelled) setError(`Mesajlar alınamadı: ${err.message}`);
      });

    return () => {
      cancelled = true;
    };
  }, [activeId, messagesMap]);

  useEffect(() => {
    if (chatBodyRef.current) {
      chatBodyRef.current.scrollTop = chatBodyRef.current.scrollHeight;
    }
  }, [currentMessages, thinking, activeId]);

  const handleSendMessage = useCallback(
    async (userText, voice = false) => {
      if (!activeId || thinking) return;

      setError('');
      // Kullanıcı mesajını hemen göster; sunucudan dönmesini bekleme.
      const optimistic = {
        id: `local-${Date.now()}`,
        role: 'user',
        text: userText
      };
      setMessagesMap((prev) => ({
        ...prev,
        [activeId]: [...(prev[activeId] || []), optimistic]
      }));
      setThinking(true);

      try {
        const result = await api.sendMessage(activeId, userText, voice);

        // İyimser mesajı sunucunun döndürdüğü gerçek kayıtlarla değiştir.
        setMessagesMap((prev) => ({
          ...prev,
          [activeId]: [
            ...(prev[activeId] || []).filter((m) => m.id !== optimistic.id),
            result.user_message,
            result.ai_message
          ]
        }));

        // İlk mesajdan sonra başlık sunucuda oluşuyor; sidebar'ı tazele.
        api.listSessions().then(setConversations).catch(() => {});
      } catch (err) {
        setError(`Cevap alınamadı: ${err.message}`);
        setMessagesMap((prev) => ({
          ...prev,
          [activeId]: (prev[activeId] || []).filter((m) => m.id !== optimistic.id)
        }));
      } finally {
        setThinking(false);
      }
    },
    [activeId, thinking]
  );

  const handleNewChat = useCallback(async () => {
    try {
      const created = await api.createSession();
      setConversations((prev) => [created, ...prev]);
      setMessagesMap((prev) => ({ ...prev, [created.id]: [] }));
      setActiveId(created.id);
    } catch (err) {
      setError(`Yeni sohbet açılamadı: ${err.message}`);
    }
  }, []);

  // Son ajan mesajının koşu özeti footer'da gösterilir (şerit, süre, token).
  const lastRun = [...currentMessages].reverse().find((m) => m.run)?.run || null;

  return (
    <div className="app-layout">
      <Sidebar
        conversations={conversations}
        activeId={activeId}
        onSelectConversation={setActiveId}
        onNewChat={handleNewChat}
      />

      <main className="main-workspace">
        <ChatHeader
          agentName="ardilAgent"
          modelStatus={
            backendStatus === 'online'
              ? 'Online'
              : backendStatus === 'offline'
                ? 'Offline'
                : 'Bağlanıyor…'
          }
          onSearch={setFilterQuery}
        />

        <div className="chat-body" ref={chatBodyRef}>
          {error && (
            <div
              style={{
                margin: '12px 0',
                padding: '10px 14px',
                borderRadius: 8,
                background: '#fef2f2',
                border: '1px solid #fecaca',
                color: '#b91c1c',
                fontSize: 14
              }}
            >
              {error}
            </div>
          )}

          {filteredMessages.map((msg) => (
            <ChatMessage key={msg.id} message={msg} />
          ))}

          {thinking && <ChatMessage message={{ id: 'thinking', role: 'ai', thinking: true }} />}
        </div>

        <ChatInput onSendMessage={handleSendMessage} disabled={thinking} runSummary={lastRun} />
      </main>
    </div>
  );
}

export default App;
