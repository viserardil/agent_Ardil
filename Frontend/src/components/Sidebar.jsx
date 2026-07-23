import React from 'react';
import { 
  Plus, 
  History, 
  Bot, 
  BookOpen, 
  HelpCircle, 
  Settings
} from 'lucide-react';
import { Logo } from './Logo';

export function Sidebar({ conversations, activeId, onSelectConversation, onNewChat }) {
  return (
    <aside className="sidebar">
      {/* Brand Header */}
      <div className="sidebar-header" style={{ marginBottom: 20 }}>
        <Logo size={36} showText={true} />
      </div>

      {/* New Chat Button */}
      <button className="new-chat-btn" onClick={onNewChat}>
        <Plus size={18} />
        <span>New Chat</span>
      </button>

      {/* Recent Conversations */}
      <div className="sidebar-section-title">Recent Conversations</div>
      <nav className="nav-list">
        {conversations.map((item) => (
          <div 
            key={item.id}
            className={`nav-item ${item.id === activeId ? 'active' : ''}`}
            onClick={() => onSelectConversation(item.id)}
          >
            <History size={16} style={{ flexShrink: 0, opacity: 0.7 }} />
            <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {item.title}
            </span>
          </div>
        ))}
      </nav>

      {/* Workspace */}
      <div className="sidebar-section-title">Workspace</div>
      <nav className="nav-list">
        <a href="#agents" className="nav-item" onClick={(e) => e.preventDefault()}>
          <Bot size={16} style={{ flexShrink: 0, opacity: 0.7 }} />
          <span>Agent Profiles</span>
        </a>
        <a href="#library" className="nav-item" onClick={(e) => e.preventDefault()}>
          <BookOpen size={16} style={{ flexShrink: 0, opacity: 0.7 }} />
          <span>Library</span>
        </a>
      </nav>

      <div className="sidebar-spacer" />

      {/* Bottom Nav Links */}
      <nav className="nav-list">
        <a href="#help" className="nav-item" onClick={(e) => e.preventDefault()}>
          <HelpCircle size={16} style={{ flexShrink: 0, opacity: 0.7 }} />
          <span>Help Center</span>
        </a>
        <a href="#settings" className="nav-item" onClick={(e) => e.preventDefault()}>
          <Settings size={16} style={{ flexShrink: 0, opacity: 0.7 }} />
          <span>Settings</span>
        </a>
      </nav>
    </aside>
  );
}
