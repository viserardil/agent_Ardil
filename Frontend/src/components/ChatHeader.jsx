import React, { useState } from 'react';
import { Search, Bell, User } from 'lucide-react';
import { Logo } from './Logo';

export function ChatHeader({ agentName, modelStatus, onSearch }) {
  const [searchTerm, setSearchTerm] = useState('');

  const handleSearchChange = (e) => {
    setSearchTerm(e.target.value);
    if (onSearch) onSearch(e.target.value);
  };

  return (
    <header className="chat-header">
      {/* Agent Status */}
      <div className="agent-profile">
        <div className="agent-avatar" style={{ background: '#ffffff', border: '1px solid #e2e8f0' }}>
          <Logo size={24} />
          <span className="online-dot" />
        </div>
        <div>
          <h2 className="agent-name">{agentName || 'ardilAgent'}</h2>
          <span className="agent-status">{modelStatus || 'Online'}</span>
        </div>
      </div>

      {/* Action Controls */}
      <div className="header-actions">
        {/* Search Bar */}
        <div className="search-bar">
          <Search size={16} color="#94a3b8" />
          <input 
            type="text" 
            className="search-input" 
            placeholder="Search in chat..."
            value={searchTerm}
            onChange={handleSearchChange}
          />
        </div>

        {/* Notification Bell */}
        <button className="icon-btn" title="Notifications" style={{ position: 'relative' }}>
          <Bell size={18} />
          <span 
            style={{
              position: 'absolute',
              top: 8,
              right: 8,
              width: 6,
              height: 6,
              backgroundColor: '#ef4444',
              borderRadius: '50%'
            }}
          />
        </button>

        {/* User Profile */}
        <button className="icon-btn" title="User Profile">
          <User size={18} />
        </button>
      </div>
    </header>
  );
}
