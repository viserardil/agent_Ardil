import React from 'react';

export function Logo({ size = 36, showText = false, textStyle = {} }) {
  return (
    <div style={{ display: 'inline-flex', alignItems: 'center', gap: 10 }}>
      <svg
        width={size}
        height={size}
        viewBox="0 0 100 100"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        style={{ flexShrink: 0 }}
      >
        {/* Outer Shield Outline */}
        <path
          d="M 50,8 
             C 72,16 86,22 88,32 
             C 88,64 72,82 50,94 
             C 28,82 12,64 12,32 
             C 14,22 28,16 50,8 Z"
          stroke="url(#shield-grad-dark)"
          strokeWidth="4.5"
          strokeLinejoin="round"
          fill="none"
        />
        
        {/* Shield Right Half Accent */}
        <path
          d="M 50,8 
             C 72,16 86,22 88,32 
             C 88,64 72,82 50,94 Z"
          stroke="#0f4c81"
          strokeWidth="4.5"
          strokeLinejoin="round"
          fill="none"
        />

        {/* Central Stylized 'A' Circuit Line */}
        <path
          d="M 50,18 L 30,72 L 40,72 L 50,44 L 60,72 L 70,72 Z"
          fill="none"
          stroke="#1e293b"
          strokeWidth="4"
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        <path
          d="M 50,18 L 68,70 M 36,54 L 64,54"
          stroke="#0f4c81"
          strokeWidth="4"
          strokeLinecap="round"
        />

        {/* Circuit Lines & Traces Inside Shield */}
        <path d="M 32,32 L 44,28" stroke="#334155" strokeWidth="2.5" strokeLinecap="round" />
        <circle cx="32" cy="32" r="3" fill="#334155" />

        <path d="M 68,30 L 58,35" stroke="#0f4c81" strokeWidth="2.5" strokeLinecap="round" />
        <circle cx="68" cy="30" r="3" fill="#0f4c81" />

        <path d="M 44,46 L 36,44" stroke="#334155" strokeWidth="2.5" strokeLinecap="round" />
        <circle cx="44" cy="46" r="3" fill="#334155" />

        <path d="M 58,46 L 64,44" stroke="#0f4c81" strokeWidth="2.5" strokeLinecap="round" />
        <circle cx="58" cy="46" r="3" fill="#0f4c81" />

        <path d="M 50,72 L 50,84" stroke="#0f4c81" strokeWidth="3" strokeLinecap="round" />
        <circle cx="50" cy="84" r="3.5" fill="#0f4c81" />

        {/* Gradients */}
        <defs>
          <linearGradient id="shield-grad-dark" x1="0" y1="0" x2="100" y2="100">
            <stop offset="0%" stopColor="#334155" />
            <stop offset="100%" stopColor="#0f4c81" />
          </linearGradient>
        </defs>
      </svg>

      {showText && (
        <span style={{ fontSize: '1.2rem', fontWeight: 700, letterSpacing: '-0.02em', color: '#1e293b', ...textStyle }}>
          ardil<span style={{ color: '#0f4c81', fontWeight: 600 }}>Agent</span>
        </span>
      )}
    </div>
  );
}
