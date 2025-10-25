import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';

// Suppress browser extension errors
window.addEventListener('unhandledrejection', event => {
  if (event.reason?.message?.includes('message channel closed')) {
    event.preventDefault();
    console.debug('Suppressed browser extension error:', event.reason.message);
  }
});

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
