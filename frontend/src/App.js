import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import './App.css';
import LandingPage from './pages/LandingPage';
import Dashboard from './pages/Dashboard';
import InterviewPage from './pages/InterviewPage';
import EvaluationPage from './pages/EvaluationPage';
import AuthCallback from './pages/AuthCallback';
import { Toaster } from './components/ui/sonner';
import api, { BACKEND_URL } from './lib/axios';

function App() {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Try to fetch current user session
    const fetchUser = async () => {
      try {
        const response = await api.get('/api/auth/me');
        // Backend returns null if not authenticated, user data if authenticated
        setUser(response.data);
      } catch (error) {
        console.error('Error checking session:', error);
        setUser(null);
      } finally {
        setLoading(false);
      }
    };
    fetchUser();
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-cyan-50">
        <div className="text-lg text-gray-600">Loading...</div>
      </div>
    );
  }

  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<LandingPage setUser={setUser} />} />
          <Route path="/auth/callback" element={<AuthCallback setUser={setUser} />} />
          <Route path="/dashboard" element={<Dashboard user={user} setUser={setUser} />} />
          <Route path="/interview/:interviewId" element={<InterviewPage user={user} />} />
          <Route path="/evaluation/:interviewId" element={<EvaluationPage user={user} />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
        <Toaster />
      </div>
    </Router>
  );
}

export default App;
