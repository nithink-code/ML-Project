import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { BrainCircuit, MessageSquare, Mic, BarChart3, ArrowRight } from 'lucide-react';
import { toast } from '@/components/ui/sonner';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';

const LandingPage = ({ setUser }) => {
  const navigate = useNavigate();

  const handleLogin = () => {
    // Redirect to Google OAuth login endpoint
    window.location.href = `${BACKEND_URL}/api/auth/google/login`;
  };

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <div className="relative overflow-hidden">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
          <div className="text-center fade-in">
            <div className="flex justify-center mb-6">
              <div className="bg-gradient-to-r from-blue-500 to-cyan-500 p-4 rounded-2xl shadow-lg">
                <BrainCircuit size={64} className="text-white" />
              </div>
            </div>
            <h1 className="text-5xl sm:text-6xl lg:text-7xl font-bold text-gray-900 mb-6">
              AI Interview Coach
            </h1>
            <p className="text-lg sm:text-xl text-gray-700 mb-8 max-w-3xl mx-auto">
              Master your interview skills with our advanced AI-powered chatbot.
              Practice technical, behavioral, and general interviews with real-time feedback.
            </p>
            <Button
              data-testid="login-button"
              onClick={handleLogin}
              size="lg"
              className="bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 text-white px-8 py-6 text-lg rounded-full shadow-xl"
            >
              Get Started with Google
              <ArrowRight className="ml-2" size={20} />
            </Button>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <h2 className="text-3xl sm:text-4xl font-bold text-center text-gray-900 mb-16">
          Why Choose AI Interview Coach?
        </h2>
        <div className="grid md:grid-cols-3 gap-8">
          <div className="glass rounded-2xl p-8 hover:shadow-2xl transition-shadow">
            <div className="bg-blue-100 w-16 h-16 rounded-xl flex items-center justify-center mb-6">
              <MessageSquare className="text-blue-600" size={32} />
            </div>
            <h3 className="text-2xl font-semibold mb-4 text-gray-900">Text-Based Interviews</h3>
            <p className="text-gray-700">
              Engage in realistic interview conversations powered by GPT-5.
              Get intelligent follow-up questions tailored to your responses.
            </p>
          </div>

          <div className="glass rounded-2xl p-8 hover:shadow-2xl transition-shadow">
            <div className="bg-cyan-100 w-16 h-16 rounded-xl flex items-center justify-center mb-6">
              <Mic className="text-cyan-600" size={32} />
            </div>
            <h3 className="text-2xl font-semibold mb-4 text-gray-900">Voice-Enabled Practice</h3>
            <p className="text-gray-700">
              Practice speaking your answers aloud with voice recording.
              Build confidence in your verbal communication skills.
            </p>
          </div>

          <div className="glass rounded-2xl p-8 hover:shadow-2xl transition-shadow">
            <div className="bg-purple-100 w-16 h-16 rounded-xl flex items-center justify-center mb-6">
              <BarChart3 className="text-purple-600" size={32} />
            </div>
            <h3 className="text-2xl font-semibold mb-4 text-gray-900">AI Evaluation & Scoring</h3>
            <p className="text-gray-700">
              Receive detailed performance analysis with scores on communication,
              technical skills, and problem-solving abilities.
            </p>
          </div>
        </div>
      </div>

      {/* Interview Types Section */}
      <div className="bg-white py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-3xl sm:text-4xl font-bold text-center text-gray-900 mb-16">
            Practice Multiple Interview Types
          </h2>
          <div className="grid md:grid-cols-3 gap-8">
            <div className="border-2 border-blue-200 rounded-2xl p-8 hover:border-blue-400 transition-colors">
              <h3 className="text-2xl font-semibold mb-4 text-blue-700">Technical Interviews</h3>
              <p className="text-gray-700">
                Algorithms, data structures, system design, and coding challenges.
                Perfect for software engineering roles.
              </p>
            </div>

            <div className="border-2 border-cyan-200 rounded-2xl p-8 hover:border-cyan-400 transition-colors">
              <h3 className="text-2xl font-semibold mb-4 text-cyan-700">Behavioral Interviews</h3>
              <p className="text-gray-700">
                STAR method questions about teamwork, leadership, and conflict resolution.
                Essential for all job seekers.
              </p>
            </div>

            <div className="border-2 border-purple-200 rounded-2xl p-8 hover:border-purple-400 transition-colors">
              <h3 className="text-2xl font-semibold mb-4 text-purple-700">General Interviews</h3>
              <p className="text-gray-700">
                Background, experience, and career goals discussions.
                Build overall interview confidence.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* CTA Section */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 text-center">
        <h2 className="text-3xl sm:text-4xl font-bold text-gray-900 mb-6">
          Ready to Ace Your Next Interview?
        </h2>
        <p className="text-lg text-gray-700 mb-8 max-w-2xl mx-auto">
          Join thousands of professionals who have improved their interview skills
          with our AI-powered coaching platform.
        </p>
        <Button
          data-testid="cta-login-button"
          onClick={handleLogin}
          size="lg"
          className="bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 text-white px-8 py-6 text-lg rounded-full shadow-xl"
        >
          Start Practicing Now
          <ArrowRight className="ml-2" size={20} />
        </Button>
      </div>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <p className="text-gray-400">
            Powered by OpenAI GPT-5 | © 2025 AI Interview Coach
          </p>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage;