import React, { useState, useEffect, useRef } from "react";
import { useParams, useNavigate } from "react-router-dom";
import api from "@/lib/axios";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { ArrowLeft, Send, Mic, MicOff, BarChart3, Loader2 } from "lucide-react";
import { toast } from '@/components/ui/sonner';

const InterviewPage = ({ user }) => {
  const { interviewId } = useParams();
  const navigate = useNavigate();
  const [interview, setInterview] = useState(null);
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState("");
  const [loading, setLoading] = useState(true);
  const [sending, setSending] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [evaluating, setEvaluating] = useState(false);
  const messagesEndRef = useRef(null);
  const recognitionRef = useRef(null);

  useEffect(() => {
    fetchInterview();
  }, [interviewId]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const fetchInterview = async () => {
    try {
      const response = await api.get(`/api/interviews/${interviewId}`);
      setInterview(response.data.interview);
      setMessages(response.data.messages);
    } catch (error) {
      console.error("Failed to fetch interview:", error);
      toast.error("Failed to load interview");
    } finally {
      setLoading(false);
    }
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || sending) return;

    // Check if interview is completed
    if (
      interview?.status === "completed" ||
      interview?.status === "evaluated"
    ) {
      toast.error(
        "This interview has been completed. You cannot send more messages."
      );
      return;
    }

    const userMessage = inputMessage;
    setInputMessage("");
    setSending(true);

    // Optimistically add user message to UI
    const tempUserMessage = {
      id: "temp-" + Date.now(),
      role: "user",
      content: userMessage,
      is_voice: false,
      timestamp: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, tempUserMessage]);

    try {
      const response = await api.post(
        `/api/interviews/${interviewId}/message`,
        { content: userMessage, is_voice: false }
      );

      // Replace temp message with actual messages from server
      setMessages((prev) => {
        const withoutTemp = prev.filter((m) => m.id !== tempUserMessage.id);
        return [
          ...withoutTemp,
          response.data.user_message,
          response.data.ai_message,
        ];
      });
    } catch (error) {
      console.error("Failed to send message:", error);
      // Remove temp message on error
      setMessages((prev) => prev.filter((m) => m.id !== tempUserMessage.id));

      // Show detailed error message
      if (error.response) {
        const errorMsg =
          error.response.data?.detail ||
          error.response.data?.message ||
          error.message;
        toast.error(`Failed to send message: ${errorMsg}`);
        console.error("Server error details:", error.response.data);
      } else if (error.request) {
        toast.error(
          "Cannot reach server. Please check if the backend is running on port 8000."
        );
      } else {
        toast.error("Failed to send message. Please try again.");
      }

      // Restore input message so user can try again
      setInputMessage(userMessage);
    } finally {
      setSending(false);
    }
  };

  const handleVoiceRecording = () => {
    if (
      !("webkitSpeechRecognition" in window) &&
      !("SpeechRecognition" in window)
    ) {
      toast.error("Voice recognition not supported in this browser");
      return;
    }

    // Check if interview is completed
    if (
      interview?.status === "completed" ||
      interview?.status === "evaluated"
    ) {
      toast.error(
        "This interview has been completed. You cannot send more messages."
      );
      return;
    }

    if (isRecording) {
      recognitionRef.current?.stop();
      setIsRecording(false);
      return;
    }

    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = "en-US";

    recognition.onstart = () => {
      setIsRecording(true);
      toast.info("Listening... Speak now");
    };

    recognition.onresult = async (event) => {
      const transcript = event.results[0][0].transcript;
      setInputMessage(transcript);

      // Optimistically add user message to UI
      const tempUserMessage = {
        id: "temp-" + Date.now(),
        role: "user",
        content: transcript,
        is_voice: true,
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, tempUserMessage]);

      // Auto-send voice message
      setSending(true);
      try {
        const response = await api.post(
          `/api/interviews/${interviewId}/message`,
          { content: transcript, is_voice: true }
        );

        // Replace temp message with actual messages from server
        setMessages((prev) => {
          const withoutTemp = prev.filter((m) => m.id !== tempUserMessage.id);
          return [
            ...withoutTemp,
            response.data.user_message,
            response.data.ai_message,
          ];
        });
        setInputMessage("");
      } catch (error) {
        console.error("Failed to send voice message:", error);
        // Remove temp message on error
        setMessages((prev) => prev.filter((m) => m.id !== tempUserMessage.id));

        if (error.response) {
          toast.error(
            `Failed to send message: ${
              error.response.data?.detail || error.message
            }`
          );
        } else if (error.request) {
          toast.error(
            "Cannot reach server. Please check if the backend is running."
          );
        } else {
          toast.error("Failed to send message. Please try again.");
        }
      } finally {
        setSending(false);
      }
    };

    recognition.onerror = (event) => {
      console.error("Speech recognition error:", event.error);
      setIsRecording(false);
      toast.error("Voice recognition error. Please try again.");
    };

    recognition.onend = () => {
      setIsRecording(false);
    };

    recognitionRef.current = recognition;
    recognition.start();
  };

  const handleCompleteInterview = async () => {
    if (evaluating) return;

    setEvaluating(true);

    try {
      // Complete the interview to stop the session
      await api.post(`/api/interviews/${interviewId}/complete`);

      // Update local interview state to prevent further messages
      setInterview((prev) => ({ ...prev, status: "completed" }));

      try {
        // Generate evaluation
        const evalResponse = await api.post(`/api/interviews/${interviewId}/evaluate`);
        
        if (evalResponse.data) {
          // Show completion message and redirect to evaluation page
          toast.success("Interview completed! Redirecting to evaluation results...");
          // Add a small delay to ensure the evaluation is saved
          setTimeout(() => {
            navigate(`/evaluation/${interviewId}`);
          }, 2000);
        }
      } catch (evalError) {
        console.error("Evaluation generation failed:", evalError);
        toast.success("Interview completed! Redirecting to evaluation page...");
        // Still redirect to evaluation page, it will handle the evaluation generation
        setTimeout(() => {
          navigate(`/evaluation/${interviewId}`);
        }, 2000);
      }
    } catch (error) {
      console.error("Failed to complete interview:", error);

      // Show detailed error
      if (error.response?.data?.detail) {
        toast.error(
          `Failed to complete interview: ${error.response.data.detail}`
        );
      } else if (
        error.message?.includes("Network Error") ||
        error.code === "ERR_NETWORK"
      ) {
        toast.error(
          "Cannot connect to server. Please ensure the backend is running on port 8000."
        );
      } else {
        toast.error("Failed to complete interview. Please try again.");
      }
    } finally {
      setEvaluating(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="glass border-b sticky top-0 z-10">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <Button
              data-testid="back-button"
              onClick={() => navigate("/dashboard")}
              variant="outline"
              size="sm"
            >
              <ArrowLeft size={16} className="mr-2" />
              Back
            </Button>
            <Badge className="capitalize">
              {interview?.interview_type} Interview
            </Badge>
          </div>
          <Button
            data-testid="complete-interview-button"
            onClick={handleCompleteInterview}
            disabled={evaluating || messages.length < 2}
            className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white"
          >
            {evaluating ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Completing...
              </>
            ) : (
              <>
                <BarChart3 size={16} className="mr-2" />
                Complete Interview
              </>
            )}
          </Button>
        </div>
      </header>

      {/* Messages */}
      <main className="flex-1 overflow-y-auto">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="space-y-6" data-testid="messages-container">
            {messages.map((message, index) => (
              <div
                key={message.id || index}
                data-testid={`message-${message.role}-${index}`}
                className={`flex ${
                  message.role === "user" ? "justify-end" : "justify-start"
                } fade-in`}
              >
                <div
                  className={`max-w-[80%] rounded-2xl px-6 py-4 ${
                    message.role === "user"
                      ? "bg-gradient-to-r from-blue-500 to-cyan-500 text-white"
                      : "glass text-gray-900"
                  }`}
                >
                  <div className="flex items-start space-x-2">
                    {message.is_voice && message.role === "user" && (
                      <Mic size={16} className="mt-1" />
                    )}
                    <p className="whitespace-pre-wrap">{message.content}</p>
                  </div>
                  <p
                    className={`text-xs mt-2 ${
                      message.role === "user"
                        ? "text-blue-100"
                        : "text-gray-500"
                    }`}
                  >
                    {new Date(message.timestamp).toLocaleTimeString()}
                  </p>
                </div>
              </div>
            ))}
            {sending && (
              <div className="flex justify-start fade-in">
                <div className="glass rounded-2xl px-6 py-4">
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"></div>
                    <div
                      className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"
                      style={{ animationDelay: "0.2s" }}
                    ></div>
                    <div
                      className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"
                      style={{ animationDelay: "0.4s" }}
                    ></div>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>
      </main>

      {/* Input */}
      <div className="glass border-t sticky bottom-0">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
                {interview?.status === "completed" ||
          interview?.status === "evaluated" ? (
            <div className="text-center py-3">
              <div className="space-y-2">
                <p className="text-gray-600 font-medium">
                  Interview has been completed.
                </p>
                <Button
                  onClick={() => navigate(`/evaluation/${interviewId}`)}
                  className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white"
                >
                  View Your Evaluation Results
                </Button>
              </div>
            </div>
          ) : (
            <div className="flex items-center space-x-4">
              <Button
                data-testid="voice-record-button"
                onClick={handleVoiceRecording}
                variant={isRecording ? "destructive" : "outline"}
                size="icon"
                className={isRecording ? "pulse-animation" : ""}
                disabled={sending}
              >
                {isRecording ? <MicOff size={20} /> : <Mic size={20} />}
              </Button>
              <Input
                data-testid="message-input"
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={(e) => e.key === "Enter" && handleSendMessage()}
                placeholder="Type your answer or use voice..."
                className="flex-1"
                disabled={sending || isRecording}
              />
              <Button
                data-testid="send-message-button"
                onClick={handleSendMessage}
                disabled={!inputMessage.trim() || sending}
                className="bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 text-white"
              >
                <Send size={20} />
              </Button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default InterviewPage;
