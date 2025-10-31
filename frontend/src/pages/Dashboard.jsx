import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import api from "@/lib/axios";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import {
  BrainCircuit,
  LogOut,
  Plus,
  Clock,
  CheckCircle,
  Star,
} from "lucide-react";
import { toast } from "sonner";

const Dashboard = ({ user, setUser }) => {
  const navigate = useNavigate();
  const [interviews, setInterviews] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showModal, setShowModal] = useState(false);

  useEffect(() => {
    // Redirect to login if no user
    if (!user && !loading) {
      navigate("/");
      return;
    }
    fetchInterviews();
  }, [user, loading]);

  const fetchInterviews = async () => {
    try {
      const response = await api.get(`/api/interviews`, {
        validateStatus: function (status) {
          return status < 500; // Don't throw for 401
        },
      });

      if (response.status === 401) {
        // Session expired, redirect to login
        navigate("/");
        toast.error("Session expired. Please login again.");
        return;
      }

      if (response.status === 200) {
        setInterviews(response.data);
      }
    } catch (error) {
      console.error("Failed to fetch interviews:", error);
      toast.error("Failed to load interviews");
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = async () => {
    try {
      await api.post(`/api/auth/logout`);
      setUser(null);
      navigate("/");
      toast.success("Logged out successfully");
    } catch (error) {
      console.error("Logout failed:", error);
    }
  };

  const handleCreateInterview = async (type) => {
    try {
      const response = await api.post(`/api/interviews`, {
        interview_type: type,
      });
      toast.success("Interview session started!");
      navigate(`/interview/${response.data.id}`);
    } catch (error) {
      console.error("Failed to create interview:", error);
      toast.error("Failed to start interview");
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case "active":
        return "bg-blue-500";
      case "completed":
        return "bg-green-500";
      case "evaluated":
        return "bg-purple-500";
      default:
        return "bg-gray-500";
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case "active":
        return <Clock size={16} />;
      case "completed":
        return <CheckCircle size={16} />;
      case "evaluated":
        return <Star size={16} />;
      default:
        return null;
    }
  };

  // Show loading while checking user
  if (!user) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-lg">Loading...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="glass border-b sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="bg-gradient-to-r from-blue-500 to-cyan-500 p-2 rounded-lg">
              <BrainCircuit size={32} className="text-white" />
            </div>
            <h1 className="text-2xl font-bold text-gray-900">
              AI Interview Coach
            </h1>
          </div>
          <div className="flex items-center space-x-4">
            <Avatar data-testid="user-avatar">
              <AvatarImage src={user.picture} alt={user.name} />
              <AvatarFallback>
                {user.name.charAt(0).toUpperCase()}
              </AvatarFallback>
            </Avatar>
            <span className="text-sm font-medium text-gray-700 hidden sm:inline">
              {user.name}
            </span>
            <Button
              data-testid="logout-button"
              onClick={handleLogout}
              variant="outline"
              size="sm"
            >
              <LogOut size={16} className="mr-2" />
              Logout
            </Button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Start New Interview Section */}
        <Card className="mb-12 glass" data-testid="new-interview-card">
          <CardHeader>
            <CardTitle className="text-2xl">Start a New Interview</CardTitle>
            <CardDescription>
              Choose an interview type to begin your practice session
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-3 gap-6">
              <button
                data-testid="start-technical-interview"
                onClick={() => handleCreateInterview("technical")}
                className="glass rounded-xl p-6 hover:shadow-xl transition-all border-2 border-blue-200 hover:border-blue-400"
              >
                <div className="text-left">
                  <div className="bg-blue-100 w-12 h-12 rounded-lg flex items-center justify-center mb-4">
                    <BrainCircuit className="text-blue-600" size={24} />
                  </div>
                  <h3 className="text-xl font-semibold mb-2 text-gray-900">
                    Technical
                  </h3>
                  <p className="text-sm text-gray-600">
                    Programming, algorithms, and system design questions
                  </p>
                </div>
              </button>

              <button
                data-testid="start-behavioral-interview"
                onClick={() => handleCreateInterview("behavioral")}
                className="glass rounded-xl p-6 hover:shadow-xl transition-all border-2 border-cyan-200 hover:border-cyan-400"
              >
                <div className="text-left">
                  <div className="bg-cyan-100 w-12 h-12 rounded-lg flex items-center justify-center mb-4">
                    <Plus className="text-cyan-600" size={24} />
                  </div>
                  <h3 className="text-xl font-semibold mb-2 text-gray-900">
                    Behavioral
                  </h3>
                  <p className="text-sm text-gray-600">
                    STAR method and soft skills evaluation
                  </p>
                </div>
              </button>

              <button
                data-testid="start-general-interview"
                onClick={() => handleCreateInterview("general")}
                className="glass rounded-xl p-6 hover:shadow-xl transition-all border-2 border-purple-200 hover:border-purple-400"
              >
                <div className="text-left">
                  <div className="bg-purple-100 w-12 h-12 rounded-lg flex items-center justify-center mb-4">
                    <CheckCircle className="text-purple-600" size={24} />
                  </div>
                  <h3 className="text-xl font-semibold mb-2 text-gray-900">
                    General
                  </h3>
                  <p className="text-sm text-gray-600">
                    Background and career discussion
                  </p>
                </div>
              </button>
            </div>
          </CardContent>
        </Card>

        {/* Interview History */}
        <Card className="glass" data-testid="interview-history-card">
          <CardHeader>
            <CardTitle className="text-2xl">Your Interview Sessions</CardTitle>
            <CardDescription>
              View and continue your past interviews
            </CardDescription>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="flex justify-center py-12">
                <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
              </div>
            ) : interviews.length === 0 ? (
              <div className="text-center py-12">
                <p className="text-gray-500">
                  No interviews yet. Start your first practice session above!
                </p>
              </div>
            ) : (
              <div className="space-y-4">
                {interviews.map((interview) => (
                  <div
                    key={interview.id}
                    data-testid={`interview-item-${interview.id}`}
                    className="glass rounded-lg p-4 hover:shadow-lg transition-shadow cursor-pointer"
                    onClick={() => {
                      if (interview.status === "evaluated") {
                        navigate(`/evaluation/${interview.id}`);
                      } else {
                        navigate(`/interview/${interview.id}`);
                      }
                    }}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-4">
                        <Badge
                          className={`${getStatusColor(
                            interview.status
                          )} text-white`}
                        >
                          <span className="flex items-center space-x-1">
                            {getStatusIcon(interview.status)}
                            <span className="ml-1">{interview.status}</span>
                          </span>
                        </Badge>
                        <div>
                          <h3 className="font-semibold text-gray-900 capitalize">
                            {interview.interview_type} Interview
                          </h3>
                          <p className="text-sm text-gray-600">
                            {new Date(interview.created_at).toLocaleDateString(
                              "en-US",
                              {
                                year: "numeric",
                                month: "long",
                                day: "numeric",
                                hour: "2-digit",
                                minute: "2-digit",
                              }
                            )}
                          </p>
                        </div>
                      </div>
                      <Button variant="outline" size="sm">
                        {interview.status === "evaluated"
                          ? "View Results"
                          : "Continue"}
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </main>
    </div>
  );
};

export default Dashboard;
