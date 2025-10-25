import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import api from '@/lib/axios';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { ArrowLeft, Star, TrendingUp, AlertCircle, CheckCircle2 } from 'lucide-react';
import { toast } from 'sonner';

const EvaluationPage = ({ user }) => {
  const { interviewId } = useParams();
  const navigate = useNavigate();
  const [evaluation, setEvaluation] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchEvaluation();
  }, [interviewId]);

  const fetchEvaluation = async () => {
    try {
      const response = await api.get(`/api/interviews/${interviewId}/evaluation`);
      setEvaluation(response.data);
    } catch (error) {
      console.error('Failed to fetch evaluation:', error);
      toast.error('Failed to load evaluation');
    } finally {
      setLoading(false);
    }
  };

  const getScoreColor = (score) => {
    if (score >= 8) return 'text-green-600';
    if (score >= 6) return 'text-blue-600';
    if (score >= 4) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getProgressColor = (score) => {
    if (score >= 8) return 'bg-green-500';
    if (score >= 6) return 'bg-blue-500';
    if (score >= 4) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  if (!evaluation) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <p className="text-gray-600 mb-4">Evaluation not found</p>
          <Button onClick={() => navigate('/dashboard')}>Go to Dashboard</Button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="glass border-b sticky top-0 z-10">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <Button
              data-testid="back-to-dashboard-button"
              onClick={() => navigate('/dashboard')}
              variant="outline"
              size="sm"
            >
              <ArrowLeft size={16} className="mr-2" />
              Back to Dashboard
            </Button>
          </div>
          <Badge className="bg-gradient-to-r from-purple-600 to-pink-600 text-white">
            <Star size={16} className="mr-1" />
            Evaluation Complete
          </Badge>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Overall Score */}
        <Card className="glass mb-8" data-testid="overall-score-card">
          <CardHeader className="text-center">
            <CardTitle className="text-3xl">Overall Performance</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-col items-center">
              <div className="text-7xl font-bold mb-4" data-testid="overall-score">
                <span className={getScoreColor(evaluation.overall_score)}>
                  {evaluation.overall_score.toFixed(1)}
                </span>
                <span className="text-3xl text-gray-400">/10</span>
              </div>
              <Progress
                value={evaluation.overall_score * 10}
                className="w-full max-w-md h-4"
              />
            </div>
          </CardContent>
        </Card>

        {/* Score Breakdown */}
        <Card className="glass mb-8" data-testid="scores-breakdown-card">
          <CardHeader>
            <CardTitle className="text-2xl">Score Breakdown</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-6">
              <div data-testid="communication-score">
                <div className="flex justify-between mb-2">
                  <span className="font-medium">Communication Skills</span>
                  <span className={`font-bold ${getScoreColor(evaluation.communication_score)}`}>
                    {evaluation.communication_score.toFixed(1)}/10
                  </span>
                </div>
                <Progress
                  value={evaluation.communication_score * 10}
                  className="h-3"
                />
              </div>

              <div data-testid="technical-score">
                <div className="flex justify-between mb-2">
                  <span className="font-medium">Technical Knowledge</span>
                  <span className={`font-bold ${getScoreColor(evaluation.technical_score)}`}>
                    {evaluation.technical_score.toFixed(1)}/10
                  </span>
                </div>
                <Progress
                  value={evaluation.technical_score * 10}
                  className="h-3"
                />
              </div>

              <div data-testid="problem-solving-score">
                <div className="flex justify-between mb-2">
                  <span className="font-medium">Problem Solving</span>
                  <span className={`font-bold ${getScoreColor(evaluation.problem_solving_score)}`}>
                    {evaluation.problem_solving_score.toFixed(1)}/10
                  </span>
                </div>
                <Progress
                  value={evaluation.problem_solving_score * 10}
                  className="h-3"
                />
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Strengths */}
        <Card className="glass mb-8" data-testid="strengths-card">
          <CardHeader>
            <CardTitle className="text-2xl flex items-center">
              <CheckCircle2 className="mr-2 text-green-600" size={24} />
              Strengths
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-3">
              {evaluation.strengths.map((strength, index) => (
                <li
                  key={index}
                  data-testid={`strength-${index}`}
                  className="flex items-start space-x-3 p-3 glass rounded-lg"
                >
                  <div className="bg-green-100 rounded-full p-1 mt-1">
                    <CheckCircle2 size={16} className="text-green-600" />
                  </div>
                  <span className="text-gray-800">{strength}</span>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>

        {/* Areas for Improvement */}
        <Card className="glass mb-8" data-testid="improvements-card">
          <CardHeader>
            <CardTitle className="text-2xl flex items-center">
              <TrendingUp className="mr-2 text-blue-600" size={24} />
              Areas for Improvement
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-3">
              {evaluation.areas_for_improvement.map((area, index) => (
                <li
                  key={index}
                  data-testid={`improvement-${index}`}
                  className="flex items-start space-x-3 p-3 glass rounded-lg"
                >
                  <div className="bg-blue-100 rounded-full p-1 mt-1">
                    <TrendingUp size={16} className="text-blue-600" />
                  </div>
                  <span className="text-gray-800">{area}</span>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>

        {/* Detailed Feedback */}
        <Card className="glass" data-testid="detailed-feedback-card">
          <CardHeader>
            <CardTitle className="text-2xl flex items-center">
              <AlertCircle className="mr-2 text-purple-600" size={24} />
              Detailed Feedback
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-gray-800 leading-relaxed whitespace-pre-wrap" data-testid="detailed-feedback">
              {evaluation.detailed_feedback}
            </p>
          </CardContent>
        </Card>

        {/* Actions */}
        <div className="flex justify-center mt-8 space-x-4">
          <Button
            data-testid="view-interview-button"
            onClick={() => navigate(`/interview/${id}`)}
            variant="outline"
            className="px-6 py-3"
          >
            View Interview
          </Button>
          <Button
            data-testid="new-interview-button"
            onClick={() => navigate('/dashboard')}
            className="bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 text-white px-6 py-3"
          >
            Start New Interview
          </Button>
        </div>
      </main>
    </div>
  );
};

export default EvaluationPage;