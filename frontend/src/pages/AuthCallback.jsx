import React, { useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { toast } from '@/components/ui/sonner';
import api from '@/lib/axios';

const AuthCallback = ({ setUser }) => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

  useEffect(() => {
    const handleCallback = async () => {
      const token = searchParams.get('token');
      const error = searchParams.get('error');

      if (error) {
        toast.error('Authentication failed. Please try again.');
        navigate('/');
        return;
      }

      if (token) {
        try {
          // Fetch user data with the token
          const response = await api.get('/api/auth/me');
          if (response.data) {
            setUser(response.data);
            toast.success('Login successful!');
            navigate('/dashboard');
          } else {
            throw new Error('Failed to get user data');
          }
        } catch (error) {
          console.error('Failed to fetch user:', error);
          toast.error('Authentication failed. Please try again.');
          navigate('/');
        }
      } else {
        toast.error('No authentication token received.');
        navigate('/');
      }
    };

    handleCallback();
  }, [searchParams, navigate, setUser]);

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-cyan-50">
      <div className="text-center">
        <div className="mb-4">
          <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
        </div>
        <p className="text-lg text-gray-600">Completing authentication...</p>
      </div>
    </div>
  );
};

export default AuthCallback;
