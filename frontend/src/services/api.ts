/**
 * API Client Service
 *
 * Handles all communication with the backend API
 */

import axios, { AxiosInstance, AxiosError } from 'axios';

// API configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Create axios instance
const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor
apiClient.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    if (error.response) {
      // Handle specific error statuses
      switch (error.response.status) {
        case 401:
          // Unauthorized - clear token and redirect to login
          localStorage.removeItem('auth_token');
          window.location.href = '/login';
          break;
        case 404:
          // Not found
          console.error('Resource not found');
          break;
        case 500:
          // Server error
          console.error('Server error:', error.response.data);
          break;
      }
    }
    return Promise.reject(error);
  }
);

// API methods
export const api = {
  // Health check
  getHealth: () => apiClient.get('/health'),

  // Studies
  getStudies: (params?: { page?: number; limit?: number; search?: string }) =>
    apiClient.get('/api/studies', { params }),

  getStudy: (id: string) => apiClient.get(`/api/studies/${id}`),

  createStudy: (data: any) => apiClient.post('/api/studies', data),

  updateStudy: (id: string, data: any) => apiClient.put(`/api/studies/${id}`, data),

  deleteStudy: (id: string) => apiClient.delete(`/api/studies/${id}`),

  // Extractions
  getExtractions: (studyId: string) =>
    apiClient.get(`/api/studies/${studyId}/extractions`),

  runExtraction: (studyId: string, config: any) =>
    apiClient.post(`/api/studies/${studyId}/extract`, config),

  // Analysis
  runAnalysis: (config: any) =>
    apiClient.post('/api/analysis/run', config),

  getAnalysis: (id: string) =>
    apiClient.get(`/api/analysis/${id}`),

  // Jobs
  getJob: (jobId: string) =>
    apiClient.get(`/api/jobs/${jobId}`),

  // Meta-analysis
  runMetaAnalysis: (config: {
    study_ids: string[];
    outcome_type: 'binary' | 'continuous';
    effect_measure: string;
    model_type: 'fixed' | 'random';
    ci_method: string;
  }) => apiClient.post('/api/meta-analysis/run', config),

  // Network meta-analysis
  runNetworkMA: (config: any) =>
    apiClient.post('/api/network-ma/run', config),

  // Export
  exportResults: (format: 'csv' | 'json' | 'pdf') =>
    apiClient.get(`/api/export/${format}`, { responseType: 'blob' }),

  // WebSocket connection for real-time updates
  connectWebSocket: (jobId: string): WebSocket => {
    const wsUrl = API_BASE_URL.replace('http', 'ws')
      .replace('https', 'ws');
    return new WebSocket(`${wsUrl}/ws/jobs/${jobId}`);
  },
};

export default apiClient;
