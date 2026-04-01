/**
 * Dashboard Page
 *
 * Main overview with statistics and recent activity
 */

import React, { useEffect, useState } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  CircularProgress,
  Alert,
} from '@mui/material';
import {
  Description as StudiesIcon,
  Science as ExtractionsIcon,
  Analytics as AnalysesIcon,
  CheckCircle as CompletedIcon,
} from '@mui/icons-material';
import { api } from '../services/api';

interface DashboardStats {
  total_studies: number;
  total_extractions: number;
  total_analyses: number;
  pending_jobs: number;
}

interface RecentActivity {
  id: string;
  type: 'extraction' | 'analysis';
  status: string;
  created_at: string;
  study_name?: string;
  outcome_name?: string;
}

function Dashboard() {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [recentActivity, setRecentActivity] = useState<RecentActivity[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      setLoading(true);
      // In a real implementation, these would be separate API calls
      const [studiesRes, analysesRes] = await Promise.all([
        api.getStudies({ limit: 10 }),
        api.getAnalysis('recent'), // This endpoint would need to be added
      ]);

      setStats({
        total_studies: studiesRes.data.total || 0,
        total_extractions: 0, // Would come from a dedicated endpoint
        total_analyses: 0,
        pending_jobs: 0,
      });

      setRecentActivity([]);
      setError(null);
    } catch (err: any) {
      console.error('Failed to load dashboard:', err);
      setError(err.response?.data?.detail || 'Failed to load dashboard data');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="60vh">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Dashboard
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        Welcome to the LLM Meta-Analysis Framework
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {/* Stats Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center">
                <StudiesIcon color="primary" sx={{ fontSize: 40, mr: 2 }} />
                <Box>
                  <Typography variant="h4">{stats?.total_studies || 0}</Typography>
                  <Typography variant="body2" color="text.secondary">
                    Total Studies
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center">
                <ExtractionsIcon color="secondary" sx={{ fontSize: 40, mr: 2 }} />
                <Box>
                  <Typography variant="h4">{stats?.total_extractions || 0}</Typography>
                  <Typography variant="body2" color="text.secondary">
                    Extractions
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center">
                <AnalysesIcon sx={{ fontSize: 40, mr: 2, color: '#4caf50' }} />
                <Box>
                  <Typography variant="h4">{stats?.total_analyses || 0}</Typography>
                  <Typography variant="body2" color="text.secondary">
                    Analyses
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center">
                <CompletedIcon sx={{ fontSize: 40, mr: 2, color: '#ff9800' }} />
                <Box>
                  <Typography variant="h4">{stats?.pending_jobs || 0}</Typography>
                  <Typography variant="body2" color="text.secondary">
                    Pending Jobs
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Quick Actions */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Quick Start
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                Get started with your meta-analysis project
              </Typography>
              <Box component="ul" sx={{ pl: 2, mt: 0 }}>
                <Typography component="li" variant="body2" sx={{ mb: 1 }}>
                  <strong>Upload Studies:</strong> Go to Extraction to upload PDF or text
                  files
                </Typography>
                <Typography component="li" variant="body2" sx={{ mb: 1 }}>
                  <strong>Run Extraction:</strong> Use LLM to extract data automatically
                </Typography>
                <Typography component="li" variant="body2" sx={{ mb: 1 }}>
                  <strong>Configure Analysis:</strong> Set up meta-analysis parameters
                </Typography>
                <Typography component="li" variant="body2">
                  <strong>Generate Reports:</strong> Export PRISMA diagrams and plots
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={6}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Features
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                Advanced capabilities powered by LLMs
              </Typography>
              <Box component="ul" sx={{ pl: 2, mt: 0 }}>
                <Typography component="li" variant="body2" sx={{ mb: 1 }}>
                  RAG-enhanced data extraction
                </Typography>
                <Typography component="li" variant="body2" sx={{ mb: 1 }}>
                  Bayesian meta-analysis with MCMC
                </Typography>
                <Typography component="li" variant="body2" sx={{ mb: 1 }}>
                  Network meta-analysis visualization
                </Typography>
                <Typography component="li" variant="body2">
                  PRISMA workflow automation
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}

export default Dashboard;
