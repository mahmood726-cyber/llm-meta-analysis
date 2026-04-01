/**
 * Analysis Page
 *
 * Meta-analysis configuration and results
 */

import React, { useState } from 'react';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  TextField,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  CircularProgress,
  Alert,
  ToggleButtonGroup,
  ToggleButton,
  Tabs,
  Tab,
} from '@mui/material';
import ForestPlot from '../components/ForestPlot';
import FunnelPlot from '../components/FunnelPlot';
import NetworkPlot from '../components/NetworkPlot';
import { api } from '../services/api';

interface AnalysisConfig {
  study_ids: string[];
  outcome_type: 'binary' | 'continuous';
  effect_measure: string;
  model_type: 'fixed' | 'random';
  ci_method: string;
}

interface AnalysisResult {
  id: string;
  pooled_effect: {
    effect: number;
    ci_lower: number;
    ci_upper: number;
    p_value: number;
    i_squared: number;
    tau_squared: number;
  };
  studies: any[];
}

function Analysis() {
  const [config, setConfig] = useState<AnalysisConfig>({
    study_ids: [],
    outcome_type: 'binary',
    effect_measure: 'OR',
    model_type: 'random',
    ci_method: 'hksj',
  });
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [plotTab, setPlotTab] = useState('forest');

  const handleRunAnalysis = async () => {
    try {
      setRunning(true);
      setError(null);

      const response = await api.runMetaAnalysis(config);
      setResult(response.data);
    } catch (err: any) {
      console.error('Analysis failed:', err);
      setError(err.response?.data?.detail || 'Analysis failed');
    } finally {
      setRunning(false);
    }
  };

  // Mock data for demonstration
  const mockForestData = [
    { study_id: '1', name: 'Study A', effect: 0.8, ci_lower: 0.6, ci_upper: 1.1, weight: 25 },
    { study_id: '2', name: 'Study B', effect: 0.7, ci_lower: 0.5, ci_upper: 0.9, weight: 30 },
    { study_id: '3', name: 'Study C', effect: 0.9, ci_lower: 0.7, ci_upper: 1.2, weight: 20 },
    { study_id: '4', name: 'Study D', effect: 0.65, ci_lower: 0.5, ci_upper: 0.85, weight: 25 },
  ];

  const mockPooledEffect = {
    effect: 0.75,
    ci_lower: 0.62,
    ci_upper: 0.9,
    p_value: 0.002,
    i_squared: 35.5,
  };

  const mockFunnelData = mockForestData;

  const mockNetworkComparisons = [
    { study_id: '1', treatment_a: 'Treatment A', treatment_b: 'Control' },
    { study_id: '2', treatment_a: 'Treatment B', treatment_b: 'Control' },
    { study_id: '3', treatment_a: 'Treatment A', treatment_b: 'Treatment B' },
  ];

  const mockTreatments = ['Control', 'Treatment A', 'Treatment B'];

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Meta-Analysis
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        Configure and run meta-analysis with advanced statistical methods
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Configuration Panel */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Configuration
              </Typography>

              <Grid container spacing={2} direction="column">
                <Grid item>
                  <FormControl fullWidth>
                    <InputLabel>Outcome Type</InputLabel>
                    <Select
                      value={config.outcome_type}
                      label="Outcome Type"
                      onChange={(e) =>
                        setConfig({
                          ...config,
                          outcome_type: e.target.value as 'binary' | 'continuous',
                        })
                      }
                    >
                      <MenuItem value="binary">Binary</MenuItem>
                      <MenuItem value="continuous">Continuous</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>

                <Grid item>
                  <FormControl fullWidth>
                    <InputLabel>Effect Measure</InputLabel>
                    <Select
                      value={config.effect_measure}
                      label="Effect Measure"
                      onChange={(e) =>
                        setConfig({ ...config, effect_measure: e.target.value })
                      }
                    >
                      <MenuItem value="OR">Odds Ratio</MenuItem>
                      <MenuItem value="RR">Risk Ratio</MenuItem>
                      <MenuItem value="RD">Risk Difference</MenuItem>
                      <MenuItem value="MD">Mean Difference</MenuItem>
                      <MenuItem value="SMD">Std. Mean Difference</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>

                <Grid item>
                  <FormControl fullWidth>
                    <InputLabel>Model Type</InputLabel>
                    <Select
                      value={config.model_type}
                      label="Model Type"
                      onChange={(e) =>
                        setConfig({
                          ...config,
                          model_type: e.target.value as 'fixed' | 'random',
                        })
                      }
                    >
                      <MenuItem value="fixed">Fixed Effect</MenuItem>
                      <MenuItem value="random">Random Effects</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>

                <Grid item>
                  <FormControl fullWidth>
                    <InputLabel>CI Method</InputLabel>
                    <Select
                      value={config.ci_method}
                      label="CI Method"
                      onChange={(e) =>
                        setConfig({ ...config, ci_method: e.target.value })
                      }
                    >
                      <MenuItem value="wald">Wald</MenuItem>
                      <MenuItem value="hksj">Hartung-Knapp-Sidik-Jonkman</MenuItem>
                      <MenuItem value="profile">Profile Likelihood</MenuItem>
                      <MenuItem value="bootstrap">Bootstrap</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>

                <Grid item>
                  <Button
                    variant="contained"
                    fullWidth
                    onClick={handleRunAnalysis}
                    disabled={running || config.study_ids.length === 0}
                    startIcon={running ? <CircularProgress size={16} /> : null}
                  >
                    {running ? 'Running...' : 'Run Analysis'}
                  </Button>
                </Grid>
              </Grid>
            </CardContent>
          </Card>

          {/* Results Summary */}
          {result && (
            <Card sx={{ mt: 2 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Results Summary
                </Typography>
                <Typography variant="body2">
                  <strong>Pooled Effect:</strong>{' '}
                  {result.pooled_effect.effect.toFixed(3)} [
                  {result.pooled_effect.ci_lower.toFixed(3)},{' '}
                  {result.pooled_effect.ci_upper.toFixed(3)}]
                </Typography>
                <Typography variant="body2">
                  <strong>p-value:</strong> {result.pooled_effect.p_value.toFixed(4)}
                </Typography>
                <Typography variant="body2">
                  <strong>I²:</strong> {result.pooled_effect.i_squared.toFixed(1)}%
                </Typography>
                <Typography variant="body2">
                  <strong>τ²:</strong> {result.pooled_effect.tau_squared.toFixed(4)}
                </Typography>
              </CardContent>
            </Card>
          )}
        </Grid>

        {/* Visualization Panel */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Box
                display="flex"
                justifyContent="space-between"
                alignItems="center"
                mb={2}
              >
                <Typography variant="h6">Plots</Typography>
                <ToggleButtonGroup
                  size="small"
                  value={plotTab}
                  exclusive
                  onChange={(e, newValue) => newValue && setPlotTab(newValue)}
                >
                  <ToggleButton value="forest">Forest</ToggleButton>
                  <ToggleButton value="funnel">Funnel</ToggleButton>
                  <ToggleButton value="network">Network</ToggleButton>
                </ToggleButtonGroup>
              </Box>

              {plotTab === 'forest' && (
                <ForestPlot
                  data={result?.studies || mockForestData}
                  pooledEffect={result?.pooled_effect || mockPooledEffect}
                  outcomeName="Primary Outcome"
                  effectMeasure={
                    config.effect_measure as 'OR' | 'RR' | 'MD' | 'SMD' | 'HR'
                  }
                />
              )}

              {plotTab === 'funnel' && (
                <FunnelPlot
                  data={mockFunnelData}
                  pooledEffect={mockPooledEffect.effect}
                  effectMeasure={
                    config.effect_measure as 'OR' | 'RR' | 'MD' | 'SMD'
                  }
                />
              )}

              {plotTab === 'network' && (
                <NetworkPlot
                  comparisons={mockNetworkComparisons}
                  treatments={mockTreatments}
                />
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}

export default Analysis;
