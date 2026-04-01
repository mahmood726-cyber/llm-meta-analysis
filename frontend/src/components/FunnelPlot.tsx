/**
 * Funnel Plot Component
 *
 * Displays publication bias assessment
 */

import React from 'react';
import Plotly from 'react-plotly.js';
import { Box, Card, CardContent, Typography } from '@mui/material';

interface FunnelData {
  study_id: string;
  name: string;
  effect: number;
  se: number;
  weight?: number;
}

interface FunnelPlotProps {
  data: FunnelData[];
  pooledEffect: number;
  effectMeasure?: 'OR' | 'RR' | 'MD' | 'SMD';
}

const EFFECT_LABELS = {
  OR: 'Odds Ratio (log scale)',
  RR: 'Risk Ratio (log scale)',
  MD: 'Mean Difference',
  SMD: 'Standardized Mean Difference',
};

function FunnelPlot({ data, pooledEffect, effectMeasure = 'OR' }: FunnelPlotProps) {
  const isLogScale = effectMeasure === 'OR' || effectMeasure === 'RR';

  // Prepare data points
  const effects = data.map((d) => d.effect);
  const ses = data.map((d) => d.se);
  const names = data.map((d) => d.name);

  // Create pseudo confidence limits for funnel shape
  const maxSe = Math.max(...ses);
  const seGrid = Array.from({ length: 100 }, (_, i) => (i / 99) * maxSe);
  const upper95 = pooledEffect + 1.96 * seGrid;
  const lower95 = pooledEffect - 1.96 * seGrid;
  const upper99 = pooledEffect + 2.58 * seGrid;
  const lower99 = pooledEffect - 2.58 * seGrid;

  const traces = [
    // 95% CI limit (upper)
    {
      x: upper95,
      y: seGrid,
      mode: 'lines',
      line: { color: 'lightgray', width: 1 },
      name: '95% CI',
      hoverinfo: 'skip',
    },
    // 95% CI limit (lower)
    {
      x: lower95,
      y: seGrid,
      mode: 'lines',
      line: { color: 'lightgray', width: 1 },
      hoverinfo: 'skip',
      showlegend: false,
    },
    // 99% CI limit (upper)
    {
      x: upper99,
      y: seGrid,
      mode: 'lines',
      line: { color: 'lightblue', width: 1, dash: 'dash' },
      name: '99% CI',
      hoverinfo: 'skip',
    },
    // 99% CI limit (lower)
    {
      x: lower99,
      y: seGrid,
      mode: 'lines',
      line: { color: 'lightblue', width: 1, dash: 'dash' },
      hoverinfo: 'skip',
      showlegend: false,
    },
    // Pooled effect line
    {
      x: [pooledEffect, pooledEffect],
      y: [0, maxSe],
      mode: 'lines',
      line: { color: 'black', width: 2, dash: 'dot' },
      name: 'Pooled effect',
      hoverinfo: 'skip',
    },
    // Study points
    {
      x: effects,
      y: ses,
      mode: 'markers',
      type: 'scatter',
      marker: { size: 10, color: '#1f77b4', opacity: 0.7 },
      text: names,
      hovertemplate: '%{text}<br>Effect: %{x:.3f}<br>SE: %{y:.3f}<extra></extra>',
      name: 'Studies',
    },
  ];

  const layout = {
    title: {
      text: `Funnel Plot (${effectMeasure})`,
      font: { size: 16 },
    },
    xaxis: {
      title: EFFECT_LABELS[effectMeasure],
      type: isLogScale ? 'log' : 'linear',
      gridcolor: '#e0e0e0',
      zeroline: true,
    },
    yaxis: {
      title: 'Standard Error',
      range: [0, maxSe * 1.1],
      gridcolor: '#e0e0e0',
    },
    hovermode: 'closest',
    margin: { l: 80, r: 50, t: 50, b: 60 },
    showlegend: true,
    height: 500,
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="body2" color="text.secondary" gutterBottom>
          Funnel plot for assessing publication bias. Points should be symmetrically distributed around the pooled effect.
        </Typography>
        <Plotly
          divId="funnel-plot"
          data={traces}
          layout={layout}
          style={{ width: '100%' }}
          config={{
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['lasso2d', 'select2d'],
          }}
        />
      </CardContent>
    </Card>
  );
}

export default FunnelPlot;
