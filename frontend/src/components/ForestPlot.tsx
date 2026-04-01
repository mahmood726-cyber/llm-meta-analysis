/**
 * Interactive Forest Plot Component
 *
 * Displays meta-analysis results with interactive features
 */

import React, { useMemo } from 'react';
import Plotly from 'react-plotly.js';
import { Box, Card, CardContent, Typography, ToggleButton, ToggleButtonGroup } from '@mui/material';

interface StudyData {
  study_id: string;
  name: string;
  effect: number;
  ci_lower: number;
  ci_upper: number;
  weight?: number;
  events?: number;
  total?: number;
}

interface ForestPlotProps {
  data: StudyData[];
  pooledEffect: {
    effect: number;
    ci_lower: number;
    ci_upper: number;
    p_value?: number;
    i_squared?: number;
  };
  outcomeName: string;
  effectMeasure?: 'OR' | 'RR' | 'MD' | 'SMD' | 'HR';
  showPooled?: boolean;
  onStudyClick?: (study: StudyData) => void;
}

const EFFECT_MEASURES = {
  OR: { label: 'Odds Ratio', scale: 'log' },
  RR: { label: 'Risk Ratio', scale: 'log' },
  MD: { label: 'Mean Difference', scale: 'linear' },
  SMD: { label: 'Std. Mean Diff', scale: 'linear' },
  HR: { label: 'Hazard Ratio', scale: 'log' },
};

function ForestPlot({
  data,
  pooledEffect,
  outcomeName,
  effectMeasure = 'OR',
  showPooled = true,
  onStudyClick,
}: ForestPlotProps) {
  const [sortBy, setSortBy] = React.useState<'name' | 'effect'>('name');

  // Sort and prepare data
  const sortedData = useMemo(() => {
    const sorted = [...data];
    if (sortBy === 'effect') {
      sorted.sort((a, b) => a.effect - b.effect);
    } else {
      sorted.sort((a, b) => a.name.localeCompare(b.name));
    }
    return sorted;
  }, [data, sortBy]);

  const effectInfo = EFFECT_MEASURES[effectMeasure];
  const isLogScale = effectInfo.scale === 'log';

  // Prepare plot traces
  const traces = [];

  // Study data
  const yPositions = sortedData.map((_, i) => i);

  sortedData.forEach((study, i) => {
    const y = i;
    const x = study.effect;

    // Confidence interval
    traces.push({
      x: [study.ci_lower, study.ci_upper],
      y: [y, y],
      mode: 'lines',
      line: { color: '#1f77b4', width: 3 },
      hoverinfo: 'skip',
      name: study.study_id,
      customdata: [study],
    });

    // Point estimate
    traces.push({
      x: [x],
      y: [y],
      mode: 'markers',
      marker: { size: 8, color: '#1f77b4' },
      hovertemplate: `%{customdata[0].name}<br>` +
                      `Effect: %{x:.3f}<br>` +
                      `CI: %{customdata[0].ci_lower:.3f} - %{customdata[0].ci_upper:.3f}<br>` +
                      `<extra></extra>`,
      customdata: [study],
    });
  });

  // Pooled effect
  if (showPooled && pooledEffect) {
    const pooledY = sortedData.length;

    // Pooled CI
    traces.push({
      x: [pooledEffect.ci_lower, pooledEffect.ci_upper],
      y: [pooledY, pooledY],
      mode: 'lines',
      line: { color: '#d62728', width: 4 },
      name: 'Pooled',
      hoverinfo: 'skip',
    });

    // Pooled point
    const pooledText = `Pooled: ${pooledEffect.effect.toFixed(3)} [${pooledEffect.ci_lower.toFixed(3)}, ${pooledEffect.ci_upper.toFixed(3)}]`;
    if (pooledEffect.p_value !== undefined) {
      pooledText += `, p=${pooledEffect.p_value.toFixed(3)}`;
    }
    if (pooledEffect.i_squared !== undefined) {
      pooledText += `, I²=${pooledEffect.i_squared.toFixed(1)}%`;
    }

    traces.push({
      x: [pooledEffect.effect],
      y: [pooledY],
      mode: 'markers',
      marker: { size: 12, color: '#d62728', symbol: 'diamond' },
      name: pooledText,
      hovertemplate: `<b>Pooled Effect</b><br>` +
                      `Effect: %{x:.3f}<br>` +
                      `CI: ${pooledEffect.ci_lower.toFixed(3)} - ${pooledEffect.ci_upper.toFixed(3)}<br>` +
                      `${pooledEffect.p_value !== undefined ? `p: ${pooledEffect.p_value.toFixed(3)}<br>` : ''}` +
                      `${pooledEffect.i_squared !== undefined ? `I²: ${pooledEffect.i_squared.toFixed(1)}%` : ''}` +
                      `<extra></extra>`,
    });
  }

  // Reference line at 1 (for log scale) or 0 (for linear)
  const refLineX = isLogScale ? [0.01, 100] : [-5, 5];
  const refLineY = [0, sortedData.length + (showPooled ? 1 : 0)];

  traces.push({
    x: isLogScale ? [1, 1] : [0, 0],
    y: refLineY,
    mode: 'lines',
    line: { color: 'black', width: 1, dash: 'dash' },
    name: 'No effect',
    hoverinfo: 'skip',
  });

  const layout = {
    title: {
      text: `Forest Plot: ${outcomeName}`,
      font: { size: 18 },
    },
    xaxis: {
      title: effectInfo.label,
      type: isLogScale ? 'log' : 'linear',
      range: isLogScale ? [0.1, 10] : undefined,
      gridcolor: '#e0e0e0',
      zeroline: true,
    },
    yaxis: {
      title: 'Study',
      tickmode: 'array',
      tickvals: yPositions,
      ticktext: sortedData.map((s) => s.name),
      gridcolor: '#e0e0e0',
    },
    hovermode: 'closest',
    margin: { l: 200, r: 50, t: 50, b: 50 },
    showlegend: false,
    height: 100 + sortedData.length * 30,
    dragmode: 'pan',
  };

  const handleRelayout = (eventData: any) => {
    if (eventData['xaxis.range']) {
      // Allow zooming
    }
  };

  return (
    <Card>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography variant="h6">{outcomeName}</Typography>
          <ToggleButtonGroup
            size="small"
            value={sortBy}
            exclusive
            onChange={(e, newValue) => newValue && setSortBy(newValue)}
          >
            <ToggleButton value="name">Sort by Name</ToggleButton>
            <ToggleButton value="effect">Sort by Effect</ToggleButton>
          </ToggleButtonGroup>
        </Box>

        <Plotly
          divId={`forest-plot-${outcomeName.replace(/\s+/g, '-')}`}
          data={traces}
          layout={layout}
          onRelayout={handleRelayout}
          style={{ width: '100%' }}
          config={{
            responsive: true,
            displayModeBar: false,
            doubleClick: 'reset+autosize',
          }}
          onClick={(data: any) => {
            const point = data.points[0];
            if (point.customdata && onStudyClick) {
              onStudyClick(point.customdata);
            }
          }}
        />

        {pooledEffect && (
          <Box mt={2} p={2} bgcolor="#f5f5f5" borderRadius={1}>
            <Typography variant="body2" component="div">
              <strong>Pooled Effect:</strong> {pooledEffect.effect.toFixed(3)}
              [{pooledEffect.ci_lower.toFixed(3)}, {pooledEffect.ci_upper.toFixed(3)}]
              {pooledEffect.p_value !== undefined && (
                <span>, p = {pooledEffect.p_value.toFixed(3)}</span>
              )}
              {pooledEffect.i_squared !== undefined && (
                <span>, I² = {pooledEffect.i_squared.toFixed(1)}%</span>
              )}
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );
}

export default ForestPlot;
