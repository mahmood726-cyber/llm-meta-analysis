/**
 * Network Plot Component
 *
 * Displays network meta-analysis treatment network
 */

import React, { useMemo } from 'react';
import { Card, CardContent, Box, Typography, ToggleButtonGroup, ToggleButton } from '@mui/material';
import Plotly from 'react-plotly.js';

interface Comparison {
  study_id: string;
  treatment_a: string;
  treatment_b: string;
  effect?: number;
}

interface NetworkPlotProps {
  comparisons: Comparison[];
  treatments: string[];
  onNodeClick?: (treatment: string) => void;
}

function NetworkPlot({ comparisons, treatments, onNodeClick }: NetworkPlotProps) {
  const [layoutType, setLayoutType] = React.useState<'spring' | 'circular'>('spring');

  const { nodes, edges } = useMemo(() => {
    // Count comparisons between each treatment pair
    const pairCounts: Record<string, number> = {};
    comparisons.forEach((comp) => {
      const key = [comp.treatment_a, comp.treatment_b].sort().join('-');
      pairCounts[key] = (pairCounts[key] || 0) + 1;
    });

    // Calculate node positions
    let nodePositions: Record<string, { x: number; y: number }>;

    if (layoutType === 'circular') {
      // Circular layout
      const n = treatments.length;
      const radius = 10;
      nodePositions = {};
      treatments.forEach((t, i) => {
        const angle = (2 * Math.PI * i) / n;
        nodePositions[t] = {
          x: radius * Math.cos(angle),
          y: radius * Math.sin(angle),
        };
      });
    } else {
      // Spring layout (simplified force-directed)
      nodePositions = {};
      const n = treatments.length;
      treatments.forEach((t, i) => {
        const row = Math.floor(i / 3);
        const col = i % 3;
        nodePositions[t] = {
          x: col * 10 - 10,
          y: -row * 10 + 5,
        };
      });
    }

    // Create node trace
    const nodeX = treatments.map((t) => nodePositions[t].x);
    const nodeY = treatments.map((t) => nodePositions[t].y);
    const nodeSizes = treatments.map((t) => {
      // Size based on number of connections
      const count = comparisons.filter(
        (c) => c.treatment_a === t || c.treatment_b === t
      ).length;
      return 30 + count * 5;
    });

    // Create edge traces
    const edgeX: number[] = [];
    const edgeY: number[] = [];
    const edgeWidths: number[] = [];

    Object.entries(pairCounts).forEach(([key, count]) => {
      const [t1, t2] = key.split('-');
      edgeX.push(nodePositions[t1].x, nodePositions[t2].x, null);
      edgeY.push(nodePositions[t1].y, nodePositions[t2].y, null);
      edgeWidths.push(Math.min(count * 2, 10));
    });

    return {
      nodes: { x: nodeX, y: nodeY, sizes: nodeSizes, names: treatments },
      edges: { x: edgeX, y: edgeY, widths: edgeWidths },
    };
  }, [comparisons, treatments, layoutType]);

  const traces = [
    // Edges
    {
      x: edges.x,
      y: edges.y,
      mode: 'lines',
      line: {
        color: '#999',
        width: 2,
      },
      hoverinfo: 'none',
      showlegend: false,
    },
    // Nodes
    {
      x: nodes.x,
      y: nodes.y,
      mode: 'markers+text',
      marker: {
        size: nodes.sizes,
        color: '#1f77b4',
        line: { color: 'white', width: 2 },
      },
      text: nodes.names,
      textposition: 'middle center',
      textfont: { size: 12, color: 'white', family: 'Arial Black' },
      hovertemplate: '%{text}<br>Connections: %{marker.size}<extra></extra>',
      type: 'scatter',
    },
  ];

  const plotLayout = {
    title: {
      text: 'Treatment Network',
      font: { size: 18 },
    },
    xaxis: {
      showgrid: false,
      showticklabels: false,
      zeroline: false,
    },
    yaxis: {
      showgrid: false,
      showticklabels: false,
      zeroline: false,
      scaleanchor: 'x',
      scaleratio: 1,
    },
    hovermode: 'closest',
    margin: { l: 50, r: 50, t: 50, b: 50 },
    showlegend: false,
    height: 500,
  };

  return (
    <Card>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography variant="h6">Network Meta-Analysis</Typography>
          <ToggleButtonGroup
            size="small"
            value={layoutType}
            exclusive
            onChange={(e, newValue) => newValue && setLayoutType(newValue)}
          >
            <ToggleButton value="spring">Spring</ToggleButton>
            <ToggleButton value="circular">Circular</ToggleButton>
          </ToggleButtonGroup>
        </Box>

        <Typography variant="body2" color="text.secondary" gutterBottom>
          Node size indicates number of comparisons. Line thickness indicates number of studies.
        </Typography>

        <Plotly
          divId="network-plot"
          data={traces}
          layout={plotLayout}
          style={{ width: '100%' }}
          config={{
            responsive: true,
            displayModeBar: false,
          }}
          onClick={(data: any) => {
            const point = data.points[0];
            if (point.text && onNodeClick) {
              onNodeClick(point.text);
            }
          }}
        />
      </CardContent>
    </Card>
  );
}

export default NetworkPlot;
