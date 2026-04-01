/**
 * Reports Page
 *
 * PRISMA diagrams, export options, and report generation
 */

import React, { useState } from 'react';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Alert,
  Tab,
  Tabs,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
} from '@mui/material';
import { Download, PictureAsPdf, TableChart, Description } from '@mui/icons-material';

interface PrismaData {
  identified: number;
  duplicates_removed: number;
  records_screened: number;
  records_excluded: number;
  full_text_assessed: number;
  full_text_excluded: number;
  studies_included: number;
}

interface RiskOfBiasEntry {
  study_id: string;
  study_name: string;
  domain_1: string; // Randomization
  domain_2: string; // Deviation from intervention
  domain_3: string; // Missing outcome data
  domain_4: string; // Outcome measurement
  domain_5: string; // Selection of reported result
  overall: string;
}

function Reports() {
  const [tabValue, setTabValue] = useState('prisma');
  const [exportFormat, setExportFormat] = useState('pdf');
  const [exporting, setExporting] = useState(false);

  // Mock PRISMA data
  const mockPrismaData: PrismaData = {
    identified: 1234,
    duplicates_removed: 56,
    records_screened: 1178,
    records_excluded: 1098,
    full_text_assessed: 80,
    full_text_excluded: 45,
    studies_included: 35,
  };

  // Mock Risk of Bias data
  const mockRobData: RiskOfBiasEntry[] = [
    {
      study_id: '1',
      study_name: 'Smith et al. (2023)',
      domain_1: 'Low',
      domain_2: 'Low',
      domain_3: 'Low',
      domain_4: 'Low',
      domain_5: 'Low',
      overall: 'Low',
    },
    {
      study_id: '2',
      study_name: 'Johnson et al. (2022)',
      domain_1: 'Some concerns',
      domain_2: 'Low',
      domain_3: 'Low',
      domain_4: 'Low',
      domain_5: 'Low',
      overall: 'Some concerns',
    },
    {
      study_id: '3',
      study_name: 'Williams et al. (2023)',
      domain_1: 'Low',
      domain_2: 'Low',
      domain_3: 'Some concerns',
      domain_4: 'Low',
      domain_5: 'Low',
      overall: 'Some concerns',
    },
  ];

  const handleExport = async (type: string) => {
    try {
      setExporting(true);
      const response = await api.exportResults(type as 'csv' | 'json' | 'pdf');

      // Create download link
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `meta-analysis-results.${type}`);
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (error) {
      console.error('Export failed:', error);
    } finally {
      setExporting(false);
    }
  };

  const getRobColor = (judgment: string) => {
    switch (judgment.toLowerCase()) {
      case 'low':
        return '#4caf50';
      case 'some concerns':
        return '#ff9800';
      case 'high':
        return '#f44336';
      default:
        return '#9e9e9e';
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Reports
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        Generate PRISMA diagrams, risk of bias assessments, and export results
      </Typography>

      <Grid container spacing={3}>
        {/* Export Options */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Export Results
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                Download analysis results in various formats
              </Typography>

              <Grid container spacing={2} direction="column">
                <Grid item>
                  <Button
                    fullWidth
                    variant="outlined"
                    startIcon={<PictureAsPdf />}
                    onClick={() => handleExport('pdf')}
                    disabled={exporting}
                  >
                    Export as PDF
                  </Button>
                </Grid>
                <Grid item>
                  <Button
                    fullWidth
                    variant="outlined"
                    startIcon={<TableChart />}
                    onClick={() => handleExport('csv')}
                    disabled={exporting}
                  >
                    Export as CSV
                  </Button>
                </Grid>
                <Grid item>
                  <Button
                    fullWidth
                    variant="outlined"
                    startIcon={<Description />}
                    onClick={() => handleExport('json')}
                    disabled={exporting}
                  >
                    Export as JSON
                  </Button>
                </Grid>
              </Grid>
            </CardContent>
          </Card>

          <Card sx={{ mt: 2 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                PRISMA 2020
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                Generate PRISMA flow diagram for publication
              </Typography>
              <Button
                fullWidth
                variant="contained"
                startIcon={<Download />}
                onClick={() => handleExport('pdf')}
                disabled={exporting}
              >
                Download PRISMA Diagram
              </Button>
            </CardContent>
          </Card>
        </Grid>

        {/* Main Content */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Tabs
                value={tabValue}
                onChange={(e, newValue) => setTabValue(newValue)}
                sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}
              >
                <Tab label="PRISMA Flow" value="prisma" />
                <Tab label="Risk of Bias" value="rob" />
                <Tab label="Summary" value="summary" />
              </Tabs>

              {tabValue === 'prisma' && (
                <Box>
                  <Typography variant="h6" gutterBottom>
                    PRISMA 2020 Flow Diagram
                  </Typography>
                  <Alert severity="info" sx={{ mb: 2 }}>
                    This diagram will be auto-generated based on your screening
                    workflow
                  </Alert>

                  {/* Simplified PRISMA diagram visualization */}
                  <Box
                    sx={{
                      border: 2,
                      borderColor: 'primary.main',
                      borderRadius: 2,
                      p: 3,
                      bgcolor: 'background.paper',
                    }}
                  >
                    <Grid container spacing={2} justifyContent="center">
                      <Grid item xs={12}>
                        <Typography variant="body1" align="center" fontWeight="bold">
                          Identification
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body2" align="center">
                          Records identified from databases (n = {mockPrismaData.identified})
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body2" align="center">
                          Records after duplicates removed (n ={' '}
                          {mockPrismaData.identified - mockPrismaData.duplicates_removed})
                        </Typography>
                      </Grid>
                      <Grid item xs={12}>
                        <Typography variant="body1" align="center" fontWeight="bold" sx={{ mt: 2 }}>
                          Screening
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body2" align="center">
                          Records screened (n = {mockPrismaData.records_screened})
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body2" align="center">
                          Records excluded (n = {mockPrismaData.records_excluded})
                        </Typography>
                      </Grid>
                      <Grid item xs={12}>
                        <Typography variant="body1" align="center" fontWeight="bold" sx={{ mt: 2 }}>
                          Included
                        </Typography>
                      </Grid>
                      <Grid item xs={12}>
                        <Typography variant="body2" align="center">
                          Studies included in review (n = {mockPrismaData.studies_included})
                        </Typography>
                      </Grid>
                    </Grid>
                  </Box>
                </Box>
              )}

              {tabValue === 'rob' && (
                <Box>
                  <Typography variant="h6" gutterBottom>
                    Risk of Bias Assessment (RoB 2)
                  </Typography>
                  <Typography variant="body2" color="text.secondary" paragraph>
                    Assessment of risk of bias for randomized trials
                  </Typography>

                  <TableContainer component={Paper} variant="outlined">
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>Study</TableCell>
                          <TableCell>D1</TableCell>
                          <TableCell>D2</TableCell>
                          <TableCell>D3</TableCell>
                          <TableCell>D4</TableCell>
                          <TableCell>D5</TableCell>
                          <TableCell>Overall</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {mockRobData.map((row) => (
                          <TableRow key={row.study_id}>
                            <TableCell>{row.study_name}</TableCell>
                            {[
                              row.domain_1,
                              row.domain_2,
                              row.domain_3,
                              row.domain_4,
                              row.domain_5,
                              row.overall,
                            ].map((judgment, idx) => (
                              <TableCell key={idx}>
                                <Box
                                  sx={{
                                    bgcolor: getRobColor(judgment),
                                    color: 'white',
                                    px: 1,
                                    py: 0.5,
                                    borderRadius: 1,
                                    textAlign: 'center',
                                    fontSize: '0.75rem',
                                    fontWeight: 'bold',
                                  }}
                                >
                                  {judgment}
                                </Box>
                              </TableCell>
                            ))}
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>

                  <Typography variant="caption" color="text.secondary" sx={{ mt: 2, display: 'block' }}>
                    D1: Randomization | D2: Deviations | D3: Missing data | D4: Outcome
                    measurement | D5: Selection of reported result
                  </Typography>
                </Box>
              )}

              {tabValue === 'summary' && (
                <Box>
                  <Typography variant="h6" gutterBottom>
                    Analysis Summary
                  </Typography>
                  <Alert severity="info">
                    Run an analysis to see summary statistics and recommendations for
                    reporting
                  </Alert>

                  <Typography variant="body2" sx={{ mt: 2 }}>
                    Based on your analysis, ensure your report includes:
                  </Typography>
                  <Box component="ul" sx={{ pl: 2, mt: 1 }}>
                    <Typography component="li" variant="body2">
                      Forest plot with effect sizes and confidence intervals
                    </Typography>
                    <Typography component="li" variant="body2">
                      Heterogeneity statistics (I², τ², Q-test)
                    </Typography>
                    <Typography component="li" variant="body2">
                      Risk of bias assessment table
                    </Typography>
                    <Typography component="li" variant="body2">
                      PRISMA flow diagram
                    </Typography>
                    <Typography component="li" variant="body2">
                      Sensitivity analysis results (if applicable)
                    </Typography>
                  </Box>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}

export default Reports;
