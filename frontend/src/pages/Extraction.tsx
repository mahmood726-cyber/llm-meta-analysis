/**
 * Extraction Page
 *
 * Study data extraction interface
 */

import React, { useState } from 'react';
import {
  Box,
  Typography,
  Tabs,
  Tab,
  Alert,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  MenuItem,
  CircularProgress,
} from '@mui/material';
import { TabContext, TabPanel } from '@mui/lab';
import StudyUploader from '../components/StudyUploader';
import { api } from '../services/api';

interface ExtractionJob {
  id: string;
  study_id: string;
  status: string;
  config: any;
  results?: any;
}

function Extraction() {
  const [tabValue, setTabValue] = useState('upload');
  const [studies, setStudies] = useState<any[]>([]);
  const [selectedStudy, setSelectedStudy] = useState<string | null>(null);
  const [extractionDialog, setExtractionDialog] = useState(false);
  const [extractionConfig, setExtractionConfig] = useState({
    outcome_type: 'binary',
    model: 'gpt-4',
    use_rag: true,
    use_cot: false,
  });
  const [runningExtraction, setRunningExtraction] = useState(false);
  const [extractionResults, setExtractionResults] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const handleTabChange = (event: React.SyntheticEvent, newValue: string) => {
    setTabValue(newValue);
  };

  const handleUpload = async (files: File[]) => {
    try {
      const formData = new FormData();
      files.forEach((file) => formData.append('files', file));

      // This would be a real API call to upload studies
      // const response = await api.uploadStudies(formData);

      // Mock response for now
      return {
        success: files.map((f) => f.name),
        errors: [],
      };
    } catch (err: any) {
      console.error('Upload failed:', err);
      return {
        success: [],
        errors: files.map((f) => f.name),
      };
    }
  };

  const handleRunExtraction = async () => {
    if (!selectedStudy) return;

    try {
      setRunningExtraction(true);
      setError(null);

      const response = await api.runExtraction(selectedStudy, extractionConfig);

      setExtractionResults(response.data);
      setExtractionDialog(false);
      setTabValue('results');
    } catch (err: any) {
      console.error('Extraction failed:', err);
      setError(err.response?.data?.detail || 'Extraction failed');
    } finally {
      setRunningExtraction(false);
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Data Extraction
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        Upload studies and extract data using LLMs
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <TabContext value={tabValue}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
          <Tabs value={tabValue} onChange={handleTabChange}>
            <Tab label="Upload Studies" value="upload" />
            <Tab label="Study List" value="studies" />
            <Tab label="Results" value="results" />
          </Tabs>
        </Box>

        <TabPanel value="upload">
          <StudyUploader
            onUpload={handleUpload}
            accept=".pdf,.txt,.csv"
          />
        </TabPanel>

        <TabPanel value="studies">
          <Box>
            <Typography variant="h6" gutterBottom>
              Studies ({studies.length})
            </Typography>
            {studies.length === 0 ? (
              <Alert severity="info">
                No studies uploaded yet. Go to the Upload tab to add studies.
              </Alert>
            ) : (
              <Typography variant="body2" color="text.secondary">
                Study list would appear here with selection and extraction options.
              </Typography>
            )}
            <Button
              variant="contained"
              sx={{ mt: 2 }}
              onClick={() => setExtractionDialog(true)}
              disabled={!selectedStudy}
            >
              Run Extraction
            </Button>
          </Box>
        </TabPanel>

        <TabPanel value="results">
          <Typography variant="h6" gutterBottom>
            Extraction Results
          </Typography>
          {!extractionResults ? (
            <Alert severity="info">
              No extraction results yet. Run an extraction to see results here.
            </Alert>
          ) : (
            <pre>{JSON.stringify(extractionResults, null, 2)}</pre>
          )}
        </TabPanel>
      </TabContext>

      {/* Extraction Configuration Dialog */}
      <Dialog
        open={extractionDialog}
        onClose={() => setExtractionDialog(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Configure Extraction</DialogTitle>
        <DialogContent>
          <TextField
            select
            fullWidth
            label="Outcome Type"
            value={extractionConfig.outcome_type}
            onChange={(e) =>
              setExtractionConfig({ ...extractionConfig, outcome_type: e.target.value })
            }
            margin="normal"
          >
            <MenuItem value="binary">Binary (events/total)</MenuItem>
            <MenuItem value="continuous">Continuous (mean/sd)</MenuItem>
            <MenuItem value="survival">Survival (hazard ratio)</MenuItem>
          </TextField>

          <TextField
            select
            fullWidth
            label="LLM Model"
            value={extractionConfig.model}
            onChange={(e) =>
              setExtractionConfig({ ...extractionConfig, model: e.target.value })
            }
            margin="normal"
          >
            <MenuItem value="gpt-4">GPT-4</MenuItem>
            <MenuItem value="gpt-3.5-turbo">GPT-3.5 Turbo</MenuItem>
            <MenuItem value="claude-3">Claude 3</MenuItem>
            <MenuItem value="fine-tuned">Fine-tuned Model</MenuItem>
          </TextField>

          <TextField
            select
            fullWidth
            label="Advanced Features"
            value={extractionConfig.use_rag ? 'rag' : 'standard'}
            onChange={(e) =>
              setExtractionConfig({
                ...extractionConfig,
                use_rag: e.target.value === 'rag',
              })
            }
            margin="normal"
          >
            <MenuItem value="standard">Standard Prompting</MenuItem>
            <MenuItem value="rag">RAG-Enhanced</MenuItem>
            <MenuItem value="cot">Chain-of-Thought</MenuItem>
            <MenuItem value="ensemble">Ensemble (Multiple Models)</MenuItem>
          </TextField>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setExtractionDialog(false)}>Cancel</Button>
          <Button
            onClick={handleRunExtraction}
            variant="contained"
            disabled={runningExtraction}
            startIcon={runningExtraction ? <CircularProgress size={16} /> : null}
          >
            {runningExtraction ? 'Running...' : 'Run Extraction'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

export default Extraction;
