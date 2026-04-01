/**
 * Study Uploader Component
 *
 * Drag-and-drop file upload for studies
 */

import React, { useState, useCallback } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  IconButton,
  LinearProgress,
  Alert,
} from '@mui/material';
import {
  CloudUpload as CloudUploadIcon,
  Description as FileIcon,
  Delete as DeleteIcon,
  CheckCircle as SuccessIcon,
  Error as ErrorIcon,
} from '@mui/icons-material';

interface FileWithStatus {
  file: File;
  status: 'pending' | 'uploading' | 'success' | 'error';
  progress?: number;
  error?: string;
}

interface StudyUploaderProps {
  onUpload?: (files: File[]) => Promise<{ success: string[]; errors: string[] }>;
  accept?: string;
  multiple?: boolean;
}

function StudyUploader({
  onUpload,
  accept = '.pdf,.txt,.csv,.json',
  multiple = true,
}: StudyUploaderProps) {
  const [files, setFiles] = useState<FileWithStatus[]>([]);
  const [isDragOver, setIsDragOver] = useState(false);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);

    const droppedFiles = Array.from(e.dataTransfer.files);
    addFiles(droppedFiles);
  }, []);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const selectedFiles = Array.from(e.target.files);
      addFiles(selectedFiles);
    }
  }, []);

  const addFiles = (newFiles: File[]) => {
    const fileWithStatus: FileWithStatus[] = newFiles.map((file) => ({
      file,
      status: 'pending',
    }));
    setFiles((prev) => [...prev, ...fileWithStatus]);
  };

  const removeFile = (index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const handleUpload = async () => {
    if (!onUpload) return;

    // Update status to uploading
    setFiles((prev) =>
      prev.map((f) => ({ ...f, status: 'uploading', progress: 0 }))
    );

    const filesToUpload = files.map((f) => f.file);

    try {
      // Simulate progress
      const progressInterval = setInterval(() => {
        setFiles((prev) =>
          prev.map((f) => ({
            ...f,
            progress: f.status === 'uploading' ? Math.min((f.progress || 0) + 10, 90) : f.progress,
          }))
        );
      }, 200);

      const result = await onUpload(filesToUpload);

      clearInterval(progressInterval);

      // Update status based on result
      setFiles((prev) =>
        prev.map((f) => {
          const fileName = f.file.name;
          if (result.success.includes(fileName)) {
            return { ...f, status: 'success', progress: 100 };
          } else if (result.errors.includes(fileName)) {
            return { ...f, status: 'error', error: 'Upload failed', progress: 100 };
          }
          return f;
        })
      );
    } catch (error) {
      setFiles((prev) =>
        prev.map((f) =>
          f.status === 'uploading'
            ? { ...f, status: 'error', error: 'Upload failed', progress: 100 }
            : f
        )
      );
    }
  };

  const getStatusIcon = (status: FileWithStatus['status']) => {
    switch (status) {
      case 'uploading':
        return null;
      case 'success':
        return <SuccessIcon color="success" />;
      case 'error':
        return <ErrorIcon color="error" />;
      default:
        return <FileIcon color="action" />;
    }
  };

  const pendingFiles = files.filter((f) => f.status === 'pending');
  const hasErrors = files.some((f) => f.status === 'error');

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Upload Studies
        </Typography>

        {/* Drop zone */}
        <Box
          sx={{
            border: 2,
            borderStyle: 'dashed',
            borderColor: isDragOver ? 'primary.main' : 'grey.300',
            borderRadius: 2,
            p: 4,
            textAlign: 'center',
            bgcolor: isDragOver ? 'action.hover' : 'transparent',
            cursor: 'pointer',
            transition: 'all 0.2s',
            mb: 2,
          }}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => document.getElementById('file-input')?.click()}
        >
          <CloudUploadIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 1 }} />
          <Typography variant="body1" gutterBottom>
            Drag and drop files here, or click to browse
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Accepted formats: PDF, TXT, CSV, JSON
          </Typography>
          <input
            id="file-input"
            type="file"
            accept={accept}
            multiple={multiple}
            style={{ display: 'none' }}
            onChange={handleFileSelect}
          />
        </Box>

        {/* File list */}
        {files.length > 0 && (
          <>
            <List dense>
              {files.map((fileWithStatus, index) => (
                <ListItem
                  key={index}
                  secondaryAction={
                    <IconButton edge="end" onClick={() => removeFile(index)}>
                      <DeleteIcon />
                    </IconButton>
                  }
                >
                  <ListItemIcon>{getStatusIcon(fileWithStatus.status)}</ListItemIcon>
                  <ListItemText
                    primary={fileWithStatus.file.name}
                    secondary={
                      fileWithStatus.status === 'uploading' ? (
                        <LinearProgress
                          variant="determinate"
                          value={fileWithStatus.progress || 0}
                        />
                      ) : fileWithStatus.status === 'error' ? (
                        <Typography variant="caption" color="error">
                          {fileWithStatus.error}
                        </Typography>
                      ) : (
                        <Typography variant="caption" color="text.secondary">
                          {(fileWithStatus.file.size / 1024).toFixed(1)} KB
                        </Typography>
                      )
                    }
                  />
                </ListItem>
              ))}
            </List>

            {hasErrors && (
              <Alert severity="error" sx={{ mb: 2 }}>
                Some files failed to upload. Please try again.
              </Alert>
            )}

            <Button
              variant="contained"
              fullWidth
              onClick={handleUpload}
              disabled={pendingFiles.length === 0}
            >
              Upload {pendingFiles.length} File{pendingFiles.length !== 1 ? 's' : ''}
            </Button>
          </>
        )}
      </CardContent>
    </Card>
  );
}

export default StudyUploader;
