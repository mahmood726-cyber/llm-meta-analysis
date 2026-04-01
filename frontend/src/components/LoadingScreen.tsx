/**
 * Loading Screen Component
 *
 * Displays loading state during app initialization
 */

import React from 'react';
import { Box, CircularProgress, Typography, Container } from '@mui/material';

interface LoadingScreenProps {
  message?: string;
}

function LoadingScreen({ message = 'Loading...' }: LoadingScreenProps) {
  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100vh',
        bgcolor: 'background.default',
      }}
    >
      <CircularProgress size={60} thickness={4} />
      <Typography variant="h6" sx={{ mt: 4, color: 'text.secondary' }}>
        {message}
      </Typography>
    </Box>
  );
}

export default LoadingScreen;
