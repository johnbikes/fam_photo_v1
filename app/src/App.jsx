import React, { useState } from 'react';
import { Button, Container, Typography, Box, Grid } from '@mui/material';
import axios from 'axios';

function App() {
  const [files, setFiles] = useState([]);
  const [previews, setPreviews] = useState([]);

  const handleFileChange = (e) => {
    const selected = Array.from(e.target.files);
    setFiles(selected);
    setPreviews(selected.map(file => URL.createObjectURL(file)));
  };

  const handleUpload = async () => {
    const formData = new FormData();
    files.forEach((file) => formData.append('files', file));

    try {
      await axios.post('http://backend:8000/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      alert('Upload successful!');
    } catch (err) {
      alert('Upload failed.');
    }
  };

  return (
    <Container>
      <Typography variant="h4" gutterBottom>Photo Uploader</Typography>
      <input type="file" multiple accept="image/*" onChange={handleFileChange} />
      <Box my={2}>
        <Grid container spacing={2}>
          {previews.map((src, i) => (
            <Grid item key={i}><img src={src} alt='' width={100} /></Grid>
          ))}
        </Grid>
      </Box>
      <Button variant="contained" color="primary" onClick={handleUpload}>Upload</Button>
    </Container>
  );
}

export default App;