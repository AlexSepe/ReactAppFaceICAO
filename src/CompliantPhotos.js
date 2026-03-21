import React, { useState, useEffect, useReducer } from 'react';
import { removeImageBackground } from './lib/backgroundRemoval';

function CompliantPhotos({ compliantSnapshots }) {
  const [position, setPosition] = useState({ x: null, y: null });
  const [dragging, setDragging] = useState(false);
  const [startPoint, setStartPoint] = useState({ x: 0, y: 0 });
  const [startPosition, setStartPosition] = useState({ x: 0, y: 0 });
  const [selectedPhoto, setSelectedPhoto] = useState(null);
  const [viewModalOpen, setViewModalOpen] = useState(false);
  const [processingIndex, setProcessingIndex] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, dispatchProgress] = useReducer((state, action) => action, 0);
  const [taskName, dispatchTaskName] = useReducer((state, action) => action, "initializing...");

  useEffect(() => {
    const handleMouseMove = (event) => {
      if (!dragging) return;
      const deltaX = event.clientX - startPoint.x;
      const deltaY = event.clientY - startPoint.y;
      setPosition({ x: startPosition.x + deltaX, y: startPosition.y + deltaY });
    };

    const handleMouseUp = () => {
      if (dragging) {
        setDragging(false);
      }
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [dragging, startPoint, startPosition]);

  const handleMouseDown = (event) => {
    setDragging(true);
    setStartPoint({ x: event.clientX, y: event.clientY });
    setStartPosition({ x: position.x || window.innerWidth - 420, y: position.y || 100 });
  };

  const downloadPhoto = (snapshot, index) => {
    const link = document.createElement('a');
    link.href = snapshot;
    link.download = `compliant-passport-photo-${index + 1}.jpg`;
    link.click();
  };

  const formatPhoto = async (snapshot, index) => {
    try {
      setProcessingIndex(index);
      setIsProcessing(true);
      dispatchProgress(0);
      console.log(`Processing photo ${index + 1} - removing background!...`);
      
      const processedImage = await removeImageBackground(snapshot, {
        progress: (key, current, total) => {
            const perc = (current / total) * 100;
            dispatchTaskName(`${key}`);
            dispatchProgress(perc);
            console.log(`Background removal progress for photo ${index + 1}: ${key} | ${current}/${total} - percentage: ${perc.toFixed(2)}%`);            
        },
        debug: true
      });
      
      // Show processed image in modal
      setSelectedPhoto(processedImage);
      setViewModalOpen(true);
      
      console.log('Background removed successfully!');
    } catch (error) {
      console.error('Background removal failed:', error);
      alert(`Error processing image: ${error.message}`);
    } finally {
      setProcessingIndex(null);
      setIsProcessing(false);
    }
  };

  const viewPhoto = (snapshot) => {
    setSelectedPhoto(snapshot);
    setViewModalOpen(true);
  };

  const wrapperStyle = {
    position: 'fixed',
    top: position.y !== null ? `${position.y}px` : '100px',
    left: position.x !== null ? `${position.x}px` : 'auto',
    right: position.x === null ? '2rem' : 'auto',
    width: '380px',
    backgroundColor: 'var(--bg-secondary, #f5f5f5)',
    border: '1px solid var(--border-color, #ddd)',
    borderRadius: '8px',
    padding: '1rem',
    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
    zIndex: 999,
    maxHeight: '70vh',
    overflowY: 'auto',
    fontFamily: 'var(--font-family, sans-serif)',
    cursor: dragging ? 'grabbing' : 'grab'
  };

  const photoItemStyle = {
    marginBottom: '1rem',
    padding: '0.75rem',
    backgroundColor: '#fff',
    border: '1px solid #ddd',
    borderRadius: '4px',
    display: 'flex',
    flexDirection: 'column',
    gap: '0.5rem'
  };

  const thumbnailStyle = {
    width: '100%',
    height: '150px',
    objectFit: 'cover',
    borderRadius: '4px',
    cursor: 'pointer',
    backgroundColor: '#f0f0f0'
  };

  const buttonGroupStyle = {
    display: 'flex',
    gap: '0.5rem',
    marginTop: '0.5rem'
  };

  const buttonStyle = {
    flex: 1,
    padding: '0.4rem 0.6rem',
    fontSize: '0.8rem',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    fontWeight: '500',
    transition: 'background-color 0.2s'
  };

  const downloadButtonStyle = {
    ...buttonStyle,
    backgroundColor: '#1a8c11',
    color: 'white'
  };

  const viewButtonStyle = {
    ...buttonStyle,
    backgroundColor: '#0066cc',
    color: 'white'
  };

  const formatButtonStyle = {
    ...buttonStyle,
    backgroundColor: '#ff9800',
    color: 'white'
  };

  if (!compliantSnapshots || compliantSnapshots.length === 0) {
    return null;
  }

  return (
    <>
      <div style={wrapperStyle} onMouseDown={handleMouseDown}>
        <h3 style={{ marginTop: 0, marginBottom: '1rem', fontSize: '1rem', fontWeight: '600', cursor: 'grab' }}>
          📸 Compliant Photos ({compliantSnapshots.length})
        </h3>

        <div style={{ maxHeight: 'calc(70vh - 100px)', overflowY: 'auto' }}>
          {compliantSnapshots.map((snapshot, index) => (
            <div key={index} style={photoItemStyle}>
              <img 
                src={snapshot} 
                alt={`Compliant photo ${index + 1}`} 
                style={thumbnailStyle}
                onClick={() => viewPhoto(snapshot)}
              />
              <div style={{ fontSize: '0.85rem', color: '#666' }}>
                Photo {index + 1}
              </div>
              <div style={buttonGroupStyle}>
                <button
                  onClick={() => downloadPhoto(snapshot, index)}
                  style={downloadButtonStyle}
                  onMouseOver={(e) => e.target.style.backgroundColor = '#158c0c'}
                  onMouseOut={(e) => e.target.style.backgroundColor = '#1a8c11'}
                  title="Download this photo"
                >
                  📥 Download
                </button>
                <button
                  onClick={() => viewPhoto(snapshot)}
                  style={viewButtonStyle}
                  onMouseOver={(e) => e.target.style.backgroundColor = '#0052a3'}
                  onMouseOut={(e) => e.target.style.backgroundColor = '#0066cc'}
                  title="View in full size"
                >
                  👁️ View
                </button>
                <button
                  onClick={() => formatPhoto(snapshot, index)}
                  disabled={processingIndex === index}
                  style={{
                    ...formatButtonStyle,
                    backgroundColor: processingIndex === index ? '#ccc' : '#ff9800',
                    opacity: processingIndex === index ? 0.6 : 1,
                    cursor: processingIndex === index ? 'not-allowed' : 'pointer'
                  }}
                  title="Remove background from photo"
                >
                  {processingIndex === index ? '⏳ Processing...' : '🎨 Remove BG'}
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Processing Popup */}
      {isProcessing && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            backgroundColor: 'rgba(0, 0, 0, 0.5)',
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            zIndex: 1500
          }}
        >
          <div
            style={{
              backgroundColor: 'white',
              padding: '20px',
              borderRadius: '8px',
              textAlign: 'center',
              boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
              minWidth: '300px'
            }}
          >
            <h3 style={{ marginTop: 0, marginBottom: '1rem' }}>🎨 Processing Background Removal</h3>
            <p style={{ marginBottom: '1rem' }}>Please wait while we remove the background...</p>
            <p style={{ marginBottom: '1rem' }}>{taskName}</p>            
            <div style={{ marginBottom: '1rem' }}>
              <div
                style={{
                  width: '200px',
                  height: '20px',
                  backgroundColor: '#f0f0f0',
                  borderRadius: '10px',
                  overflow: 'hidden',
                  margin: '0 auto'
                }}
              >
                <div
                  style={{
                    width: `${progress}%`,
                    height: '100%',
                    backgroundColor: '#4caf50'
                  }}
                ></div>
              </div>
              <p style={{ marginTop: '0.5rem', fontSize: '0.9rem', color: '#666' }}>
                {progress.toFixed(2)}% Complete
              </p>
            </div>
          </div>
        </div>
      )}

      {/* View Modal */}
      {viewModalOpen && selectedPhoto && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            backgroundColor: 'rgba(0, 0, 0, 0.7)',
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            zIndex: 2000
          }}
          onClick={() => setViewModalOpen(false)}
        >
          <div
            style={{
              position: 'relative',
              backgroundColor: 'white',
              borderRadius: '8px',
              padding: '1rem',
              maxWidth: '90vw',
              maxHeight: '90vh',
              overflow: 'auto'
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <button
              onClick={() => setViewModalOpen(false)}
              style={{
                position: 'absolute',
                top: '10px',
                right: '10px',
                backgroundColor: '#c41e3a',
                color: 'white',
                border: 'none',
                borderRadius: '50%',
                width: '32px',
                height: '32px',
                fontSize: '18px',
                cursor: 'pointer',
                fontWeight: 'bold'
              }}
            >
              ✕
            </button>
            <img 
              src={selectedPhoto} 
              alt="Full size compliant photo"
              style={{ maxWidth: '100%', maxHeight: '100%' }}
            />
          </div>
        </div>
      )}
    </>
  );
}

export default CompliantPhotos;
