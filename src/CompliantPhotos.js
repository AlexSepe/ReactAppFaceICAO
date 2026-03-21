import React, { useState, useEffect } from 'react';

function CompliantPhotos({ compliantSnapshots }) {
  const [position, setPosition] = useState({ x: null, y: null });
  const [dragging, setDragging] = useState(false);
  const [startPoint, setStartPoint] = useState({ x: 0, y: 0 });
  const [startPosition, setStartPosition] = useState({ x: 0, y: 0 });
  const [selectedPhoto, setSelectedPhoto] = useState(null);
  const [viewModalOpen, setViewModalOpen] = useState(false);

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

  const formatPhoto = (snapshot, index) => {
    console.log(`Format option for photo ${index + 1}:`, snapshot);
    console.log('This will eventually apply formatting/processing to the photo');
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
                  style={formatButtonStyle}
                  onMouseOver={(e) => e.target.style.backgroundColor = '#e68900'}
                  onMouseOut={(e) => e.target.style.backgroundColor = '#ff9800'}
                  title="Format/Process photo"
                >
                  🎨 Format
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

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
