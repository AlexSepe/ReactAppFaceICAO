import React, { useEffect, useState } from 'react';

function AnalysisMetrics({ analysisSummary, report, qualityScore, qualityGrade, overallPass }) {
  const [position, setPosition] = useState({ x: null, y: null });
  const [dragging, setDragging] = useState(false);
  const [startPoint, setStartPoint] = useState({ x: 0, y: 0 });
  const [startPosition, setStartPosition] = useState({ x: 0, y: 0 });

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
    setStartPosition({ x: position.x || window.innerWidth - 360, y: position.y || window.innerHeight - 320 });
  };

  const wrapperStyle = {
    position: 'fixed',
    top: position.y !== null ? `${position.y}px` : 'auto',
    left: position.x !== null ? `${position.x}px` : 'auto',
    right: position.x === null ? '2rem' : 'auto',
    bottom: position.y === null ? '2rem' : 'auto',
    width: '320px',
    backgroundColor: 'var(--bg-secondary, #f5f5f5)',
    border: '1px solid var(--border-color, #ddd)',
    borderRadius: '8px',
    padding: '1rem',
    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
    zIndex: 1000,
    maxHeight: '50vh',
    overflowY: 'auto',
    fontFamily: 'var(--font-family, sans-serif)',
    cursor: dragging ? 'grabbing' : 'grab'
  };

  return (
    <div style={wrapperStyle} onMouseDown={handleMouseDown}>
      <h3 style={{ marginTop: 0, marginBottom: '0.75rem', fontSize: '0.95rem', fontWeight: 600 }}>Analysis Metrics</h3>
      <ul style={{ margin: 0, padding: '0 0 0 1.25rem', fontSize: '0.85rem', lineHeight: 1.6, color: 'var(--text-color, #333)' }}>
        {analysisSummary.length > 0 ? analysisSummary.map((item, idx) => (
          <li key={idx} style={{ marginBottom: '0.5rem' }}>{item}</li>
        )) : <li style={{ color: '#666' }}>No metrics available yet.</li>}
      </ul>

      <div style={{ marginTop: '1rem', fontSize: '0.9rem' }}>
        <strong>Quality Score:</strong> {qualityScore !== null ? `${qualityScore}%` : 'N/A'}<br />
        <strong>Grade:</strong> {qualityGrade}
      </div>

      <div style={{ marginTop: '0.75rem', padding: '0.6rem', borderRadius: '4px', backgroundColor: overallPass ? '#f0f9f0' : '#fff5f5', border: `2px solid ${overallPass ? '#1a8c11' : '#c41e3a'}`, textAlign: 'center', fontWeight: 'bold', color: overallPass ? '#1a8c11' : '#c41e3a' }}>
        {overallPass ? '✅ ICAO compliant (pass)' : '❌ Not compliant (fail)'}
      </div>
    </div>
  );
}

export default AnalysisMetrics;
