import React, { useEffect, useMemo, useRef, useState } from 'react';

import Webcam from 'react-webcam';
import * as faceapi from '@vladmandic/face-api';

const width = 500;
const height = 500;

const LOCAL_MODEL_URL = '/models';
const REMOTE_MODEL_URL = 'https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js@master/weights';

//{"faceDetected":true,"alignment":false,"faceSize":true,"resolution":true,"background":false,"lighting":true}
const initialReport = [
  { id: 'faceDetected', label: 'Face detected', passed: null, message: 'Waiting for detection' },
  { id: 'alignment', label: 'Head alignment (roll/yaw/pitch)', passed: null, message: 'Waiting for detection' },
  { id: 'faceSize', label: 'Face size ratio (>=40% && <=75%)', passed: null, message: 'Waiting for detection' },
  { id: 'resolution', label: 'Image resolution (>=800x600)', passed: null, message: 'Waiting for detection' },
  { id: 'background', label: 'Background uniformity (low variance)', passed: null, message: 'Waiting for detection' },
  { id: 'lighting', label: 'Lighting uniformity (even exposure)', passed: null, message: 'Waiting for detection' }
];

function getPointCenter(points) {
  const sum = points.reduce((acc, p) => ({ x: acc.x + p.x, y: acc.y + p.y }), { x: 0, y: 0 });
  return { x: sum.x / points.length, y: sum.y / points.length };
}

function analyzeImageData(img, detection) {
  const width = img.width;
  const height = img.height;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0, width, height);
  const raw = ctx.getImageData(0, 0, width, height);

  const faceBox = detection.detection.box;
  const faceHeightRatio = faceBox.height / height;
  const faceWidthRatio = faceBox.width / width;

  const leftEyeCenter = getPointCenter(detection.landmarks.getLeftEye());
  const rightEyeCenter = getPointCenter(detection.landmarks.getRightEye());
  const noseTip = detection.landmarks.getNose()[3] || detection.landmarks.getNose()[0];
  const mouthCenter = getPointCenter(detection.landmarks.getMouth());

  const rollDeg = Math.abs(Math.atan2(rightEyeCenter.y - leftEyeCenter.y, rightEyeCenter.x - leftEyeCenter.x) * 180 / Math.PI);
  const eyeMidX = (leftEyeCenter.x + rightEyeCenter.x) / 2;
  const yawDeg = Math.abs(((noseTip.x - eyeMidX) / faceBox.width) * 100);
  const pitchDeg = Math.abs(((noseTip.y - (leftEyeCenter.y + rightEyeCenter.y) / 2) / faceBox.height) * 100);
  
  const alignmentGood = rollDeg <= 7 && yawDeg <= 15 && pitchDeg <= 15;
  const minFaceSizeGood = faceHeightRatio >= 0.40 && faceHeightRatio <= 0.75;
  const resolutionGood = width >= 800 && height >= 600;

  let bgSum = 0;
  let bgSumSq = 0;
  let bgCount = 0;
  let faceSum = 0;
  let faceSumSq = 0;
  let faceCount = 0;

  const marginFactor = 0.14;
  const xA = Math.max(0, faceBox.x - faceBox.width * marginFactor);
  const yA = Math.max(0, faceBox.y - faceBox.height * marginFactor);
  const xB = Math.min(width, faceBox.x + faceBox.width * (1 + marginFactor));
  const yB = Math.min(height, faceBox.y + faceBox.height * (1 + marginFactor));

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const idx = (y * width + x) * 4;
      const r = raw.data[idx];
      const g = raw.data[idx + 1];
      const b = raw.data[idx + 2];
      const lum = 0.299 * r + 0.587 * g + 0.114 * b;

      const isFaceArea = x >= xA && x <= xB && y >= yA && y <= yB;
      if (isFaceArea) {
        faceSum += lum;
        faceSumSq += lum * lum;
        faceCount += 1;
      } else {
        bgSum += lum;
        bgSumSq += lum * lum;
        bgCount += 1;
      }
    }
  }

  const bgMean = bgSum / bgCount;
  const bgVar = Math.max(0, bgSumSq / bgCount - bgMean * bgMean);
  const faceMean = faceSum / faceCount;
  const faceVar = Math.max(0, faceSumSq / faceCount - faceMean * faceMean);

  const backgroundGood = bgVar <= 2200;
  const lightingGood = faceMean >= 60 && faceMean <= 190 && faceVar <= 4500;

  return {
    res: {
      faceDetected: true,
      alignment: alignmentGood,
      faceSize: minFaceSizeGood,
      resolution: resolutionGood,
      background: backgroundGood,
      lighting: lightingGood
    },
    metrics: {
      rollDeg: rollDeg.toFixed(1),
      yawDeg: yawDeg.toFixed(1),
      pitchDeg: pitchDeg.toFixed(1),
      faceHeightRatio: (faceHeightRatio * 100).toFixed(1),
      faceWidthRatio: (faceWidthRatio * 100).toFixed(1),
      resolution: `${width}x${height}`,
      bgVariance: bgVar.toFixed(1),
      faceVariance: faceVar.toFixed(1),
      faceBrightness: faceMean.toFixed(1)
    }
  };
}

async function loadImageFromDataURL(dataUrl) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = dataUrl;
  });
}

function getGrade(score) {
  if (score >= 90) return 'A';
  if (score >= 75) return 'B';
  return 'C';
}

function App() {
  const [modelsReady, setModelsReady] = useState(false);
  const [capturedImage, setCapturedImage] = useState(null);
  const [report, setReport] = useState(initialReport);
  const [analysisSummary, setAnalysisSummary] = useState([]);
  const [stateMessage, setStateMessage] = useState('Loading face API models...');
  const [qualityScore, setQualityScore] = useState(null);
  const [qualityGrade, setQualityGrade] = useState('-');
  const [framingGuidance, setFramingGuidance] = useState('Waiting for face in webcam...');

  const [availableCameras, setAvailableCameras] = useState([]);
  const [selectedCameraId, setSelectedCameraId] = useState(null);
  const [theme, setTheme] = useState('light');
  const [autoCapture, setAutoCapture] = useState(false);
  const [lastAutoCapture, setLastAutoCapture] = useState(0);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const [mpDetected, setMpDetected] = useState(false);
  const [facesDetected, setFacesDetected] = useState(0);
  const [boundingBox, setBoundingBox] = useState([]);
  const [mpIsLoading, setMpIsLoading] = useState(true);

  const webcamRef = useRef(null);
  const webcamOverlayRef = useRef(null);
  const capturedCanvasRef = useRef(null);

  useEffect(() => {
    document.body.classList.remove('theme-light', 'theme-dark');
    document.body.classList.add(`theme-${theme}`);
  }, [theme]);

  useEffect(() => {
    async function initDevices() {
      if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const cameras = devices.filter(d => d.kind === 'videoinput');
        setAvailableCameras(cameras);
        if (!selectedCameraId && cameras.length > 0) {
          setSelectedCameraId(cameras[0].deviceId);
        }
      } catch (err) {
        console.error('Unable to enumerate media devices', err);
      }
    }

    initDevices();
    if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
      navigator.mediaDevices.addEventListener('devicechange', initDevices);
    }

    return () => {
      if (navigator.mediaDevices && navigator.mediaDevices.removeEventListener) {
        navigator.mediaDevices.removeEventListener('devicechange', initDevices);
      }
    };
  }, [selectedCameraId]);

  useEffect(() => {
    if (!webcamRef.current?.video || !modelsReady) return;

    let isActive = true;
    const interval = setInterval(async () => {
      if (!isActive) return;
      const video = webcamRef.current?.video;
      if (!video || video.readyState !== 4) return;

      const results = await faceapi
        .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.45 }))
        .withFaceLandmarks(true);

      if (results && results.length > 0) {
        setMpDetected(true);
        setFacesDetected(results.length);
        setBoundingBox(results.map(det => ({
          xMin: det.detection.box.x / video.videoWidth,
          yMin: det.detection.box.y / video.videoHeight,
          xMax: (det.detection.box.x + det.detection.box.width) / video.videoWidth,
          yMax: (det.detection.box.y + det.detection.box.height) / video.videoHeight,
          xCenter: (det.detection.box.x + det.detection.box.width / 2) / video.videoWidth,
          yCenter: (det.detection.box.y + det.detection.box.height / 2) / video.videoHeight,
        })));
      } else {
        setMpDetected(false);
        setFacesDetected(0);
        setBoundingBox([]);
      }
    }, 250);

    return () => {
      isActive = false;
      clearInterval(interval);
    };
  }, [modelsReady, selectedCameraId]);

  useEffect(() => {
    (async function loadModels() {
      try {
        if (faceapi.tf) {
          await faceapi.tf.setBackend('webgl');
          await faceapi.tf.ready();
        }

        try {
          await faceapi.nets.tinyFaceDetector.loadFromUri(REMOTE_MODEL_URL);
          await faceapi.nets.faceLandmark68Net.loadFromUri(REMOTE_MODEL_URL);
          await faceapi.nets.faceLandmark68TinyNet.loadFromUri(REMOTE_MODEL_URL);
          setStateMessage('Loaded remote models from CDN.');
        } catch (remoteErr) {
          console.warn('Remote models failed, trying local.', remoteErr);
          await faceapi.nets.tinyFaceDetector.loadFromUri(LOCAL_MODEL_URL);
          await faceapi.nets.faceLandmark68Net.loadFromUri(LOCAL_MODEL_URL);
          await faceapi.nets.faceLandmark68TinyNet.loadFromUri(LOCAL_MODEL_URL);
          setStateMessage('Loaded local models from /models.');
        }

        setModelsReady(true);
      } catch (err) {
        setStateMessage('Failed to load face-api models. Check network or setup /models folder.');
        console.error(err);
      }
    })();
  }, []);

  useEffect(() => {
    const canvas = webcamOverlayRef.current;
    const video = webcamRef.current?.video;
    if (!canvas || !video) return;

    const drawOverlay = () => {
      const rect = video.getBoundingClientRect();
      canvas.width = rect.width;
      canvas.height = rect.height;
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, rect.width, rect.height);

      if (!mpDetected || !boundingBox?.length) {
        setFramingGuidance('No face bounding box from MediaPipe yet.');
        return;
      }

      const bb = boundingBox[0];
      if (!bb) {
        setFramingGuidance('No bounding box data available.');
        return;
      }

      const isMirrored = true;
      let xMin = bb.xMin * rect.width;
      const yMin = bb.yMin * rect.height;
      const w = (bb.xMax - bb.xMin) * rect.width;
      const h = (bb.yMax - bb.yMin) * rect.height;

      if (isMirrored) {
        xMin = rect.width - (bb.xMax * rect.width);
      }

      const centerOK = Math.abs((bb.xCenter || (bb.xMin + bb.xMax) / 2) - 0.5) <= 0.1 &&
        Math.abs((bb.yCenter || (bb.yMin + bb.yMax) / 2) - 0.5) <= 0.1;
      const sizeRatio = h / rect.height;
      const sizeOK = sizeRatio >= 0.4 && sizeRatio <= 0.75;

      const good = centerOK && sizeOK;
      ctx.strokeStyle = good ? 'rgba(42, 203, 53, 0.9)' : 'rgba(236, 80, 86, 0.95)';
      ctx.lineWidth = 3;
      ctx.strokeRect(xMin, yMin, w, h);

      setFramingGuidance(good
        ? `Framing good: center ${centerOK ? 'OK' : 'off-center'}, size ${sizeOK ? 'OK' : 'bad'} (${(sizeRatio * 100).toFixed(0)}%).`
        : `Adjust framing: center ${centerOK ? 'OK' : 'off-center'}, size ${sizeOK ? 'OK' : 'bad'} (${(sizeRatio * 100).toFixed(0)}%).`);
    };

    drawOverlay();
    const timer = setInterval(drawOverlay, 100);
    return () => clearInterval(timer);
  }, [mpDetected, boundingBox]);

  const globalStatus = useMemo(() => {
    if (!modelsReady) return 'Loading face API models...';
    const webcamReady = !!webcamRef.current?.video;
    if (!webcamReady) return 'Starting webcam...';
    if (!mpDetected) return 'Webcam ready, no face detected yet. Position your face in frame.';
    return `Face detected by MediaPipe (${facesDetected || 0}); click Capture to run ICAO checks.`;
  }, [modelsReady, mpDetected, facesDetected]);

  useEffect(() => {
    if (!autoCapture || !modelsReady) return;

    const interval = setInterval(() => {
      if (!mpDetected || isAnalyzing) return;

      const now = Date.now();
      if (now - lastAutoCapture < 1000) return;

      setLastAutoCapture(now);
      captureAndAnalyze();
    }, 500);

    return () => clearInterval(interval);
  }, [autoCapture, modelsReady, mpDetected, lastAutoCapture, isAnalyzing]);

  const drawCapturedOverlay = (img, detection) => {
    const canvas = capturedCanvasRef.current;
    if (!canvas || !img) return;
    const ctx = canvas.getContext('2d');
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.clearRect(0, 0, img.width, img.height);
    ctx.drawImage(img, 0, 0, img.width, img.height);

    if (!detection) return;

    const box = detection.detection.box;
    ctx.strokeStyle = 'rgba(42, 203, 53, 0.85)';
    ctx.lineWidth = 4;
    ctx.strokeRect(box.x, box.y, box.width, box.height);

    try {
      faceapi.draw.drawFaceLandmarks(canvas, detection);
    } catch (err) {
      console.warn('drawFaceLandmarks failed', err);
    }

    const leftEye = detection.landmarks.getLeftEye();
    const rightEye = detection.landmarks.getRightEye();
    const nose = detection.landmarks.getNose();

    ctx.strokeStyle = 'rgba(255, 255, 0, 0.9)';
    ctx.lineWidth = 2;

    const drawBox = (points) => {
      const xs = points.map(p => p.x);
      const ys = points.map(p => p.y);
      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      const minY = Math.min(...ys);
      const maxY = Math.max(...ys);
      ctx.strokeRect(minX - 4, minY - 4, maxX - minX + 8, maxY - minY + 8);
    };

    drawBox(leftEye);
    drawBox(rightEye);
    drawBox(nose);
  };

  async function captureAndAnalyze() {
    if (isAnalyzing) {
      return;
    }
    if (!webcamRef.current) return;

    setIsAnalyzing(true);
    try {
      const screenshot = webcamRef.current.getScreenshot({ width: 1280, height: 960 });
      if (!screenshot) {
        setStateMessage('Unable to capture webcam frame.');
        return;
      }

      setCapturedImage(screenshot);
      setStateMessage('Analyzing captured image with face-api.js...');

      const img = await loadImageFromDataURL(screenshot);
      let detection = await faceapi
        .detectSingleFace(img, new faceapi.TinyFaceDetectorOptions({ inputSize: 256, scoreThreshold: 0.4 }))
        .withFaceLandmarks(true);

      if (!detection) {
        const backup = await faceapi
          .detectSingleFace(img, new faceapi.TinyFaceDetectorOptions({ inputSize: 256, scoreThreshold: 0.3 }))
          .withFaceLandmarks(true, { useTinyModel: true });

        if (backup) {
          console.warn('Primary landmark model missed; using backup tiny landmark model.');
          detection = backup;
        }
      }

      if (!detection) {
        setReport(initialReport.map(item => ({ ...item, passed: false, message: 'Face not detected' })));
        setAnalysisSummary(['No face detected in the image.']);
        setQualityScore(0);
        setQualityGrade('C');
        drawCapturedOverlay(img, null);
        setStateMessage('Face not detected. Retake with a clearer view.');
        return;
      }

      const { res, metrics } = analyzeImageData(img, detection);
      drawCapturedOverlay(img, detection);

      console.info(JSON.stringify(res));
      //{"faceDetected":true,"alignment":false,"faceSize":true,"resolution":true,"background":false,"lighting":true}
      const nextReport = initialReport.map(rule => {
        const passed = res[rule.id];
        return { ...rule, passed, message: passed ? 'Pass' : 'Fail' };
      });

      const successes = Object.values(res).filter(v => !!v).length;
      const score = Math.round((successes / Object.keys(res).length) * 100);
      const grade = getGrade(score);

      setReport(nextReport);
      setQualityScore(score);
      setQualityGrade(grade);

      setAnalysisSummary([
        `Roll ${metrics.rollDeg}° (<=7° allowed)`,
        `Yaw ${metrics.yawDeg}° (<=15° allowed)`,
        `Pitch ${metrics.pitchDeg}° (<=15° allowed)`,
        `Face height ratio ${metrics.faceHeightRatio}%`,
        `Resolution ${metrics.resolution}`,
        `Background variance ${metrics.bgVariance}`,
        `Lighting variance ${metrics.faceVariance}, brightness ${metrics.faceBrightness}`
      ]);
      setStateMessage('Analysis complete. See structured compliance report below.');
    } catch (err) {
      console.error(err);
      setStateMessage('Error during analysis. Review console for details.');
    } finally {
      setIsAnalyzing(false);
    }
  }

  const overallPass = report.every(item => item.passed === true);

  return (
    <div className="app-container">
      <h1>Passport / ID Photo ICAO Compliance</h1>

      <div className="panel">
        <div>
          <div className="page-top-right">
            <label className="theme-selector">
              Theme:&nbsp;
              <select value={theme} onChange={e => setTheme(e.target.value)}>
                <option value="dark">Dark</option>
                <option value="light">Light</option>
              </select>
            </label>
          </div>
          <div className="status"><strong>Model/Webcam status:</strong> {globalStatus}</div>
          <div className="controls">
            <button onClick={captureAndAnalyze} disabled={!modelsReady || !mpDetected || isAnalyzing}>Capture & Validate</button>
            <button onClick={() => {
              setReport(initialReport);
              setAnalysisSummary([]);
              setQualityScore(null);
              setQualityGrade('-');
              setCapturedImage(null);
              setStateMessage('Reset to initial state.');
            }} disabled={!modelsReady}>Reset Report</button>
            <label className="auto-capture-label">
              <input
                type="checkbox"
                checked={autoCapture}
                onChange={() => setAutoCapture(prev => !prev)}
                disabled={!modelsReady}
              />
              Auto Capture & Validate (1s cooldown)
            </label>
            <div className="status">
              Auto-capture: {autoCapture ? 'On' : 'Off'} | Last run: {lastAutoCapture ? new Date(lastAutoCapture).toLocaleTimeString() : 'Never'}
            </div>

          {availableCameras.length > 0 && (
            <div className="camera-selector">
              <label>
                Select camera:&nbsp;
                <select value={selectedCameraId || ''} onChange={e => setSelectedCameraId(e.target.value)}>
                  {availableCameras.map((cam, index) => (
                    <option key={cam.deviceId} value={cam.deviceId}>
                      {cam.label || `Camera ${index + 1}`}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          )}
        </div>
        </div>
        
      </div>

      <div className="video-row">
        <div className="panel webcam-box">
          <h2>Live Webcam with Framing Guidance</h2>          
          <div className="webcam-wrapper">
            <Webcam
              key={selectedCameraId || 'default-camera'}
              audio={false}
              ref={webcamRef}
              screenshotFormat="image/jpeg"
              videoConstraints={selectedCameraId ? { deviceId: { exact: selectedCameraId }, width: 640, height: 480 } : { facingMode: 'user', width: 640, height: 480 }}
              className="webcam-video"
            />
            <canvas ref={webcamOverlayRef} className="webcam-overlay" />
          </div>
          <div className="status">Faces detected by MediaPipe: {facesDetected}</div>
          <div className="status"><strong>Framing:</strong> {framingGuidance}</div>
        </div>

        <div className="panel photo-box">
          <h2>Captured frame with landmark overlay</h2>
          <canvas ref={capturedCanvasRef} className="captured-canvas" />
          {!capturedImage && <div style={{ padding: '1rem', color: '#777' }}>No snapshot captured yet.</div>}
        </div>
      </div>

      <div className="panel">
        <h2>ICAO Compliance Report</h2>
        <p><strong>Quality Score:</strong> {qualityScore !== null ? `${qualityScore}%` : 'N/A'}&nbsp;|&nbsp;<strong>Grade:</strong> {qualityGrade}</p>
        <table className="rule-table">
          <thead>
            <tr>
              <th>Rule</th>
              <th>Outcome</th>
              <th>Details</th>
            </tr>
          </thead>
          <tbody>
            {report.map(item => (
              <tr key={item.id}>
                <td>{item.label}</td>
                <td className={item.passed ? 'pass' : item.passed === false ? 'fail' : ''}>
                  {item.passed === null ? 'Pending' : item.passed ? 'Pass' : 'Fail'}
                </td>
                <td>{item.message}</td>
              </tr>
            ))}
          </tbody>
        </table>

        <div className="status">
          Overall: <span className={overallPass ? 'pass' : 'fail'}>{overallPass ? 'ICAO compliant (pass)' : 'Not compliant (fail)'}</span>
        </div>

        <div>
          <h3>Analysis metrics</h3>
          <ul className="summary-list">
            {analysisSummary.map((item, idx) => <li key={idx}>{item}</li>)}
          </ul>
        </div>
      </div>

      <div className="panel">
        <h3>Tips for better passport/ID photos</h3>
        <ul>
          <li>Maintain a straight head pose, low tilt and no yaw.</li>
          <li>Keep the face centered and occupying 40-75% of image height.</li>
          <li>Use a plain, light, uniform background with no glare.</li>
          <li>Ensure high contrast/lighting across the face and avoid deep shadows.</li>
          <li>Capture at 800x600 or greater resolution.</li>
        </ul>
      </div>

      <div className="status"><em>{stateMessage}</em></div>
    </div>
  );
}

export default App;
