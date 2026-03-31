import React, { useEffect, useMemo, useRef, useState } from 'react';

import Webcam from 'react-webcam';
import * as faceapi from '@vladmandic/face-api';
import AnalysisMetrics from './AnalysisMetrics';
import CompliantPhotos from './CompliantPhotos';

const width = 500;
const height = 500;

const LOCAL_MODEL_URL = '/models';
const REMOTE_MODEL_URL = 'https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js@master/weights';

//{"faceDetected":true,"alignment":false,"faceSize":true,"resolution":true,"background":false,"lighting":true}
const initialReport = [
  { id: 'faceDetected', label: 'Face detected', passed: null, message: 'Waiting for detection' },
  { id: 'neutralExpression', label: 'Neutral expression', passed: null, message: 'Waiting for detection' },
  { id: 'alignment', label: 'Head alignment (roll/yaw/pitch)', passed: null, message: 'Waiting for detection' },
  { id: 'centered', label: 'Face centered', passed: null, message: 'Waiting for detection' },
  { id: 'faceSize', label: 'Face size ratio (>=60% && <=75%)', passed: null, message: 'Waiting for detection' },
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
  
  const eyeMidX = (leftEyeCenter.x + rightEyeCenter.x) / 2;
  const eyeMidY = (leftEyeCenter.y + rightEyeCenter.y) / 2;

  // const rollDeg = Math.abs(Math.atan2(rightEyeCenter.y - leftEyeCenter.y, rightEyeCenter.x - leftEyeCenter.x) * 180 / Math.PI);
  // const yawDeg = Math.abs(((noseTip.x - eyeMidX) / faceBox.width) * 100);
  // const pitchDeg = Math.abs(Math.atan2(noseTip.y - eyeMidY, faceBox.height * 0.5) * 180 / Math.PI);
  const rollDeg = detection.angle?.roll || 0;
  const yawDeg = detection.angle?.yaw || 0;
  const pitchDeg = detection.angle?.pitch || 0;

  const alignmentGood = rollDeg <= 7 && yawDeg <= 15 && pitchDeg <= 10;
  const minFaceSizeGood = faceHeightRatio >= 0.60 && faceHeightRatio <= 0.75;
  const resolutionGood = width >= 800 && height >= 600;

  //expressão neutra: neutral >= 0.95 e todas as outras expressões < 0.05
  const neutralThreshold = 0.95;   // mínimo aceitável para neutral
  const otherThreshold = 0.05;     // máximo aceitável para outras expressões
  let neutralExpressionGood = false;
  // Verifica se neutral é suficientemente alto
  if (detection?.expressions?.neutral >= neutralThreshold) {        
    // Verifica se alguma outra expressão ultrapassa o limite
    for (const [expression, value] of Object.entries(detection?.expressions || {})) {
      if (expression !== "neutral" && value > otherThreshold) {
        neutralExpressionGood = false;
        break;
      }
    }
    neutralExpressionGood = true;
  } else {  
    neutralExpressionGood = false;
  }

  //Analisa se rosto esta centralizado
  function checkFaceCentered(detection, videoWidth, videoHeight, tolerance = 50) {
    if (!detection) return false;
    const faceCenterX = detection.box.x + detection.box.width / 2;
    //const faceCenterY = detection.box.y + detection.box.height / 2;

    const frameCenterX = videoWidth / 2;
    //const frameCenterY = videoHeight / 2;

    const centeredX = Math.abs(faceCenterX - frameCenterX) < tolerance;
    //const centeredY = Math.abs(faceCenterY - frameCenterY) < tolerance;

    return centeredX;// && centeredY;
  }
  const isCentered = checkFaceCentered(detection.detection, width, height);

  
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
      lighting: lightingGood,
      neutralExpression: neutralExpressionGood,
      centered: isCentered
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
  const [autoCapture, setAutoCapture] = useState(true);
  const [lastAutoCapture, setLastAutoCapture] = useState(0);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const [autoCompliant, setAutoCompliant] = useState(false);
  const [lastCompliantTime, setLastCompliantTime] = useState(0);
  const [compliantSnapshots, setCompliantSnapshots] = useState([]);

  const [mpDetected, setMpDetected] = useState(false);
  const [fps, setFps] = useState("0");
  const [facesDetected, setFacesDetected] = useState(0);
  const [boundingBox, setBoundingBox] = useState([]);
  const [detections, setDetections] = useState([]);
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
      const t0 = performance.now();

      const results = await faceapi
        .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.45 }))
        .withFaceLandmarks(true);

      if (results && results.length > 0) {
        const fps = 1000 / (performance.now() - t0);
        setFps(fps.toLocaleString());
        setMpDetected(true);
        setFacesDetected(results.length);
        setDetections(results);
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
        setDetections([]);
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
          await faceapi.nets.faceExpressionNet.loadFromUri(REMOTE_MODEL_URL);
          setStateMessage('Loaded remote models from CDN.');
        } catch (remoteErr) {
          console.warn('Remote models failed, trying local.', remoteErr);
          await faceapi.nets.tinyFaceDetector.loadFromUri(LOCAL_MODEL_URL);
          await faceapi.nets.faceLandmark68Net.loadFromUri(LOCAL_MODEL_URL);
          await faceapi.nets.faceLandmark68TinyNet.loadFromUri(LOCAL_MODEL_URL);
          await faceapi.nets.faceExpressionNet.loadFromUri(LOCAL_MODEL_URL);
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

      // Desenha máscara oval
      ctx.fillStyle = "rgba(0,0,0,0.6)";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.globalCompositeOperation = "destination-out";
      ctx.beginPath();
      ctx.ellipse(
        canvas.width / 2,
        canvas.height / 2,
        120, // raio horizontal
        150, // raio vertical
        0,
        0,
        2 * Math.PI
      );
      ctx.fill();
      ctx.globalCompositeOperation = "source-over";
      
      
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

      // Draw landmarks if available
      if (detections && detections.length > 0) {
        const detection = detections[0];
        if (detection.landmarks) {
          const scaleX = rect.width / video.videoWidth;
          const scaleY = rect.height / video.videoHeight;

          // Helper to draw landmark points and boxes
          const drawLandmarkBox = (points, color) => {
            if (!points || points.length === 0) return;
            const xs = points.map(p => (isMirrored ? video.videoWidth - p.x : p.x) * scaleX);
            const ys = points.map(p => p.y * scaleY);
            const minX = Math.min(...xs);
            const maxX = Math.max(...xs);
            const minY = Math.min(...ys);
            const maxY = Math.max(...ys);
            
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.strokeRect(minX - 4, minY - 4, maxX - minX + 8, maxY - minY + 8);

            // Draw points
            ctx.fillStyle = color;
            points.forEach(p => {
              const px = (isMirrored ? video.videoWidth - p.x : p.x) * scaleX;
              const py = p.y * scaleY;
              ctx.beginPath();
              ctx.arc(px, py, 3, 0, 2 * Math.PI);
              ctx.fill();
            });
          };

          //Draw eyes, nose, and mouth
          // try {
          //   drawLandmarkBox(detection.landmarks.getLeftEye(), 'rgba(255, 0, 0, 0.8)');
          //   drawLandmarkBox(detection.landmarks.getRightEye(), 'rgba(255, 0, 0, 0.8)');
          //   drawLandmarkBox(detection.landmarks.getNose(), 'rgba(0, 255, 0, 0.8)');
          //   drawLandmarkBox(detection.landmarks.getMouth(), 'rgba(0, 0, 255, 0.8)');
          // } catch (err) {
          //   console.warn('Error drawing landmarks:', err);
          // }
        }
      }

      setFramingGuidance(good
        ? `Framing good: center ${centerOK ? 'OK' : 'off-center'}, size ${sizeOK ? 'OK' : 'bad'} (${(sizeRatio * 100).toFixed(0)}%). FPS: ${fps}`
        : `Adjust framing: center ${centerOK ? 'OK' : 'off-center'}, size ${sizeOK ? 'OK' : 'bad'} (${(sizeRatio * 100).toFixed(0)}%). FPS: ${fps}`);
    };

    drawOverlay();
    const timer = setInterval(drawOverlay, 100);
    return () => clearInterval(timer);
  }, [mpDetected, boundingBox, detections]);

  const globalStatus = useMemo(() => {
    if (!modelsReady) return 'Loading face API models...';
    const webcamReady = !!webcamRef.current?.video;
    if (!webcamReady) return 'Starting webcam...';
    if (!mpDetected) return 'Webcam ready, no face detected yet. Position your face in frame.';
    if (autoCapture) return `Face detected by MediaPipe (${facesDetected || 0}); real-time automatic ICAO analysis running.`;
    return `Face detected by MediaPipe (${facesDetected || 0}); click Capture to run ICAO checks.`;
  }, [modelsReady, mpDetected, facesDetected, autoCapture]);

  useEffect(() => {
    if (!modelsReady) return;

    if (!mpDetected) {
      setAutoCompliant(false);
      setReport(prev => prev.map(rule => {
        if (rule.id === 'faceDetected') {
          return { ...rule, passed: false, message: 'No face detected in live camera' };
        }
        return rule;
      }));
      setQualityScore(0);
      setQualityGrade('C');
      setStateMessage('No face detected in live stream.');
    }
  }, [mpDetected, modelsReady]);

  useEffect(() => {
    if (!autoCapture || !modelsReady) return;

    const interval = setInterval(() => {
      if (!mpDetected || isAnalyzing) return;

      const now = Date.now();
      if (now - lastAutoCapture < 200) return;
      if (autoCompliant && now - lastCompliantTime < 5000) return;

      setLastAutoCapture(now);
      captureAndAnalyze();
    }, 200);

    return () => clearInterval(interval);
  }, [autoCapture, modelsReady, mpDetected, lastAutoCapture, isAnalyzing, autoCompliant, lastCompliantTime]);

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

    // drawBox(leftEye);
    // drawBox(rightEye);
    // drawBox(nose);
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

      setStateMessage('Analyzing captured image with face-api.js...');

      const img = await loadImageFromDataURL(screenshot);
      let detection = await faceapi
        .detectSingleFace(img, new faceapi.TinyFaceDetectorOptions({ inputSize: 256, scoreThreshold: 0.4 }))
        .withFaceLandmarks(true)
        .withFaceExpressions();

      if (!detection) {
        const backup = await faceapi
          .detectSingleFace(img, new faceapi.TinyFaceDetectorOptions({ inputSize: 256, scoreThreshold: 0.3 }))
          .withFaceLandmarks(true, { useTinyModel: true })
          .withFaceExpressions();

        if (backup) {
          console.warn('Primary landmark model missed; using backup tiny landmark model.');
          detection = backup;
        }
      }

      if (!detection) {
        setAutoCompliant(false);
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
      const allPass = successes === Object.keys(res).length;

      setReport(nextReport);
      setQualityScore(score);
      setQualityGrade(grade);
      setAutoCompliant(allPass);

      if (allPass) {
        const now = Date.now();
        setCapturedImage(screenshot);
        setCompliantSnapshots(prev => [...prev, screenshot]);
        setLastCompliantTime(now);
        setStateMessage('✅ Auto-captured compliant image saved. Auto-capture stopped.');
        setAutoCapture(false);
      } else {
        setStateMessage('⚠️ Auto-analysis complete, not compliant yet.');
      }

      setAnalysisSummary([
        `Roll ${metrics.rollDeg}° (<=7° allowed)`,
        `Yaw ${metrics.yawDeg}° (<=15° allowed)`,
        `Pitch ${metrics.pitchDeg}° (<=10° allowed)`,
        `Face height ratio ${metrics.faceHeightRatio}%`,
        `Resolution ${metrics.resolution}`,
        `Background variance ${metrics.bgVariance}`,
        `Lighting variance ${metrics.faceVariance}, brightness ${metrics.faceBrightness}`
      ]);
    } catch (err) {
      console.error(err);
      setAutoCompliant(false);
      setStateMessage('Error during analysis. Review console for details.');
    } finally {
      setIsAnalyzing(false);
    }
  }

  const overallPass = report.every(item => item.passed === true);

  const downloadCompliantPhoto = () => {
    if (compliantSnapshots.length === 0) return;
    const link = document.createElement('a');
    link.href = compliantSnapshots[compliantSnapshots.length - 1];
    link.download = 'compliant-photo.jpg';
    link.click();
  };

  return (
    <div className="app-container">
      <h1>ID Photo ICAO Compliance</h1>

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
            <button
              onClick={captureAndAnalyze}
              disabled={!modelsReady || !mpDetected || isAnalyzing || autoCapture}
              title={autoCapture ? 'Real-time auto mode is active' : 'Manual capture'}>
              {autoCapture ? 'Real-time running (manual disabled)' : 'Capture & Validate'}
            </button>
            <button onClick={() => {
              setReport(initialReport);
              setAnalysisSummary([]);
              setQualityScore(null);
              setQualityGrade('-');
              setCapturedImage(null);
              setAutoCompliant(false);
              setCompliantSnapshots([]);
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
            <div className="status">
              Auto-compliant: {autoCompliant ? 'Yes' : 'No'}{autoCompliant && lastCompliantTime ? ` (saved at ${new Date(lastCompliantTime).toLocaleTimeString()})` : ''}
            </div>
            {compliantSnapshots.length > 0 && (
              <div className="status">Saved {compliantSnapshots.length} compliant snapshot(s) available.</div>
            )}

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

          {autoCompliant && compliantSnapshots.length > 0 && (
            <div style={{ marginTop: '1rem', padding: '0.75rem', backgroundColor: '#f0f9f0', border: '2px solid #1a8c11', borderRadius: '4px' }}>
              <div style={{ color: '#1a8c11', fontWeight: 'bold', marginBottom: '0.5rem' }}>✅ {compliantSnapshots.length} Compliant snapshot(s) ready</div>
              <button 
                onClick={downloadCompliantPhoto}
                style={{ 
                  padding: '0.5rem 1rem', 
                  backgroundColor: '#1a8c11', 
                  color: 'white', 
                  border: 'none', 
                  borderRadius: '4px', 
                  cursor: 'pointer',
                  fontWeight: 'bold'
                }}
                onMouseOver={(e) => e.target.style.backgroundColor = '#158c0c'}
                onMouseOut={(e) => e.target.style.backgroundColor = '#1a8c11'}
              >
                Download compliant photo
              </button>
            </div>
          )}
        </div>
      </div>

      <div 
        style={{
          position: 'fixed',
          top: '2rem',
          left: '2rem',
          width: '400px',
          maxHeight: '70vh',
          backgroundColor: 'var(--bg-secondary, #f5f5f5)',
          border: '1px solid var(--border-color, #ddd)',
          borderRadius: '8px',
          padding: '1.25rem',
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
          zIndex: 999,
          overflowY: 'auto',
          fontFamily: 'var(--font-family, sans-serif)'
        }}
      >
        <h2 style={{ marginTop: 0, marginBottom: '1rem', fontSize: '0.95rem', fontWeight: 600 }}>ICAO Compliance Report</h2>
        <p style={{ marginBottom: '1rem', fontSize: '0.8rem' }}>
          <strong>Quality Score:</strong> {qualityScore !== null ? `${qualityScore}%` : 'N/A'}&nbsp;|&nbsp;<strong>Grade:</strong> {qualityGrade}
        </p>
        <table className="rule-table" style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8rem' }}>
          <thead>
            <tr>
              <th style={{ textAlign: 'left', padding: '0.5rem', borderBottom: '1px solid #ccc' }}>Rule</th>
              <th style={{ textAlign: 'left', padding: '0.5rem', borderBottom: '1px solid #ccc' }}>Outcome</th>
              {/* <th style={{ textAlign: 'left', padding: '0.5rem', borderBottom: '1px solid #ccc' }}>Details</th> */}
            </tr>
          </thead>
          <tbody>
            {report.map(item => (
              <tr key={item.id} style={{ borderBottom: '1px solid #eee' }}>
                <td style={{ padding: '0.5rem' }}>{item.label}</td>
                <td 
                  style={{ 
                    padding: '0.5rem',
                    color: item.passed ? '#1a8c11' : item.passed === false ? '#c41e3a' : '#666',
                    fontWeight: 'bold'
                  }}>
                  {item.passed === null ? 'Pending' : item.passed ? 'Pass' : 'Fail'}
                </td>
                {/* <td style={{ padding: '0.5rem' }}>{item.message}</td> */}
              </tr>
            ))}
          </tbody>
        </table>

        <div style={{
          marginTop: '1rem',
          padding: '0.75rem',
          borderRadius: '4px',
          backgroundColor: overallPass ? '#f0f9f0' : '#fff5f5',
          border: `2px solid ${overallPass ? '#1a8c11' : '#c41e3a'}`,
          textAlign: 'center',
          fontWeight: 'bold',
          color: overallPass ? '#1a8c11' : '#c41e3a'
        }}>
          {overallPass ? '✅ ICAO compliant (pass)' : '❌ Not compliant (fail)'}
        </div>
      </div>

      <AnalysisMetrics
        analysisSummary={analysisSummary}
        report={report}
        qualityScore={qualityScore}
        qualityGrade={qualityGrade}
        overallPass={overallPass}
      />

      <CompliantPhotos compliantSnapshots={compliantSnapshots} />

      <div className="panel">
      <h3>Dicas para melhores fotos de identificação</h3>
        <ul>
          <li>Mantenha a cabeça reta, com pouca inclinação e sem rotação lateral.</li>
          <li>Mantenha o rosto centralizado e ocupando entre 60% e 75% da altura da imagem.</li>
          <li>Use um fundo claro, uniforme e sem reflexos.</li>
          <li>Garanta boa iluminação e contraste no rosto, evitando sombras profundas.</li>
          <li>Capture em resolução mínima de 800x600 ou superior.</li>
        </ul>
      </div>      

      <div className="status"><em>{stateMessage}</em></div>
    </div>
  );
}

export default App;
