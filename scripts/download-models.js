const fs = require('fs');
const path = require('path');
const https = require('https');

const modelFiles = [
  'tiny_face_detector_model-shard1',
  'tiny_face_detector_model-weights_manifest',
  'face_landmark_68_model-shard1',
  'face_landmark_68_model-weights_manifest'
];
const baseUrl = 'https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js@master/weights';
const outputDir = path.resolve(__dirname, '..', 'public', 'models');

async function ensureDir() {
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }
}

function downloadFile(name) {
  return new Promise((resolve, reject) => {
    const url = `${baseUrl}/${name}.json`;
    const filePath = path.join(outputDir, `${name}.json`);
    const file = fs.createWriteStream(filePath);
    https.get(url, (res) => {
      if (res.statusCode !== 200) {
        reject(new Error(`Failed to download ${url}, status: ${res.statusCode}`));
        return;
      }
      res.pipe(file);
      file.on('finish', () => {
        file.close(resolve);
      });
    }).on('error', (err) => {
      fs.unlinkSync(filePath);
      reject(err);
    });
  });
}

function downloadBin(url, dest) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(dest);
    https.get(url, (res) => {
      if (res.statusCode !== 200) {
        reject(new Error(`Failed to download ${url}, status: ${res.statusCode}`));
        return;
      }
      res.pipe(file);
      file.on('finish', () => file.close(resolve));
    }).on('error', (err) => {
      fs.unlinkSync(dest);
      reject(err);
    });
  });
}

async function main() {
  await ensureDir();

  console.log('Downloading model manifests and data (tiny_face_detector + face_landmark_68)');

  // Download both json and weight shards
  for (const modelName of ['tiny_face_detector_model', 'face_landmark_68_model']) {
    const manifestUrl = `${baseUrl}/${modelName}-weights_manifest.json`;
    const manifestPath = path.join(outputDir, `${modelName}-weights_manifest.json`);
    console.log('Downloading', manifestUrl);
    await new Promise((resolve, reject) => {
      const req = https.get(manifestUrl, (res) => {
        if (res.statusCode !== 200) {
          reject(new Error(`Status ${res.statusCode}`));
          return;
        }
        const file = fs.createWriteStream(manifestPath);
        res.pipe(file);
        file.on('finish', () => file.close(resolve));
      });
      req.on('error', reject);
    });

    const manifestContent = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
    const manifestEntries = Array.isArray(manifestContent) ? manifestContent : [manifestContent];
    let downloadedShard = false;

    for (const entry of manifestEntries) {
      const weightEntries = entry.weights || [];
      for (const weightEntry of weightEntries) {
        const weightFile = Array.isArray(weightEntry.paths) && weightEntry.paths[0];
        if (!weightFile) continue;
        const remotePath = `${baseUrl}/${weightFile}`;
        const localPath = path.join(outputDir, weightFile);
        if (!fs.existsSync(localPath)) {
          console.log('Downloading', remotePath);
          await downloadBin(remotePath, localPath);
        } else {
          console.log('Already downloaded', weightFile);
        }
      }

      if (weightEntries.length && !downloadedShard) {
        // Some manifests only contain weight metadata (no paths). Download the default shard file(s).
        const shardName = `${modelName}-shard1`;
        const shardPath = path.join(outputDir, shardName);
        if (!fs.existsSync(shardPath)) {
          console.log('Downloading shard fallback', `${baseUrl}/${shardName}`);
          await downloadBin(`${baseUrl}/${shardName}`, shardPath);
        } else {
          console.log('Already downloaded shard fallback', shardName);
        }
        downloadedShard = true;
      }
    }
  }

  console.log('Model download complete at', outputDir);
}

main().catch((err) => {
  console.error('Download failed:', err);
  process.exit(1);
});
