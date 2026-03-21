import { removeBackground } from '@imgly/background-removal';

/**
 * Removes background from an image using @imgly/background-removal
 * @param {string} imageSrc - Data URL or image source
 * @param {object} options - Options for processing
 * @param {function} options.progress - Progress callback function
 * @returns {Promise<string>} - Data URL of the processed image with background removed
 */
export async function removeImageBackground(imageSrc, options = {}) {
  try {
    console.log('Starting background removal process...');
    
    // Load image from data URL
    const blob = await fetch(imageSrc).then(res => res.blob());
    
    // Process image with @imgly/background-removal
    const processedBlob = await removeBackground(blob, options);
    
    // Convert blob back to data URL
    const reader = new FileReader();
    return new Promise((resolve, reject) => {
      reader.onloadend = () => {
        console.log('Background removal completed successfully');
        resolve(reader.result);
      };
      reader.onerror = reject;
      reader.readAsDataURL(processedBlob);
    });
  } catch (error) {
    console.error('Error removing background:', error);
    throw new Error(`Failed to remove background: ${error.message}`);
  }
}

/**
 * Downloads a processed image with a custom filename
 * @param {string} imageSrc - Data URL of the image
 * @param {string} filename - Desired filename
 */
export function downloadProcessedImage(imageSrc, filename = 'processed-photo.png') {
  try {
    const link = document.createElement('a');
    link.href = imageSrc;
    link.download = filename;
    link.click();
    console.log(`Image downloaded as ${filename}`);
  } catch (error) {
    console.error('Error downloading image:', error);
    throw new Error(`Failed to download image: ${error.message}`);
  }
}
