export function findPeaks(data: number[], distance: number, prominence: number) {
  const peaks: number[] = [];
  // 1. Find local maxima
  for (let i = 1; i < data.length - 1; i++) {
    if (data[i] > data[i - 1] && data[i] > data[i + 1]) {
      peaks.push(i);
    }
  }

  // 2. Calculate prominence
  const peakProminences = peaks.map(peakIdx => {
    const val = data[peakIdx];
    // Left base
    let leftMin = val;
    for (let i = peakIdx - 1; i >= 0; i--) {
      if (data[i] > val) break; // Found a higher peak
      if (data[i] < leftMin) leftMin = data[i];
    }
    // Right base
    let rightMin = val;
    for (let i = peakIdx + 1; i < data.length; i++) {
      if (data[i] > val) break; // Found a higher peak
      if (data[i] < rightMin) rightMin = data[i];
    }
    const baseMax = Math.max(leftMin, rightMin);
    return { idx: peakIdx, prominence: val - baseMax };
  });

  // 3. Filter by prominence
  let validPeaks = peakProminences.filter(p => p.prominence >= prominence);

  // 4. Filter by distance
  validPeaks.sort((a, b) => b.prominence - a.prominence);
  const finalPeaks: number[] = [];
  for (const p of validPeaks) {
    let keep = true;
    for (const fp of finalPeaks) {
      if (Math.abs(p.idx - fp) < distance) {
        keep = false;
        break;
      }
    }
    if (keep) {
      finalPeaks.push(p.idx);
    }
  }

  return finalPeaks.sort((a, b) => a - b);
}

export function analyzeImageLocal(
  imageData: ImageData,
  roiRatio: [number, number] = [0.3, 0.7],
  distance: number = 6,
  prominence: number = 15
) {
  const { width, height, data } = imageData;
  const y1 = Math.floor(height * roiRatio[0]);
  const y2 = Math.floor(height * roiRatio[1]);
  const roiHeight = y2 - y1;

  // Convert to grayscale
  const gray = new Float32Array(width * height);
  for (let i = 0; i < width * height; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    gray[i] = 0.299 * r + 0.587 * g + 0.114 * b;
  }

  // Sobel Y on ROI
  const projection = new Float32Array(roiHeight);

  for (let y = y1 + 1; y < y2 - 1; y++) {
    let rowSum = 0;
    for (let x = 1; x < width - 1; x++) {
      // Sobel Y kernel:
      // -1 -2 -1
      //  0  0  0
      //  1  2  1
      const val =
        -1 * gray[(y - 1) * width + (x - 1)] - 2 * gray[(y - 1) * width + x] - 1 * gray[(y - 1) * width + (x + 1)] +
         1 * gray[(y + 1) * width + (x - 1)] + 2 * gray[(y + 1) * width + x] + 1 * gray[(y + 1) * width + (x + 1)];

      rowSum += Math.abs(val);
    }
    projection[y - y1] = rowSum / (width - 2);
  }

  // Normalize projection to 0-255
  const maxProj = Math.max(...Array.from(projection));
  if (maxProj > 0) {
    for (let i = 0; i < projection.length; i++) {
      projection[i] = (projection[i] / maxProj) * 255;
    }
  }

  const peaks = findPeaks(Array.from(projection), distance, prominence);

  // Map peaks back to original image coordinates
  const absolutePeaks = peaks.map(p => p + y1);

  return { count: peaks.length, peaks: absolutePeaks, projection: Array.from(projection), y1, y2, height };
}
