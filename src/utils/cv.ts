type PeakInfo = {
  idx: number;
  prominence: number;
};

function percentile(values: number[], ratio: number) {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.max(0, Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * ratio)));
  return sorted[index];
}

function mean(values: number[]) {
  if (values.length === 0) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function std(values: number[]) {
  if (values.length === 0) return 0;
  const avg = mean(values);
  const variance = mean(values.map((value) => (value - avg) ** 2));
  return Math.sqrt(variance);
}

function smoothSignal(data: number[], radius: number) {
  if (radius <= 0) return [...data];

  const smoothed = new Array<number>(data.length).fill(0);
  for (let i = 0; i < data.length; i++) {
    let sum = 0;
    let count = 0;
    for (let j = Math.max(0, i - radius); j <= Math.min(data.length - 1, i + radius); j++) {
      sum += data[j];
      count += 1;
    }
    smoothed[i] = sum / count;
  }
  return smoothed;
}

function normalizeSignal(data: number[]) {
  const floor = percentile(data, 0.1);
  const ceil = percentile(data, 0.995);
  const span = Math.max(1, ceil - floor);
  return data.map((value) => Math.max(0, Math.min(255, ((value - floor) / span) * 255)));
}

export function findPeaks(data: number[], distance: number, prominence: number) {
  const peaks: number[] = [];
  for (let i = 1; i < data.length - 1; i++) {
    if (data[i] > data[i - 1] && data[i] > data[i + 1]) {
      peaks.push(i);
    }
  }

  const peakProminences: PeakInfo[] = peaks.map((peakIdx) => {
    const val = data[peakIdx];
    let leftMin = val;
    for (let i = peakIdx - 1; i >= 0; i--) {
      if (data[i] > val) break;
      if (data[i] < leftMin) leftMin = data[i];
    }

    let rightMin = val;
    for (let i = peakIdx + 1; i < data.length; i++) {
      if (data[i] > val) break;
      if (data[i] < rightMin) rightMin = data[i];
    }

    return { idx: peakIdx, prominence: val - Math.max(leftMin, rightMin) };
  });

  const validPeaks = peakProminences
    .filter((peak) => peak.prominence >= prominence)
    .sort((a, b) => b.prominence - a.prominence);

  const finalPeaks: number[] = [];
  for (const peak of validPeaks) {
    if (finalPeaks.every((value) => Math.abs(peak.idx - value) >= distance)) {
      finalPeaks.push(peak.idx);
    }
  }

  return finalPeaks.sort((a, b) => a - b);
}

function mergeClosePeaks(peaks: number[], mergeRatio: number) {
  if (peaks.length < 2) return peaks;

  const sorted = [...peaks].sort((a, b) => a - b);
  const spacings = sorted.slice(1).map((value, index) => value - sorted[index]);
  const median = percentile(spacings, 0.5);
  if (median <= 0) return sorted;

  const merged = [sorted[0]];
  for (let i = 1; i < sorted.length; i++) {
    const spacing = sorted[i] - merged[merged.length - 1];
    if (spacing < median * mergeRatio) {
      merged[merged.length - 1] = Math.round((merged[merged.length - 1] + sorted[i]) / 2);
    } else {
      merged.push(sorted[i]);
    }
  }

  return merged;
}

function filterBySpacing(peaks: number[], signal: number[]) {
  if (peaks.length < 4) return peaks;

  const spacings = peaks.slice(1).map((value, index) => value - peaks[index]);
  const median = percentile(spacings, 0.5);
  if (median <= 0) return peaks;

  const filtered = [peaks[0]];
  for (let i = 0; i < spacings.length; i++) {
    const spacing = spacings[i];
    if (spacing >= median * 0.35 && spacing <= median * 3.2) {
      filtered.push(peaks[i + 1]);
    }
  }

  if (filtered.length >= 3) return filtered;

  const peakValues = peaks.map((peak) => signal[peak] ?? 0);
  const sortedIndices = peakValues
    .map((value, index) => ({ value, index }))
    .sort((a, b) => b.value - a.value)
    .slice(0, Math.max(2, Math.round(peaks.length * 0.75)))
    .map((item) => item.index)
    .sort((a, b) => a - b);

  return sortedIndices.map((index) => peaks[index]);
}

function scorePeaks(signal: number[], peaks: number[], regionHeight: number) {
  if (peaks.length < 2) return 0;

  const peakValues = peaks.map((peak) => signal[peak] ?? 0);
  const spacings = peaks.slice(1).map((value, index) => value - peaks[index]);
  const spacingCv = spacings.length > 1 ? std(spacings) / Math.max(1, mean(spacings)) : 1;
  const prominenceScore = Math.min(1, mean(peakValues) / Math.max(1, mean(signal)) / 2);
  const spacingScore = spacingCv < 0.5 ? 1 : Math.max(0, 1 - (spacingCv - 0.5) * 0.8);
  const densityScore = Math.min(1, peaks.length / Math.max(2, regionHeight / 18));
  return prominenceScore * 0.45 + spacingScore * 0.35 + densityScore * 0.2;
}

function estimateDominantSpacing(signal: number[], regionHeight: number) {
  if (signal.length < 24) return null;

  const centered = signal.map((value) => value - mean(signal));
  const minLag = Math.max(4, Math.round(regionHeight / 120));
  const maxLag = Math.min(24, Math.max(minLag + 2, Math.round(regionHeight / 10)));
  let bestLag = minLag;
  let bestScore = -Infinity;

  for (let lag = minLag; lag <= maxLag; lag++) {
    let numerator = 0;
    let leftEnergy = 0;
    let rightEnergy = 0;
    for (let i = 0; i < centered.length - lag; i++) {
      const left = centered[i];
      const right = centered[i + lag];
      numerator += left * right;
      leftEnergy += left * left;
      rightEnergy += right * right;
    }

    if (leftEnergy === 0 || rightEnergy === 0) continue;
    const corr = numerator / Math.sqrt(leftEnergy * rightEnergy);
    const weighted = corr - lag * 0.003;
    if (weighted > bestScore) {
      bestScore = weighted;
      bestLag = lag;
    }
  }

  if (bestScore < 0.12) return null;
  return bestLag;
}

function findLocalPeakNear(signal: number[], target: number, radius: number) {
  let bestIndex = Math.max(0, Math.min(signal.length - 1, Math.round(target)));
  let bestValue = -Infinity;
  const start = Math.max(0, Math.floor(target - radius));
  const end = Math.min(signal.length - 1, Math.ceil(target + radius));

  for (let i = start; i <= end; i++) {
    if (signal[i] > bestValue) {
      bestValue = signal[i];
      bestIndex = i;
    }
  }

  return bestIndex;
}

function densifyPeaks(signal: number[], peaks: number[], spacing: number) {
  if (peaks.length < 2) return peaks;

  const dense = [peaks[0]];
  const searchRadius = Math.max(2, spacing * 0.4);
  const minCandidateStrength = percentile(signal, 0.62);

  for (let i = 1; i < peaks.length; i++) {
    const previous = dense[dense.length - 1];
    const current = peaks[i];
    const gap = current - previous;

    if (gap > spacing * 1.75 && gap < spacing * 4.2) {
      const steps = Math.max(1, Math.round(gap / spacing) - 1);
      for (let step = 1; step <= steps; step++) {
        const target = previous + (gap * step) / (steps + 1);
        const candidate = findLocalPeakNear(signal, target, searchRadius);
        if (
          signal[candidate] >= minCandidateStrength &&
          candidate - dense[dense.length - 1] >= Math.max(2, spacing * 0.45)
        ) {
          dense.push(candidate);
        }
      }
    }

    if (current - dense[dense.length - 1] >= Math.max(2, spacing * 0.45)) {
      dense.push(current);
    }
  }

  return dense;
}

function detectDensePeaks(signal: number[], regionHeight: number, distance: number, prominence: number) {
  const spacing = estimateDominantSpacing(signal, regionHeight);
  if (!spacing) return null;

  const relaxedDistance = Math.max(3, Math.min(distance, Math.round(spacing * 0.7)));
  const relaxedProminence = Math.max(4, Math.min(prominence, 5));
  let peaks = findPeaks(signal, relaxedDistance, relaxedProminence);
  peaks = peaks.filter((peak) => peak >= Math.floor(signal.length * 0.015) && peak <= Math.floor(signal.length * 0.99));
  peaks = mergeClosePeaks(peaks, 0.24);
  peaks = densifyPeaks(signal, peaks, spacing);
  peaks = mergeClosePeaks(peaks, 0.18);
  peaks = filterBySpacing(peaks, signal);

  return {
    peaks,
    spacing
  };
}

function detectPeaks(signal: number[], regionHeight: number, distance: number, prominence: number) {
  const adaptiveDistance = Math.max(distance, Math.round(regionHeight / 220));
  const adaptiveProminence = Math.max(prominence, regionHeight < 500 ? 5 : regionHeight < 900 ? 6 : 8);
  const mergeRatio = regionHeight < 500 ? 0.32 : regionHeight < 900 ? 0.38 : 0.44;

  let peaks = findPeaks(signal, adaptiveDistance, adaptiveProminence);
  const topClip = Math.floor(signal.length * 0.02);
  const bottomClip = Math.floor(signal.length * 0.985);
  peaks = peaks.filter((peak) => peak >= topClip && peak <= bottomClip);
  peaks = mergeClosePeaks(peaks, mergeRatio);
  peaks = filterBySpacing(peaks, signal);

  return peaks;
}

function getAdaptiveXRange(width: number) {
  const roiWidth = Math.max(16, Math.round(width * 0.4));
  const center = Math.round(width * 0.5);
  const x1 = Math.max(0, center - Math.floor(roiWidth / 2));
  const x2 = Math.min(width, center + Math.ceil(roiWidth / 2));
  return [x1, x2] as const;
}

function isBoundaryRed(r: number, g: number, b: number) {
  return r >= 185 && g <= 140 && b <= 140 && r - Math.max(g, b) >= 65;
}

function detectRedBoundaryLines(imageData: ImageData) {
  const { width, height, data } = imageData;
  const rowHits = new Array<number>(height).fill(0);
  const longestRuns = new Array<number>(height).fill(0);

  for (let y = 0; y < height; y++) {
    let hits = 0;
    let currentRun = 0;
    let longestRun = 0;
    for (let x = 0; x < width; x++) {
      const index = (y * width + x) * 4;
      const r = data[index];
      const g = data[index + 1];
      const b = data[index + 2];

      if (isBoundaryRed(r, g, b)) {
        hits += 1;
        currentRun += 1;
        if (currentRun > longestRun) longestRun = currentRun;
      } else {
        currentRun = 0;
      }
    }
    rowHits[y] = hits;
    longestRuns[y] = longestRun;
  }

  const minCoverage = Math.max(24, Math.floor(width * 0.55));
  const minRun = Math.max(24, Math.floor(width * 0.5));
  const candidateRows: number[] = [];
  for (let y = 0; y < height; y++) {
    if (rowHits[y] >= minCoverage && longestRuns[y] >= minRun) {
      candidateRows.push(y);
    }
  }

  if (candidateRows.length < 2) return null;

  const groups: Array<{ start: number; end: number; center: number; score: number }> = [];
  let groupStart = candidateRows[0];
  let groupEnd = candidateRows[0];

  for (let i = 1; i < candidateRows.length; i++) {
    const row = candidateRows[i];
    if (row <= groupEnd + 2) {
      groupEnd = row;
      continue;
    }

    const thickness = groupEnd - groupStart + 1;
    const coverage = rowHits.slice(groupStart, groupEnd + 1);
    const runs = longestRuns.slice(groupStart, groupEnd + 1);
    const avgCoverage = mean(coverage) / width;
    const avgRun = mean(runs) / width;
    const score = avgCoverage * 0.6 + avgRun * 0.3 + Math.min(0.1, thickness / 20);
    groups.push({
      start: groupStart,
      end: groupEnd,
      center: Math.round((groupStart + groupEnd) / 2),
      score
    });
    groupStart = row;
    groupEnd = row;
  }
  {
    const thickness = groupEnd - groupStart + 1;
    const coverage = rowHits.slice(groupStart, groupEnd + 1);
    const runs = longestRuns.slice(groupStart, groupEnd + 1);
    const avgCoverage = mean(coverage) / width;
    const avgRun = mean(runs) / width;
    const score = avgCoverage * 0.6 + avgRun * 0.3 + Math.min(0.1, thickness / 20);
    groups.push({
      start: groupStart,
      end: groupEnd,
      center: Math.round((groupStart + groupEnd) / 2),
      score
    });
  }

  const validGroups = groups
    .filter((group) => group.end - group.start + 1 >= 2)
    .filter((group) => group.score >= 0.78)
    .sort((a, b) => a.center - b.center);

  if (validGroups.length < 2) return null;

  const top = validGroups[0].center;
  const bottom = validGroups[validGroups.length - 1].center;
  if (bottom - top < Math.max(80, height * 0.1)) return null;

  return { y1: top, y2: bottom };
}

export function analyzeImageLocal(
  imageData: ImageData,
  roiRatio: [number, number] = [0.12, 0.94],
  distance: number = 4,
  prominence: number = 6
) {
  const { width, height, data } = imageData;
  const boundaryLines = detectRedBoundaryLines(imageData);
  const y1 = boundaryLines?.y1 ?? Math.floor(height * roiRatio[0]);
  const y2 = boundaryLines?.y2 ?? Math.floor(height * roiRatio[1]);
  const roiHeight = Math.max(0, y2 - y1);
  const gray = new Float32Array(width * height);

  for (let i = 0; i < width * height; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    gray[i] = 0.299 * r + 0.587 * g + 0.114 * b;
  }

  const [x1, x2] = getAdaptiveXRange(width);
  const roiWidth = Math.max(1, x2 - x1);
  const brightnessProjection = new Array<number>(roiHeight).fill(0);
  const edgeProjection = new Array<number>(roiHeight).fill(0);

  for (let y = y1 + 1; y < y2 - 1; y++) {
    let brightnessSum = 0;
    let edgeSum = 0;

    for (let x = x1 + 1; x < x2 - 1; x++) {
      const center = gray[y * width + x];
      brightnessSum += 255 - center;

      const verticalGradient = Math.abs(gray[(y + 1) * width + x] - gray[(y - 1) * width + x]);
      const leftGradient = Math.abs(center - gray[y * width + (x - 1)]);
      const rightGradient = Math.abs(center - gray[y * width + (x + 1)]);
      edgeSum += verticalGradient * 0.75 + (leftGradient + rightGradient) * 0.25;
    }

    brightnessProjection[y - y1] = brightnessSum / roiWidth;
    edgeProjection[y - y1] = edgeSum / roiWidth;
  }

  const smoothRadius = Math.max(1, Math.round(roiHeight / 260));
  const brightnessSignal = normalizeSignal(smoothSignal(brightnessProjection, smoothRadius));
  const edgeSignal = normalizeSignal(smoothSignal(edgeProjection, smoothRadius));

  const brightnessPeaks = detectPeaks(brightnessSignal, roiHeight, distance, prominence);
  const edgePeaks = detectPeaks(edgeSignal, roiHeight, Math.max(2, distance - 1), Math.max(4, prominence - 2));

  const brightnessScore = scorePeaks(brightnessSignal, brightnessPeaks, roiHeight);
  const edgeScore = scorePeaks(edgeSignal, edgePeaks, roiHeight);

  let peaks = brightnessPeaks;
  let projection = brightnessSignal;
  let score = brightnessScore;

  if (edgePeaks.length > brightnessPeaks.length && edgeScore >= brightnessScore * 0.9) {
    peaks = edgePeaks;
    projection = edgeSignal;
    score = edgeScore;
  }

  const denseSignal = brightnessSignal.map((value, index) => Math.max(value, edgeSignal[index] ?? 0));
  const denseResult = detectDensePeaks(denseSignal, roiHeight, distance, prominence);
  if (denseResult) {
    const denseScore = scorePeaks(denseSignal, denseResult.peaks, roiHeight);
    const denseCountThreshold = Math.max(peaks.length + 4, Math.round(peaks.length * 1.22));
    const spacingThreshold = denseResult.spacing <= Math.max(12, roiHeight / 35);
    const denseRecallMode = roiHeight >= 450 && peaks.length < roiHeight / 18;
    const denseUpperBound = denseResult.peaks.length <= roiHeight / 12.2;

    if (
      denseRecallMode &&
      spacingThreshold &&
      denseUpperBound &&
      denseResult.peaks.length >= denseCountThreshold &&
      denseScore >= score * 0.64
    ) {
      peaks = denseResult.peaks;
      projection = denseSignal;
      score = denseScore;
    }
  }

  const absolutePeaks = peaks.map((peak) => peak + y1);

  return {
    count: absolutePeaks.length,
    peaks: absolutePeaks,
    projection,
    y1,
    y2,
    height,
    boundarySource: boundaryLines ? 'red_lines' : 'ratio'
  };
}
