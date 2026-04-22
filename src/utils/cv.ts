type PeakInfo = {
  idx: number;
  prominence: number;
};

type ImageDataLike = {
  width: number;
  height: number;
  data: Uint8ClampedArray;
  colorSpace?: PredefinedColorSpace;
};

type SignalExtractionMode = 'single-band' | 'multi-band-parallel';

type BandSignalBundle = {
  ratio: number;
  centerX: number;
  brightnessSignal: number[];
  edgeSignal: number[];
  continuitySignal: number[];
  roiCoverage: number;
};

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

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

function coefficientOfVariation(values: number[]) {
  if (values.length < 2) return 0;
  return std(values) / Math.max(1, mean(values));
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

function toGrayscale(imageData: ImageDataLike) {
  const { width, height, data } = imageData;
  const gray = new Float32Array(width * height);
  for (let i = 0; i < width * height; i++) {
    gray[i] = 0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2];
  }
  return gray;
}

function resizeImageData(imageData: ImageDataLike, maxResolution: number) {
  const { width, height, data } = imageData;
  const longestEdge = Math.max(width, height);
  if (maxResolution <= 0 || longestEdge <= maxResolution) {
    return {
      resizedImageData: imageData,
      actualResolution: longestEdge
    };
  }

  const scale = maxResolution / longestEdge;
  const targetWidth = Math.max(1, Math.round(width * scale));
  const targetHeight = Math.max(1, Math.round(height * scale));

  const newData = new Uint8ClampedArray(targetWidth * targetHeight * 4);

  for (let y = 0; y < targetHeight; y++) {
    const srcY = Math.min(height - 1, Math.floor(y / scale));
    const dstOffsetRow = y * targetWidth * 4;
    const srcOffsetRow = srcY * width * 4;
    
    for (let x = 0; x < targetWidth; x++) {
      const srcX = Math.min(width - 1, Math.floor(x / scale));
      const dstOffset = dstOffsetRow + x * 4;
      const srcOffset = srcOffsetRow + srcX * 4;
      
      newData[dstOffset] = data[srcOffset];
      newData[dstOffset + 1] = data[srcOffset + 1];
      newData[dstOffset + 2] = data[srcOffset + 2];
      newData[dstOffset + 3] = data[srcOffset + 3];
    }
  }

  let finalImageData: ImageData;
  const tmpCanvas = document.createElement('canvas');
  const tmpCtx = tmpCanvas.getContext('2d');
  if (tmpCtx && tmpCtx.createImageData) {
    finalImageData = tmpCtx.createImageData(targetWidth, targetHeight);
    finalImageData.data.set(newData);
  } else if (typeof ImageData !== 'undefined') {
    finalImageData = new ImageData(newData, targetWidth, targetHeight);
  } else {
    finalImageData = { width: targetWidth, height: targetHeight, data: newData } as unknown as ImageData;
  }

  return {
    resizedImageData: finalImageData,
    actualResolution: Math.max(targetWidth, targetHeight)
  };
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

  // Use a more relaxed threshold for high-density stacks to catch thinner layers
  const minRatio = peaks.length > 20 ? 0.18 : 0.25;
  const filtered = [peaks[0]];
  for (let i = 0; i < spacings.length; i++) {
    const spacing = spacings[i];
    if (spacing >= median * minRatio && spacing <= median * 3.2) {
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

function scorePeaks(
  signal: number[],
  peaks: number[],
  regionHeight: number
) {
  if (peaks.length < 2) return 0;

  const peakValues = peaks.map((peak) => signal[peak] ?? 0);
  const spacings = peaks.slice(1).map((value, index) => value - peaks[index]);
  const avgSpacing = spacings.reduce((a, b) => a + b, 0) / spacings.length;
  const spacingCv = spacings.length > 1 ? std(spacings) / Math.max(1, avgSpacing) : 1;
  
  // Consistency: Lower CV is better, but less harsh than before
  const consistencyScore = Math.max(0, 1 - spacingCv * 1.2);

  // Intensity: How strong are the peaks compared to signal range
  const minSignal = Math.min(...signal);
  const maxSignal = Math.max(...signal);
  const range = Math.max(1, maxSignal - minSignal);
  const peakIntensities = peakValues.map((v) => (v - minSignal) / range);
  const avgIntensity = peakIntensities.reduce((a, b) => a + b, 0) / peaks.length;

  // ROI Coverage Score (Improved):
  // Measures how much of the ROI is actually spanned by the peak sequence.
  const span = peaks[peaks.length - 1] - peaks[0];
  const coverageFit = Math.min(1, span / (regionHeight * 0.82));

  return (consistencyScore * 0.45 + avgIntensity * 0.35 + coverageFit * 0.2);
}

function estimateDominantSpacing(signal: number[], regionHeight: number) {
  if (signal.length < 24) return null;

  const centered = signal.map((value) => value - mean(signal));
  const minLag = Math.max(4, Math.round(regionHeight / 150));
  const maxLag = Math.min(120, Math.max(minLag + 10, Math.round(regionHeight / 8)));
  const lags: Array<{ lag: number; score: number }> = [];

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
    // Use a very light lag penalty to allow detecting larger spacings on high-res images
    const score = corr - lag * 0.0015;
    lags.push({ lag, score });
  }

  if (lags.length === 0) return null;

  const scoreAtLag = (targetLag: number, tolerance: number = 2) => {
    const nearest = lags
      .filter((entry) => Math.abs(entry.lag - targetLag) <= tolerance)
      .sort((a, b) => b.score - a.score)[0];
    return nearest?.score ?? 0;
  };

  // Find local peaks with higher sensitivity
  const peaks: typeof lags = [];
  for (let i = 1; i < lags.length - 1; i++) {
    if (lags[i].score > lags[i - 1].score && lags[i].score > lags[i + 1].score && lags[i].score > 0.02) {
      peaks.push(lags[i]);
    }
  }

  if (peaks.length === 0) {
    const absoluteMax = [...lags].sort((a, b) => b.score - a.score)[0];
    return absoluteMax.score > 0.1 ? absoluteMax.lag : null;
  }

  const sortedPeaks = [...peaks].sort((a, b) => b.score - a.score);
  const globalMax = sortedPeaks[0];
  const theoreticalMaxCount = regionHeight / 6;
  const earliestSignificantPeak = [...peaks]
    .filter((peak) => peak.score >= globalMax.score * 0.2)
    .sort((a, b) => a.lag - b.lag)[0];

  if (earliestSignificantPeak) {
    const earliestEstimatedCount = regionHeight / Math.max(1, earliestSignificantPeak.lag);
    const plausibleEarliest = earliestEstimatedCount <= theoreticalMaxCount * 1.5;
    const harmonicLockedToEarliestDouble =
      globalMax.lag >= earliestSignificantPeak.lag * 1.85 &&
      globalMax.lag <= earliestSignificantPeak.lag * 2.2;
    const harmonicLockedToEarliestTriple =
      globalMax.lag >= earliestSignificantPeak.lag * 2.7 &&
      globalMax.lag <= earliestSignificantPeak.lag * 3.25;
    const significantEarlierAlternative =
      globalMax.lag >= earliestSignificantPeak.lag * 1.6 &&
      earliestSignificantPeak.score >= globalMax.score * 0.35;

    if (
      plausibleEarliest &&
      (harmonicLockedToEarliestDouble || harmonicLockedToEarliestTriple || significantEarlierAlternative)
    ) {
      return earliestSignificantPeak.lag;
    }
  }

  const peakCandidates = peaks
    .map((peak) => {
      const estimatedCount = regionHeight / Math.max(1, peak.lag);
      const implausibleDensityPenalty =
        estimatedCount > theoreticalMaxCount * 1.5
          ? clamp((estimatedCount / Math.max(1, theoreticalMaxCount)) - 1.5, 0, 1.2)
          : 0;
      const secondHarmonic = scoreAtLag(peak.lag * 2, 2);
      const thirdHarmonic = scoreAtLag(peak.lag * 3, 3);
      const subHarmonic = peak.lag >= minLag * 2 ? scoreAtLag(peak.lag / 2, 1) : 0;
      const harmonicSupport =
        peak.score +
        secondHarmonic * 0.65 +
        thirdHarmonic * 0.35 +
        subHarmonic * 0.12 -
        implausibleDensityPenalty * 0.18;

      return {
        ...peak,
        estimatedCount,
        harmonicSupport,
        secondHarmonic,
        thirdHarmonic,
        subHarmonic
      };
    })
    .filter((candidate) => candidate.score >= globalMax.score * 0.08)
    .sort((a, b) => {
      const supportGap = b.harmonicSupport - a.harmonicSupport;
      if (Math.abs(supportGap) > 1e-6) return supportGap;
      if (a.lag !== b.lag) return a.lag - b.lag;
      return b.score - a.score;
    });

  const bestSupportedPeak = peakCandidates[0];
  const earliestSupportedPeak = peakCandidates
    .filter((candidate) => candidate.harmonicSupport >= bestSupportedPeak.harmonicSupport * 0.78)
    .sort((a, b) => a.lag - b.lag)[0];

  if (!earliestSupportedPeak) {
    return globalMax.lag;
  }

  const harmonicLockedToDouble =
    globalMax.lag >= earliestSupportedPeak.lag * 1.7 &&
    globalMax.lag <= earliestSupportedPeak.lag * 2.3 &&
    earliestSupportedPeak.secondHarmonic >= earliestSupportedPeak.score * 0.7;

  const harmonicLockedToTriple =
    globalMax.lag >= earliestSupportedPeak.lag * 2.6 &&
    globalMax.lag <= earliestSupportedPeak.lag * 3.4 &&
    earliestSupportedPeak.thirdHarmonic >= earliestSupportedPeak.score * 0.45;

  if (harmonicLockedToDouble || harmonicLockedToTriple) {
    return earliestSupportedPeak.lag;
  }

  if (globalMax.lag > 0 && regionHeight / globalMax.lag > theoreticalMaxCount * 1.5) {
    return earliestSupportedPeak.lag;
  }

  return bestSupportedPeak.harmonicSupport >= globalMax.score * 1.04
    ? bestSupportedPeak.lag
    : globalMax.lag;
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

function densifyPeaks(
  signal: number[],
  peaks: number[],
  spacing: number,
  gapMultiplier: number = 1.35,
  minStrengthQuantile: number = 0.75
) {
  if (peaks.length < 2) return peaks;

  const searchRadius = Math.max(2, spacing * 0.4);
  const minCandidateStrength = percentile(signal, minStrengthQuantile);
  const dense: number[] = [];

  // 1. Check Top Edge Gap (before the first peak)
  const firstPeak = peaks[0];
  if (firstPeak > spacing * 0.75) {
    const target = firstPeak - spacing;
    const candidate = findLocalPeakNear(signal, target, searchRadius);
    if (signal[candidate] >= minCandidateStrength && candidate < firstPeak - Math.max(2, spacing * 0.5)) {
      dense.push(candidate);
    }
  }

  // If no top-edge peak found, still need to start with the first original peak
  if (dense.length === 0 || dense[dense.length - 1] !== peaks[0]) {
    if (dense.length === 0 || peaks[0] - dense[dense.length - 1] >= Math.max(2, spacing * 0.4)) {
      dense.push(peaks[0]);
    }
  }

  // 2. Original Middle Gap Filling
  for (let i = 1; i < peaks.length; i++) {
    const previous = dense[dense.length - 1];
    const current = peaks[i];
    const gap = current - previous;

    if (gap > spacing * gapMultiplier && gap < spacing * 4.5) {
      const steps = Math.max(1, Math.round(gap / spacing) - 1);
      for (let step = 1; step <= steps; step++) {
        const target = previous + (gap * step) / (steps + 1);
        const candidate = findLocalPeakNear(signal, target, searchRadius);
        if (
          signal[candidate] >= minCandidateStrength &&
          candidate - dense[dense.length - 1] >= Math.max(2, spacing * 0.5)
        ) {
          dense.push(candidate);
        }
      }
    }

    if (current - dense[dense.length - 1] >= Math.max(2, spacing * 0.5)) {
      dense.push(current);
    }
  }

  // 3. Check Bottom Edge Gap (after the last peak)
  const lastPeak = dense[dense.length - 1];
  const remainingGap = (signal.length - 1) - lastPeak;
  if (remainingGap > spacing * 0.75) {
     const target = lastPeak + spacing;
     const candidate = findLocalPeakNear(signal, target, searchRadius);
     if (signal[candidate] >= minCandidateStrength && candidate > lastPeak + Math.max(2, spacing * 0.5)) {
       dense.push(candidate);
     }
  }

  return dense;
}

function estimateBaseSpacingFromEmpirical(spacings: number[], currentSpacing: number) {
  if (spacings.length < 6 || currentSpacing <= 0) return null;

  const positiveSpacings = spacings.filter((spacing) => spacing > 0).sort((a, b) => a - b);
  if (positiveSpacings.length < 6) return null;

  const medianSpacing = percentile(positiveSpacings, 0.5);
  const lowerQuantile = percentile(positiveSpacings, 0.28);
  const lowerCluster = positiveSpacings.filter(
    (spacing) => spacing >= lowerQuantile * 0.82 && spacing <= lowerQuantile * 1.22
  );

  if (lowerCluster.length < Math.max(3, Math.floor(positiveSpacings.length * 0.22))) {
    return null;
  }

  const lowerMean = mean(lowerCluster);
  const harmonicFromMedian =
    medianSpacing >= lowerMean * 1.55 &&
    medianSpacing <= lowerMean * 2.45;
  const harmonicFromCurrent =
    currentSpacing >= lowerMean * 1.5 &&
    currentSpacing <= lowerMean * 2.5;

  if (!harmonicFromMedian && !harmonicFromCurrent) {
    return null;
  }

  return lowerMean;
}

function detectDensePeaks(signal: number[], regionHeight: number, distance: number, prominence: number) {
  let spacing = estimateDominantSpacing(signal, regionHeight);
  
  // 1. Empirical Anchor: In dense stacks, trust real peaks over math models
  const initialDistance = Math.max(2, Math.round((spacing || 12) * 0.5));
  const initialPeaks = findPeaks(signal, initialDistance, 3);
  let finalSpacing = spacing || 12;
  
  if (initialPeaks.length >= 8) {
    const empiricalSpacings = initialPeaks.slice(1).map((val, idx) => val - initialPeaks[idx]);
    const empiricalMedian = percentile(empiricalSpacings, 0.5);
    const empiricalBaseSpacing = estimateBaseSpacingFromEmpirical(empiricalSpacings, spacing || empiricalMedian);
    
    // If we have enough empirical evidence, trust it. 
    // Autocorrelation often locks onto 2T (double spacing).
    if (empiricalMedian > 0) {
      if (!spacing || Math.abs(empiricalMedian - spacing) > spacing * 0.25 || initialPeaks.length > 15) {
        finalSpacing = empiricalMedian;
      }
    }

    if (empiricalBaseSpacing && empiricalBaseSpacing < finalSpacing * 0.82) {
      finalSpacing = empiricalBaseSpacing;
    }
  }

  // 2. Base detection with refined spacing
  const relaxedDistance = Math.max(3, Math.round(finalSpacing * 0.75));
  const relaxedProminence = Math.max(2, Math.min(prominence, 4));
  let peaks = findPeaks(signal, relaxedDistance, relaxedProminence);
  peaks = peaks.filter((peak) => peak >= Math.floor(signal.length * 0.005) && peak <= Math.floor(signal.length * 0.995));
  
  // 3. Pruning: Remove noise before densification
  peaks = mergeClosePeaks(peaks, 0.25);

  // 4. Exact Count Protection
  const theoreticalMax = regionHeight / finalSpacing;
  const alreadyFull = peaks.length >= theoreticalMax * 0.95;
  
  if (!alreadyFull) {
    // Only densify if there's significant room and we use a much higher strength threshold
    peaks = densifyPeaks(signal, peaks, finalSpacing, 1.6);
  }
  
  // 5. Final Physical cleanup: ensure minimum physical gap matches physical constraints
  const finalPeaks: number[] = [];
  const minPhysicalGap = Math.max(3, Math.round(finalSpacing * 0.82));
  for (const p of peaks) {
    if (finalPeaks.length === 0 || p - finalPeaks[finalPeaks.length - 1] >= minPhysicalGap) {
      finalPeaks.push(p);
    }
  }
  
  return {
    peaks: filterBySpacing(finalPeaks, signal),
    spacing: finalSpacing
  };
}

function detectPeaks(signal: number[], regionHeight: number, distance: number, prominence: number) {
  // 1. Light Signal Pre-smoothing (2-point to preserve density)
  const smoothed = new Array(signal.length);
  for (let i = 0; i < signal.length; i++) {
    const prev = signal[Math.max(0, i - 1)];
    smoothed[i] = (prev + signal[i]) / 2;
  }

  // 2. Adaptive Scaling with Floor Protection
  const baseScale = regionHeight / 1920;
  const rawAdaptiveDistance = distance * Math.pow(baseScale, 0.5);
  const adaptiveDistance = Math.max(2, Math.min(distance, Math.round(rawAdaptiveDistance)));

  // Refinement: Dynamic floor for prominence based on signal intensity
  const signalMax = Math.max(...signal);
  const signalMin = Math.min(...signal);
  const signalRange = Math.max(1, signalMax - signalMin);
  const noiseFloor = signalRange * 0.04; // Increased from 3% to 4%
  const adaptiveProminence = Math.max(noiseFloor, Math.round(prominence * 1.1 * Math.pow(baseScale, 0.4)));

  const mergeRatio = 0.22;

  let peaks = findPeaks(smoothed, adaptiveDistance, adaptiveProminence);
  
  // 3. Edge Effect Suppression (Clipping)
  const topClip = Math.floor(signal.length * 0.005);
  const bottomClip = Math.floor(signal.length * 0.998);
  peaks = peaks.filter((peak) => peak >= topClip && peak <= bottomClip);
  peaks = mergeClosePeaks(peaks, mergeRatio);
  peaks = filterBySpacing(peaks, signal);

  return peaks;
}

function enforceMinimumPeakGap(signal: number[], peaks: number[], spacing: number, ratio: number) {
  if (peaks.length < 2 || spacing <= 0) return peaks;

  const minGap = Math.max(3, Math.round(spacing * ratio));
  const ranked = [...peaks]
    .map((peak) => ({ peak, strength: signal[peak] ?? 0 }))
    .sort((a, b) => b.strength - a.strength);

  const kept: number[] = [];
  for (const candidate of ranked) {
    if (kept.every((peak) => Math.abs(peak - candidate.peak) >= minGap)) {
      kept.push(candidate.peak);
    }
  }

  return kept.sort((a, b) => a - b);
}

function estimateUpperSpacing(spacings: number[]) {
  if (spacings.length === 0) return 0;

  const upperQuantile = percentile(spacings, 0.72);
  const trimmed = spacings.filter(
    (spacing) => spacing >= upperQuantile * 0.82 && spacing <= upperQuantile * 1.45
  );

  return trimmed.length > 0 ? mean(trimmed) : upperQuantile;
}

function detectCoarsePeaks(signal: number[], peaks: number[], regionHeight: number) {
  if (peaks.length < 6 || peaks.length > 18) return null;

  const spacings = peaks.slice(1).map((value, index) => value - peaks[index]);
  const medianSpacing = spacings.length > 0 ? percentile(spacings, 0.5) : 0;
  const maxSpacing = spacings.length > 0 ? Math.max(...spacings) : 0;
  if (medianSpacing < 12 || maxSpacing < medianSpacing * 1.75) return null;

  const coarseSignal = normalizeSignal(
    smoothSignal(signal, Math.max(5, Math.round(medianSpacing * 1.15)))
  );
  const coarseDistance = Math.max(9, Math.round(medianSpacing * 1.45));
  let coarsePeaks = findPeaks(coarseSignal, coarseDistance, 3);
  coarsePeaks = mergeClosePeaks(coarsePeaks, 0.48);
  coarsePeaks = filterBySpacing(coarsePeaks, coarseSignal);

  if (coarsePeaks.length < 2 || coarsePeaks.length >= peaks.length) return null;

  return {
    peaks: coarsePeaks,
    projection: coarseSignal,
    qualityScore: scorePeaks(coarseSignal, coarsePeaks, regionHeight)
  };
}

function detectMacroPeaks(signal: number[], peaks: number[], regionHeight: number) {
  if (peaks.length < 40 || peaks.length > 80) return null;

  const spacings = peaks.slice(1).map((value, index) => value - peaks[index]);
  const medianSpacing = spacings.length > 0 ? percentile(spacings, 0.5) : 0;
  if (medianSpacing < 9) return null;

  const macroSignal = normalizeSignal(
    smoothSignal(signal, Math.max(4, Math.round(medianSpacing * 0.72)))
  );
  const macroDistance = Math.max(10, Math.round(medianSpacing * 1.4));
  let macroPeaks = findPeaks(macroSignal, macroDistance, 3);
  macroPeaks = mergeClosePeaks(macroPeaks, 0.52);
  macroPeaks = filterBySpacing(macroPeaks, macroSignal);

  if (macroPeaks.length < 18 || macroPeaks.length >= Math.round(peaks.length * 0.72)) return null;

  return {
    peaks: macroPeaks,
    projection: macroSignal,
    qualityScore: scorePeaks(macroSignal, macroPeaks, regionHeight)
  };
}

function countPeaksInRegion(
  brightnessSignal: number[],
  edgeSignal: number[],
  continuitySignal: number[],
  roiHeight: number,
  distance: number,
  prominence: number,
  strictSpacing: boolean = false,
  intensity: 'stable' | 'aggressive' = 'stable'
) {
  let brightnessPeaks = detectPeaks(brightnessSignal, roiHeight, distance, prominence, intensity);
  let edgePeaks = detectPeaks(edgeSignal, roiHeight, distance, prominence, intensity);

  // Filter peaks by spatial continuity to suppress noise
  // Real layers should have at least some horizontal connectivity
  brightnessPeaks = brightnessPeaks.filter(p => continuitySignal[p] > 0.18);
  edgePeaks = edgePeaks.filter(p => continuitySignal[p] > 0.18);

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
    // Also filter dense peaks
    denseResult.peaks = denseResult.peaks.filter(p => continuitySignal[p] > 0.18);

    const denseScore = scorePeaks(denseSignal, denseResult.peaks, roiHeight);
    const denseCountThreshold = Math.max(peaks.length + 1, Math.round(peaks.length * 1.1));
    const spacingThreshold = denseResult.spacing <= Math.max(24, roiHeight / 20);
    const potentialCount = Math.floor(roiHeight / Math.max(1, denseResult.spacing));
    const denseRecallMode = (roiHeight >= 200 && peaks.length < roiHeight / 11) || 
                           (peaks.length < potentialCount * 0.85);
    const denseUpperBound = denseResult.peaks.length <= roiHeight / 7.5;

    if (
      (denseRecallMode && spacingThreshold && denseUpperBound && denseResult.peaks.length >= denseCountThreshold) ||
      (denseResult.peaks.length > peaks.length && denseScore > score * 1.1)
    ) {
      peaks = denseResult.peaks;
      projection = denseSignal;
      score = denseScore;
    }
  }

  const currentSpacings = peaks.slice(1).map((value, index) => value - peaks[index]);
  const empiricalSpacing = currentSpacings.length >= 2 ? percentile(currentSpacings, 0.5) : 0;
  const spacingReference = denseResult?.spacing ?? empiricalSpacing;

  if (strictSpacing && spacingReference > 0 && peaks.length >= 8) {
    peaks = enforceMinimumPeakGap(projection, peaks, spacingReference, 0.68);
  }

  const spatialConfidence = peaks.length > 0 
    ? peaks.reduce((sum, p) => sum + (continuitySignal[p] ?? 0), 0) / peaks.length 
    : 0;

  return {
    peaks,
    projection,
    qualityScore: score * (0.3 + 0.7 * spatialConfidence),
    spatialConfidence,
    spacing: spacingReference,
    brightnessPeakCount: brightnessPeaks.length,
    edgePeakCount: edgePeaks.length
  };
}

function findSplitBoundaries(peaks: number[], regionHeight: number) {
  if (peaks.length < 12) return [];

  const spacings = peaks.slice(1).map((value, index) => value - peaks[index]);
  const medianSpacing = percentile(spacings, 0.5);
  if (medianSpacing <= 0) return [];

  const minSegmentHeight = Math.max(90, Math.round(regionHeight * 0.18));
  const boundaries: number[] = [];

  for (let i = 0; i < spacings.length; i++) {
    const gap = spacings[i];
    const midpoint = Math.round((peaks[i] + peaks[i+1]) / 2);
    
    const pronouncedGap = gap >= Math.max(medianSpacing * 1.8, medianSpacing + 10);
    const validHeight = midpoint >= minSegmentHeight && (regionHeight - midpoint) >= minSegmentHeight;

    if (pronouncedGap && validHeight) {
      boundaries.push(midpoint);
    }
  }

  return boundaries.slice(0, 2);
}

function findSpacingShiftBoundaries(signal: number[], regionHeight: number) {
  if (signal.length < 120) return [];

  const windowSize = Math.max(80, Math.min(Math.round(regionHeight * 0.24), 220));
  const step = Math.max(24, Math.round(windowSize / 3));
  const samples: Array<{ center: number; spacing: number }> = [];

  for (let start = 0; start + windowSize <= signal.length; start += step) {
    const end = start + windowSize;
    const spacing = estimateDominantSpacing(signal.slice(start, end), windowSize);
    if (spacing) {
      samples.push({
        center: Math.round((start + end) / 2),
        spacing
      });
    }
  }

  if (samples.length < 4) return [];

  const minSegmentHeight = Math.max(90, Math.round(regionHeight * 0.18));
  const boundaries: number[] = [];

  for (let i = 1; i < samples.length; i++) {
    const prev = samples[i - 1];
    const next = samples[i];
    const ratio = Math.max(prev.spacing, next.spacing) / Math.max(1, Math.min(prev.spacing, next.spacing));
    const boundary = Math.round((prev.center + next.center) / 2);

    if (
      ratio >= 1.35 &&
      boundary >= minSegmentHeight &&
      regionHeight - boundary >= minSegmentHeight
    ) {
      boundaries.push(boundary);
    }
  }

  return boundaries.slice(0, 2);
}

function isBoundaryRed(r: number, g: number, b: number) {
  const maxGB = Math.max(g, b);
  const brightness = (r + g + b) / 3;
  const chromaGap = r - maxGB;
  return (
    r >= 150 &&
    maxGB <= 120 &&
    r > maxGB * 1.6 &&
    chromaGap >= 60 &&
    brightness <= 210
  );
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

  // A red line might not span the entire width, but should be prominent.
  // Require at least 8% coverage or a continuous run of 4% of the width.
  const minCoverage = Math.max(8, Math.floor(width * 0.08));
  const minRun = Math.max(8, Math.floor(width * 0.04));
  
  const candidateRows: number[] = [];
  for (let y = 0; y < height; y++) {
    if (rowHits[y] >= minCoverage && longestRuns[y] >= minRun) {
      candidateRows.push(y);
    }
  }

  if (candidateRows.length < 2) return null;

  const groups: Array<{ start: number; end: number; center: number; maxHits: number }> = [];
  let groupStart = candidateRows[0];
  let groupEnd = candidateRows[0];
  let maxHitsInGroup = rowHits[candidateRows[0]];

  for (let i = 1; i < candidateRows.length; i++) {
    const row = candidateRows[i];
    if (row <= groupEnd + 12) { // Slightly wider gap tolerance
      groupEnd = row;
      maxHitsInGroup = Math.max(maxHitsInGroup, rowHits[row]);
      continue;
    }

    groups.push({
      start: groupStart,
      end: groupEnd,
      center: Math.round((groupStart + groupEnd) / 2),
      maxHits: maxHitsInGroup
    });
    
    groupStart = row;
    groupEnd = row;
    maxHitsInGroup = rowHits[row];
  }
  groups.push({
    start: groupStart,
    end: groupEnd,
    center: Math.round((groupStart + groupEnd) / 2),
    maxHits: maxHitsInGroup
  });

  // Filter groups by requiring a solid horizontal presence
  const validGroups = groups
    .filter((group) => group.maxHits >= width * 0.12 && (group.end - group.start) <= Math.max(25, height * 0.035))
    .sort((a, b) => b.maxHits - a.maxHits);

  if (validGroups.length < 2) return null;

  // Optimized scoring: Prioritize centered, strong pairs with reasonable gaps
  const candidates = validGroups.slice(0, 8);
  let bestPair: [typeof candidates[number], typeof candidates[number]] | null = null;
  let bestScore = -Infinity;

  const vCenter = height / 2;

  for (let i = 0; i < candidates.length; i++) {
    for (let j = i + 1; j < candidates.length; j++) {
      const top = candidates[i].center < candidates[j].center ? candidates[i] : candidates[j];
      const bottom = top === candidates[i] ? candidates[j] : candidates[i];
      const gap = bottom.center - top.center;
      
      // Sanity checks for a stack: Gap must be reasonable
      if (gap < Math.max(30, height * 0.1)) continue;
      if (gap > height * 0.95) continue;

      // Centrality: Pairs closer to the middle of the image are preferred
      const pairCenter = (top.center + bottom.center) / 2;
      const centralityPenalty = Math.abs(pairCenter - vCenter) / vCenter;
      
      // Bonus for being "inside" the typical ROI [0.05, 0.95]
      const roiBonus = (top.center > height * 0.05 && bottom.center < height * 0.95) ? 30 : 0;

      // New scoring formula: balance prominence, gap size (capped), and centrality
      const score = (top.maxHits + bottom.maxHits) * 0.5 + 
                    Math.min(gap, height * 0.7) * 0.15 - 
                    centralityPenalty * 80 + 
                    roiBonus;

      if (score > bestScore) {
        bestScore = score;
        bestPair = [top, bottom];
      }
    }
  }

  if (!bestPair) return null;

  const top = bestPair[0].center;
  const bottom = bestPair[1].center;
  
  if (bottom - top < Math.max(20, height * 0.05)) return null;

  return { y1: top, y2: bottom };
}

function detectStackBoundariesByTexture(gray: Float32Array, width: number, height: number) {
  const rowGrads = new Float32Array(height);
  const sampleStep = Math.max(1, Math.floor(width / 100)); // Sample 100 points per row for speed

  for (let y = 1; y < height - 1; y++) {
    let sum = 0;
    let count = 0;
    for (let x = 10; x < width - 10; x += sampleStep) {
      const g = Math.abs(gray[y * width + x] - gray[(y - 1) * width + x]);
      sum += g;
      count++;
    }
    rowGrads[y] = sum / count;
  }

  // Smooth the gradient signal
  const windowSize = Math.max(5, Math.floor(height * 0.04));
  const smoothed = new Float32Array(height);
  for (let i = 0; i < height; i++) {
    let sum = 0;
    let count = 0;
    for (let j = Math.max(0, i - windowSize); j <= Math.min(height - 1, i + windowSize); j++) {
      sum += rowGrads[j];
      count++;
    }
    smoothed[i] = sum / count;
  }

  const avgGrad = mean(Array.from(smoothed));
  // Relaxed threshold from 1.35 to 1.15 to catch weaker textures
  const threshold = avgGrad * 1.15;
  
  let y1 = -1;
  let y2 = -1;

  // Expanded search range from [0.05, 0.95] to [0.02, 0.98]
  for (let y = Math.floor(height * 0.02); y < height * 0.98; y++) {
    if (smoothed[y] > threshold) {
      if (y1 === -1) y1 = y;
      y2 = y;
    }
  }

  // Reduced minimum height requirement from 10% to 5%
  if (y1 !== -1 && y2 !== -1 && (y2 - y1) > height * 0.05) {
    return {
      y1: Math.max(0, y1 - Math.floor(height * 0.01)),
      y2: Math.min(height - 1, y2 + Math.floor(height * 0.01)),
      source: 'texture' as const
    };
  }

  return null;
}

function detectTextureOuterBoundaries(
  gray: Float32Array,
  width: number,
  height: number,
  anchorY1: number,
  anchorY2: number
) {
  const wideX1 = Math.max(0, Math.floor(width * 0.12));
  const wideX2 = Math.min(width, Math.ceil(width * 0.88));
  const wideStep = Math.max(1, Math.floor((wideX2 - wideX1) / 220));
  const envelopeSignal = new Array<number>(height).fill(0);

  for (let y = 1; y < height - 1; y++) {
    const samples: number[] = [];
    let gradSum = 0;
    let gradCount = 0;

    for (let x = wideX1; x < wideX2; x += wideStep) {
      const value = gray[y * width + x];
      samples.push(value);
      gradSum += Math.abs(gray[(y + 1) * width + x] - gray[(y - 1) * width + x]);
      gradCount += 1;
    }

    const rowStd = std(samples);
    const rowGrad = gradCount > 0 ? gradSum / gradCount : 0;
    envelopeSignal[y] = rowStd * 0.65 + rowGrad * 0.35;
  }

  const smoothedEnvelope = smoothSignal(envelopeSignal, Math.max(6, Math.floor(height / 160)));
  const envelopeThreshold = Math.max(
    percentile(smoothedEnvelope, 0.68),
    mean(smoothedEnvelope) * 1.05
  );
  const minRunLength = Math.max(8, Math.floor(height / 140));
  const envelopeRuns: Array<{ start: number; end: number }> = [];
  let runStart = -1;

  for (let y = 0; y < smoothedEnvelope.length; y++) {
    if (smoothedEnvelope[y] >= envelopeThreshold) {
      if (runStart === -1) runStart = y;
      continue;
    }

    if (runStart !== -1 && y - runStart >= minRunLength) {
      envelopeRuns.push({ start: runStart, end: y - 1 });
    }
    runStart = -1;
  }

  if (runStart !== -1 && smoothedEnvelope.length - runStart >= minRunLength) {
    envelopeRuns.push({ start: runStart, end: smoothedEnvelope.length - 1 });
  }

  if (envelopeRuns.length === 0) return null;

  const searchMin = Math.max(
    Math.floor(height * 0.015),
    anchorY1 - Math.floor(height * 0.08)
  );
  const searchMax = Math.min(
    height - 1,
    anchorY2 + Math.floor(height * 0.20)
  );

  const selectedRuns = envelopeRuns.filter(
    (run) => run.end >= searchMin && run.start <= searchMax
  );

  if (selectedRuns.length === 0) return null;

  return {
    y1: Math.max(0, selectedRuns[0].start - Math.floor(height * 0.006)),
    y2: Math.min(height - 1, selectedRuns[selectedRuns.length - 1].end + Math.floor(height * 0.006))
  };
}

function meanInRange(values: ArrayLike<number>, start: number, end: number) {
  const safeStart = Math.max(0, Math.floor(start));
  const safeEnd = Math.min(values.length, Math.ceil(end));
  if (safeEnd <= safeStart) return 0;

  let sum = 0;
  for (let i = safeStart; i < safeEnd; i++) {
    sum += values[i];
  }
  return sum / Math.max(1, safeEnd - safeStart);
}

function buildActiveRuns(signal: number[], minRunLength: number, threshold: number) {
  const runs: Array<{ start: number; end: number }> = [];
  let runStart = -1;

  for (let index = 0; index < signal.length; index++) {
    if (signal[index] >= threshold) {
      if (runStart === -1) runStart = index;
      continue;
    }

    if (runStart !== -1 && index - runStart >= minRunLength) {
      runs.push({ start: runStart, end: index - 1 });
    }
    runStart = -1;
  }

  if (runStart !== -1 && signal.length - runStart >= minRunLength) {
    runs.push({ start: runStart, end: signal.length - 1 });
  }

  return runs;
}

function validateTextureBoundaries(
  gray: Float32Array,
  width: number,
  height: number,
  candidateY1: number,
  candidateY2: number,
  fallbackY1: number,
  fallbackY2: number
) {
  const x1 = Math.max(1, Math.floor(width * 0.14));
  const x2 = Math.min(width - 1, Math.ceil(width * 0.86));
  const sampleStep = Math.max(1, Math.floor((x2 - x1) / 180));
  const rowTexture = new Array<number>(height).fill(0);
  const rowGradient = new Array<number>(height).fill(0);

  for (let y = 1; y < height - 1; y++) {
    const samples: number[] = [];
    let gradientSum = 0;
    let gradientCount = 0;

    for (let x = x1; x < x2; x += sampleStep) {
      const value = gray[y * width + x];
      samples.push(value);
      gradientSum += Math.abs(gray[(y + 1) * width + x] - gray[(y - 1) * width + x]);
      gradientCount += 1;
    }

    rowTexture[y] = samples.length > 1 ? std(samples) : 0;
    rowGradient[y] = gradientCount > 0 ? gradientSum / gradientCount : 0;
  }

  const smoothedTexture = smoothSignal(rowTexture, Math.max(6, Math.floor(height / 180)));
  const smoothedGradient = smoothSignal(rowGradient, Math.max(6, Math.floor(height / 180)));
  const gradientThreshold = Math.max(
    percentile(smoothedGradient, 0.72),
    mean(smoothedGradient) * 1.08
  );

  const rowDensity = new Array<number>(height).fill(0);
  for (let y = 1; y < height - 1; y++) {
    let denseCount = 0;
    let totalCount = 0;

    for (let x = x1; x < x2; x += sampleStep) {
      const gradient = Math.abs(gray[(y + 1) * width + x] - gray[(y - 1) * width + x]);
      if (gradient >= gradientThreshold) {
        denseCount += 1;
      }
      totalCount += 1;
    }

    rowDensity[y] = totalCount > 0 ? denseCount / totalCount : 0;
  }

  const smoothedDensity = smoothSignal(rowDensity, Math.max(6, Math.floor(height / 180)));
  const normalizedTexture = normalizeSignal(smoothedTexture);
  const normalizedGradient = normalizeSignal(smoothedGradient);
  const normalizedDensity = normalizeSignal(smoothedDensity);
  const activitySignal = normalizedTexture.map((value, index) =>
    (value / 255) * 0.4 + (normalizedGradient[index] / 255) * 0.35 + (normalizedDensity[index] / 255) * 0.25
  );

  const activityThreshold = Math.max(
    percentile(activitySignal, 0.64),
    mean(activitySignal) * 1.04
  );
  const activeRuns = buildActiveRuns(activitySignal, Math.max(10, Math.floor(height / 160)), activityThreshold);
  const candidateMargin = Math.max(12, Math.floor(height * 0.03));
  const overlappedRuns = activeRuns.filter(
    (run) => run.end >= candidateY1 - candidateMargin && run.start <= candidateY2 + candidateMargin
  );

  let adjustedY1 = candidateY1;
  let adjustedY2 = candidateY2;
  if (overlappedRuns.length > 0) {
    adjustedY1 = Math.max(0, overlappedRuns[0].start - Math.max(4, Math.floor(height * 0.008)));
    adjustedY2 = Math.min(height - 1, overlappedRuns[overlappedRuns.length - 1].end + Math.max(4, Math.floor(height * 0.008)));
  }

  const candidateCoverage = (candidateY2 - candidateY1) / Math.max(1, height);
  const adjustedCoverage = (adjustedY2 - adjustedY1) / Math.max(1, height);

  const outsideWindow = Math.max(18, Math.floor(height * 0.05));
  const insideTexture = meanInRange(smoothedTexture, adjustedY1, adjustedY2);
  const outsideTexture = (
    meanInRange(smoothedTexture, adjustedY1 - outsideWindow, adjustedY1) +
    meanInRange(smoothedTexture, adjustedY2, adjustedY2 + outsideWindow)
  ) / 2;
  const insideGradient = meanInRange(smoothedGradient, adjustedY1, adjustedY2);
  const outsideGradient = (
    meanInRange(smoothedGradient, adjustedY1 - outsideWindow, adjustedY1) +
    meanInRange(smoothedGradient, adjustedY2, adjustedY2 + outsideWindow)
  ) / 2;
  const insideDensity = meanInRange(smoothedDensity, adjustedY1, adjustedY2);
  const outsideDensity = (
    meanInRange(smoothedDensity, adjustedY1 - outsideWindow, adjustedY1) +
    meanInRange(smoothedDensity, adjustedY2, adjustedY2 + outsideWindow)
  ) / 2;

  const activeCoverage = meanInRange(activitySignal, adjustedY1, adjustedY2);
  const textureContrast = insideTexture / Math.max(1, outsideTexture);
  const gradientContrast = insideGradient / Math.max(1, outsideGradient);
  const densityContrast = insideDensity / Math.max(0.02, outsideDensity + 0.02);
  const textureScore =
    clamp((textureContrast - 1) / 0.55, 0, 1) * 0.65 +
    clamp((activeCoverage - 0.38) / 0.26, 0, 1) * 0.35;
  const gradientScore = clamp((gradientContrast - 1) / 0.6, 0, 1);
  const densityScore =
    clamp((densityContrast - 1) / 0.85, 0, 1) * 0.6 +
    clamp((insideDensity - 0.12) / 0.18, 0, 1) * 0.4;

  const candidateCoverageScore =
    candidateCoverage >= 0.18 && candidateCoverage <= 0.82
      ? 1
      : clamp(1 - Math.abs(candidateCoverage - 0.5) * 2.3, 0.15, 1);
  const adjustedCoverageScore =
    adjustedCoverage >= 0.16 && adjustedCoverage <= 0.78
      ? 1
      : clamp(1 - Math.abs(adjustedCoverage - 0.47) * 2.6, 0.1, 1);
  const confidence =
    (textureScore * 0.4 + gradientScore * 0.35 + densityScore * 0.25) *
    Math.min(candidateCoverageScore, adjustedCoverageScore);

  const accepted =
    confidence >= 0.65 &&
    activeCoverage >= 0.28 &&
    adjustedCoverage >= 0.10 &&
    adjustedCoverage <= 0.95;

  return {
    accepted,
    y1: accepted ? adjustedY1 : fallbackY1,
    y2: accepted ? adjustedY2 : fallbackY2,
    candidateY1,
    candidateY2,
    adjustedY1,
    adjustedY2,
    confidence,
    textureScore,
    gradientScore,
    densityScore,
    candidateCoverage,
    adjustedCoverage,
    activeCoverage,
    activeRuns: activeRuns.length
  };
}


// ============================================================
// STAGE 1: Hough Line Transform Deskew + Center Line Annotation
// ============================================================

/**
 * Compute Sobel gradients (Gx, Gy) and magnitude for a grayscale image.
 */
function computeGradients(
  gray: Float32Array,
  width: number,
  height: number
): { gx: Float32Array; gy: Float32Array; magnitude: Float32Array } {
  const gx = new Float32Array(width * height);
  const gy = new Float32Array(width * height);
  const magnitude = new Float32Array(width * height);

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const idx = y * width + x;
      const dx =
        -gray[(y - 1) * width + (x - 1)] + gray[(y - 1) * width + (x + 1)] +
        -2 * gray[y * width + (x - 1)] + 2 * gray[y * width + (x + 1)] +
        -gray[(y + 1) * width + (x - 1)] + gray[(y + 1) * width + (x + 1)];
      const dy =
        -gray[(y - 1) * width + (x - 1)] - 2 * gray[(y - 1) * width + x] - gray[(y - 1) * width + (x + 1)] +
        gray[(y + 1) * width + (x - 1)] + 2 * gray[(y + 1) * width + x] + gray[(y + 1) * width + (x + 1)];
      gx[idx] = dx;
      gy[idx] = dy;
      magnitude[idx] = Math.sqrt(dx * dx + dy * dy);
    }
  }
  return { gx, gy, magnitude };
}

function localStretch(signal: number[], windowSize: number): number[] {
  if (signal.length === 0) return [];
  const result = new Array(signal.length);
  const halfWindow = Math.floor(windowSize / 2);

  for (let i = 0; i < signal.length; i++) {
    let min = Infinity;
    let max = -Infinity;
    const start = Math.max(0, i - halfWindow);
    const end = Math.min(signal.length - 1, i + halfWindow);

    for (let j = start; j <= end; j++) {
      if (signal[j] < min) min = signal[j];
      if (signal[j] > max) max = signal[j];
    }

    const range = max - min;
    if (range > 0) {
      result[i] = (signal[i] - min) / range;
    } else {
      result[i] = 0.5;
    }
  }
  return result;
}

function extractSingleBandSignals(
  gray: Float32Array,
  width: number,
  y1: number,
  y2: number,
  centerRatio: number = 0.5,
  intensity: 'stable' | 'aggressive' = 'stable'
) : BandSignalBundle {
  let roiHeight = Math.max(0, y2 - y1);
  const centerX = Math.round(width * clamp(centerRatio, 0.1, 0.9));
  const bandHalfW = Math.max(6, Math.round(width * 0.04));
  const bx1 = Math.max(1, centerX - bandHalfW);
  const bx2 = Math.min(width - 1, centerX + bandHalfW);
  const bandW = Math.max(1, bx2 - bx1);

  const brightnessRaw = new Array<number>(roiHeight).fill(0);
  const edgeRaw = new Array<number>(roiHeight).fill(0);
  const continuityRaw = new Array<number>(roiHeight).fill(0);

  // Find local gradient threshold for continuity
  let totalGrad = 0;
  let gradCount = 0;
  for (let y = y1 + 1; y < y2 - 1; y++) {
    for (let x = bx1; x < bx2; x++) {
      totalGrad += Math.abs(gray[(y + 1) * width + x] - gray[(y - 1) * width + x]);
      gradCount++;
    }
  }
  const avgGrad = totalGrad / Math.max(1, gradCount);
  const continuityThreshold = avgGrad * 0.6;

  for (let y = y1 + 1; y < y2 - 1; y++) {
    let bSum = 0;
    let eSum = 0;
    let maxRun = 0; // track the longest continuous run of edge pixels
    let currentRun = 0;
    
    for (let x = bx1; x < bx2; x++) {
      const c = gray[y * width + x];
      bSum += 255 - c;
      const vGrad = Math.abs(gray[(y + 1) * width + x] - gray[(y - 1) * width + x]);
      const lGrad = Math.abs(c - gray[y * width + (x - 1)]);
      const rGrad = Math.abs(c - gray[y * width + (x + 1)]);
      eSum += vGrad * 0.8 + (lGrad + rGrad) * 0.2;
      
      // A pixel is continuous if it has strong vertical gradient AND weak horizontal gradient (it's part of a line)
      if (vGrad > continuityThreshold && (lGrad + rGrad) < vGrad * 0.6) {
        currentRun++;
        if (currentRun > maxRun) maxRun = currentRun;
      } else {
        currentRun = 0;
      }
    }
    brightnessRaw[y - y1] = bSum / bandW;
    edgeRaw[y - y1] = eSum / bandW;
    // Measure continuity by the longest run relative to band width (capped at 60% to allow some gaps)
    continuityRaw[y - y1] = Math.min(1.0, maxRun / (bandW * 0.6));
  }

  const smoothRadius = Math.max(2, Math.round(roiHeight / 160));
  
  // Phase 3.5: Adaptive stretching intensity
  const stretchWindow = intensity === 'aggressive' 
    ? Math.max(20, Math.round(roiHeight / 45))
    : Math.max(30, Math.round(roiHeight / 30));

  const brightnessStretched = localStretch(smoothSignal(brightnessRaw, smoothRadius), stretchWindow);
  const edgeStretched = localStretch(smoothSignal(edgeRaw, smoothRadius), stretchWindow);

  // Apply non-linear dampening
  const power = intensity === 'aggressive' ? 1.0 : 1.15;
  const dampen = (s: number[]) => s.map(v => Math.pow(v, power));
  
  const brightnessSignal = normalizeSignal(dampen(brightnessStretched));
  const edgeSignal = normalizeSignal(dampen(edgeStretched));
  const continuitySignal = smoothSignal(continuityRaw, smoothRadius);

  return {
    ratio: centerRatio,
    centerX,
    brightnessSignal,
    edgeSignal,
    continuitySignal,
    roiCoverage: bandW / Math.max(1, width)
  };
}

function buildParallelBandRatios(centerRatio: number, width: number) {
  const offset = width >= 1200 ? 0.06 : 0.05;
  const outerOffset = width >= 1200 ? 0.12 : 0.1;
  const ratios = [
    clamp(centerRatio - outerOffset, 0.18, 0.82),
    clamp(centerRatio - offset, 0.15, 0.85),
    clamp(centerRatio, 0.12, 0.88),
    clamp(centerRatio + offset, 0.15, 0.85),
    clamp(centerRatio + outerOffset, 0.18, 0.82)
  ];

  return ratios.filter((ratio, index, values) => values.findIndex((value) => Math.abs(value - ratio) < 1e-6) === index);
}

function extractMultiBandSignals(
  gray: Float32Array,
  width: number,
  y1: number,
  y2: number,
  bandRatios: number[]
) {
  const bandSignals = bandRatios.map((ratio) => extractSingleBandSignals(gray, width, y1, y2, ratio));
  if (bandSignals.length === 0) {
    const fallbackBand = extractSingleBandSignals(gray, width, y1, y2, 0.5);
    return {
      centerX: fallbackBand.centerX,
      brightnessSignal: fallbackBand.brightnessSignal,
      edgeSignal: fallbackBand.edgeSignal,
      continuitySignal: fallbackBand.continuitySignal,
      bandSignals: [fallbackBand],
      signalExtractionMode: 'multi-band-parallel' as const
    };
  }

  const centerBand = bandSignals[Math.floor(bandSignals.length / 2)];
  const brightnessSignal = centerBand.brightnessSignal.map((_, index) =>
    percentile(
      bandSignals
        .map((band) => band.brightnessSignal[index] ?? 0)
        .filter((value) => Number.isFinite(value)),
      0.5
    )
  );
  const edgeSignal = centerBand.edgeSignal.map((_, index) =>
    percentile(
      bandSignals
        .map((band) => band.edgeSignal[index] ?? 0)
        .filter((value) => Number.isFinite(value)),
      0.65
    )
  );
  const continuitySignal = centerBand.continuitySignal.map((_, index) =>
    percentile(
      bandSignals
        .map((band) => band.continuitySignal[index] ?? 0)
        .filter((value) => Number.isFinite(value)),
      0.5
    )
  );

  return {
    centerX: centerBand.centerX,
    brightnessSignal,
    edgeSignal,
    continuitySignal,
    bandSignals,
    signalExtractionMode: 'multi-band-parallel' as const
  };
}

function extractCenterBandSignals(
  gray: Float32Array,
  width: number,
  y1: number,
  y2: number,
  centerRatio: number = 0.5,
  signalExtractionMode: SignalExtractionMode = 'single-band',
  parallelBandRatios?: number[],
  intensity: 'stable' | 'aggressive' = 'stable'
) {
  if (signalExtractionMode === 'multi-band-parallel') {
    const ratios = parallelBandRatios && parallelBandRatios.length > 0
      ? parallelBandRatios
      : buildParallelBandRatios(centerRatio, width);
    return extractMultiBandSignals(gray, width, y1, y2, ratios);
  }

  const band = extractSingleBandSignals(gray, width, y1, y2, centerRatio, intensity);
  return {
    centerX: band.centerX,
    brightnessSignal: band.brightnessSignal,
    edgeSignal: band.edgeSignal,
    continuitySignal: band.continuitySignal,
    bandSignals: [band],
    signalExtractionMode: 'single-band' as const
  };
}

function analyzeBandSignals(
  brightnessSignal: number[],
  edgeSignal: number[],
  continuitySignal: number[],
  roiHeight: number,
  distance: number,
  prominence: number,
  strictSpacing: boolean,
  intensity: 'stable' | 'aggressive' = 'stable'
) {
  const baseResult = countPeaksInRegion(
    brightnessSignal, 
    edgeSignal, 
    continuitySignal,
    roiHeight, 
    distance, 
    prominence, 
    strictSpacing, 
    intensity
  );
  return baseResult;
}

function selectTextureBandResult(
  gray: Float32Array,
  width: number,
  y1: number,
  y2: number,
  distance: number,
  prominence: number,
  strictSpacing: boolean,
  bandRatios: number[] = [0.22, 0.28, 0.35, 0.42, 0.48, 0.52, 0.58, 0.65, 0.72, 0.78, 0.85],
  fusionStrategy: 'single' | 'median' | 'clustering' = 'clustering'
) {
  let roiHeight = Math.max(0, y2 - y1);
  const candidateBands = bandRatios.map((ratio) => {
    // Pass 1: Stable Intensity
    const bandStable = extractCenterBandSignals(
      gray,
      width,
      y1,
      y2,
      ratio,
      'single-band',
      undefined,
      'stable'
    );
    const analysisStable = analyzeBandSignals(
      bandStable.brightnessSignal,
      bandStable.edgeSignal,
      bandStable.continuitySignal,
      roiHeight,
      distance,
      prominence,
      strictSpacing,
      'stable'
    );

    // CRITICAL: Only escalate to aggressive if the stable pass is TRULY failing
    // (e.g. count is way too low for a large ROI, or quality is non-existent)
    const looksAliased = analysisStable.peaks.length < 15 && roiHeight > 600;
    const isGarbage = analysisStable.qualityScore < 0.35;

    if (looksAliased || isGarbage) {
      const bandAggressive = extractCenterBandSignals(
        gray,
        width,
        y1,
        y2,
        ratio,
        'single-band',
        undefined,
        'aggressive'
      );
      const analysisAggressive = analyzeBandSignals(
        bandAggressive.brightnessSignal,
        bandAggressive.edgeSignal,
        bandAggressive.continuitySignal,
        roiHeight,
        distance,
        prominence,
        strictSpacing,
        'aggressive'
      );

      // Only trust aggressive if it restores a high-density pattern (count > 45)
      // AND has significantly better relative quality than the garbage stable pass
      if (analysisAggressive.peaks.length > 45 && analysisAggressive.qualityScore > 0.42) {
        const spacings = analysisAggressive.peaks.slice(1).map((value, index) => value - analysisAggressive.peaks[index]);
        const spacing = spacings.length > 0 ? percentile(spacings, 0.5) : undefined;
        return {
          ratio,
          centerX: bandAggressive.centerX,
          peaks: analysisAggressive.peaks,
          projection: analysisAggressive.projection,
          qualityScore: analysisAggressive.qualityScore,
          spatialConfidence: analysisAggressive.spatialConfidence,
          count: analysisAggressive.peaks.length,
          spacing,
          intensity: 'aggressive' as const
        };
      }
    }

    const spacings = analysisStable.peaks.slice(1).map((value, index) => value - analysisStable.peaks[index]);
    const spacing = spacings.length > 0 ? percentile(spacings, 0.5) : undefined;
    return {
      ratio,
      centerX: bandStable.centerX,
      peaks: analysisStable.peaks,
      projection: analysisStable.projection,
      qualityScore: analysisStable.qualityScore,
      spatialConfidence: analysisStable.spatialConfidence,
      count: analysisStable.peaks.length,
      spacing,
      intensity: 'stable' as const
    };
  });

  // Phase 3.2: Frequency Domain Guidance (Autocorrelation)
  const centerBandIdx = Math.floor(candidateBands.length / 2);
  const priorSpacing = estimateDominantSpacing(
    candidateBands[centerBandIdx].projection, 
    roiHeight
  );
  const priorCount = priorSpacing ? (roiHeight / priorSpacing) : undefined;

  const candidateSummary = candidateBands.map((band) => ({
    ratio: band.ratio,
    centerX: band.centerX,
    count: band.peaks.length,
    qualityScore: band.qualityScore,
    spatialConfidence: band.spatialConfidence,
    spacing: band.spacing
  }));

  if (fusionStrategy === 'clustering') {
    const clusters: Array<{ counts: number[], bands: typeof candidateBands }> = [];
    
    for (const band of candidateBands) {
      let addedToCluster = false;
      for (const cluster of clusters) {
        const clusterMean = cluster.counts.reduce((sum, c) => sum + c, 0) / cluster.counts.length;
        if (Math.abs(band.count - clusterMean) / Math.max(1, clusterMean) <= 0.12) {
          cluster.counts.push(band.count);
          cluster.bands.push(band);
          addedToCluster = true;
          break;
        }
      }
      if (!addedToCluster) {
        clusters.push({ counts: [band.count], bands: [band] });
      }
    }
    
    // Phase 3.6: Harmonic-Aware Strategy Gating
    const highDensityCluster = clusters
      .filter(c => {
        const meanCount = c.counts.reduce((s, x) => s + x, 0) / c.counts.length;
        const avgQual = c.bands.reduce((s, b) => s + b.qualityScore, 0) / c.bands.length;
        const avgCont = c.bands.reduce((s, b) => s + (b.spatialConfidence ?? 0), 0) / c.bands.length;
        
        if (meanCount <= 42) return false;

        const matchesPrior = priorCount && Math.abs(meanCount - priorCount) / priorCount <= 0.12;
        
        // GATING: High density MUST have either a strong prior match 
        // OR very high quality AND decent 2D spatial continuity.
        if (matchesPrior) {
          return c.counts.length >= 2 && avgCont > 0.50;
        }
        return c.counts.length >= 4 && avgQual > 0.70 && avgCont > 0.65;
      })
      .sort((a, b) => {
        const aMean = a.counts.reduce((s, x) => s + x, 0) / a.counts.length;
        const bMean = b.counts.reduce((s, x) => s + x, 0) / b.counts.length;
        return bMean - aMean;
      })[0];

    const rankedClusters = clusters.sort((a, b) => {
      const aMean = a.counts.reduce((s, c) => s + c, 0) / a.counts.length;
      const bMean = b.counts.reduce((s, c) => s + c, 0) / b.counts.length;
      const aQual = a.bands.reduce((s, b) => s + b.qualityScore, 0) / a.bands.length;
      const bQual = b.bands.reduce((s, b) => s + b.qualityScore, 0) / b.bands.length;
      const aCont = a.bands.reduce((s, b) => s + (b.spatialConfidence ?? 0), 0) / a.bands.length;
      const bCont = b.bands.reduce((s, b) => s + (b.spatialConfidence ?? 0), 0) / b.bands.length;

      let aPriorBonus = 0;
      let bPriorBonus = 0;
      if (priorCount) {
        if (Math.abs(aMean - priorCount) / priorCount <= 0.15) aPriorBonus = 15;
        if (Math.abs(bMean - priorCount) / priorCount <= 0.15) bPriorBonus = 15;
      }

      const isAHigh = highDensityCluster && Math.abs(aMean - (highDensityCluster.counts.reduce((s,x)=>s+x,0)/highDensityCluster.counts.length)) < 5;
      const isBHigh = highDensityCluster && Math.abs(bMean - (highDensityCluster.counts.reduce((s,x)=>s+x,0)/highDensityCluster.counts.length)) < 5;
      const aHighDensityWeight = isAHigh ? 15 : 0;
      const bHighDensityWeight = isBHigh ? 15 : 0;

      const aDensityBonus = a.counts.length > 1 ? (aMean / 5.0) : 0;
      const bDensityBonus = b.counts.length > 1 ? (bMean / 5.0) : 0;

      // Penalize low continuity clusters, especially those claiming high density
      // Use non-linear penalty to heavily suppress low-continuity noise
      const aContFactor = aMean > 35 ? Math.pow(aCont, 2.0) : aCont;
      const bContFactor = bMean > 35 ? Math.pow(bCont, 2.0) : bCont;

      const aScore = (a.counts.length * 6.0 + aQual * 15.0 + aDensityBonus + aPriorBonus + aHighDensityWeight) * aContFactor;
      const bScore = (b.counts.length * 6.0 + bQual * 15.0 + bDensityBonus + bPriorBonus + bHighDensityWeight) * bContFactor;

      return bScore - aScore;
    });

    const winningCluster = rankedClusters[0];
    const centroidCount = Math.round(
      winningCluster.counts.reduce((s, c) => s + c, 0) / winningCluster.counts.length
    );
    
    const winningBand = winningCluster.bands.sort((a, b) => {
      const aDiff = Math.abs(a.count - centroidCount);
      const bDiff = Math.abs(b.count - centroidCount);
      if (aDiff !== bDiff) return aDiff - bDiff;
      return b.qualityScore - a.qualityScore;
    })[0];

    return {
      ...winningBand,
      selectedRatio: winningBand.ratio,
      candidateBands: candidateSummary,
      consensusCount: centroidCount,
      fusionStrategy: 'clustering' as const
    };
  }

  const sortedByCount = [...candidateBands].sort((a, b) => a.count - b.count);
  const medianBand = sortedByCount[Math.floor(sortedByCount.length / 2)];
  
  if (fusionStrategy === 'median') {
    return {
      ...medianBand,
      selectedRatio: medianBand.ratio,
      candidateBands: candidateSummary,
      consensusCount: undefined,
      fusionStrategy: 'median' as const
    };
  }

  return {
    ...medianBand,
    selectedRatio: medianBand.ratio,
    candidateBands: candidateSummary,
    consensusCount: undefined,
    fusionStrategy: 'single' as const
  };
}

function computeAxisDominance(
  gray: Float32Array,
  width: number,
  height: number,
  y1: number,
  y2: number
) {
  const x1 = Math.max(1, Math.floor(width * 0.2));
  const x2 = Math.min(width - 1, Math.ceil(width * 0.8));
  const cy1 = Math.max(1, y1);
  const cy2 = Math.min(height - 1, y2);
  const sampleStep = Math.max(1, Math.floor(Math.min(width, height) / 240));

  let rowGrad = 0;
  let rowCount = 0;
  for (let y = cy1; y < cy2; y += sampleStep) {
    for (let x = x1; x < x2; x += sampleStep) {
      rowGrad += Math.abs(gray[y * width + x] - gray[(y - 1) * width + x]);
      rowCount += 1;
    }
  }

  let colGrad = 0;
  let colCount = 0;
  for (let y = cy1; y < cy2; y += sampleStep) {
    for (let x = x1; x < x2; x += sampleStep) {
      colGrad += Math.abs(gray[y * width + x] - gray[y * width + (x - 1)]);
      colCount += 1;
    }
  }

  const avgRowGrad = rowCount > 0 ? rowGrad / rowCount : 0;
  const avgColGrad = colCount > 0 ? colGrad / colCount : 0;
  return (avgRowGrad + 1) / (avgColGrad + 1);
}

function evaluateQuarterTurnCandidate(
  imageData: ImageData,
  roiRatio: [number, number],
  distance: number,
  prominence: number
) {
  const { width, height } = imageData;
  const gray = toGrayscale(imageData);
  const textureBoundaries = detectStackBoundariesByTexture(gray, width, height);

  let y1 = Math.floor(height * roiRatio[0]);
  let y2 = Math.floor(height * roiRatio[1]);
  let boundarySource: 'texture' | 'ratio' = 'ratio';

  if (textureBoundaries) {
    y1 = textureBoundaries.y1;
    y2 = textureBoundaries.y2;
    boundarySource = 'texture';
  }

  y1 = Math.max(0, Math.min(y1, height - 1));
  y2 = Math.max(y1 + 1, Math.min(y2, height));

  let roiHeight = Math.max(0, y2 - y1);
  if (roiHeight < 40) {
    return {
      confidence: 0,
      axisRatio: 1,
      peakScore: 0,
      count: 0,
      roiCoverage: roiHeight / Math.max(1, height),
      boundarySource
    };
  }

  const { brightnessSignal, edgeSignal } = extractCenterBandSignals(gray, width, y1, y2);
  const peakResult = countPeaksInRegion(brightnessSignal, edgeSignal, roiHeight, distance, prominence, false);
  const peakScore = scorePeaks(peakResult.projection, peakResult.peaks, roiHeight);
  const axisRatio = computeAxisDominance(gray, width, height, y1, y2);
  const roiCoverage = roiHeight / Math.max(1, height);
  const coverageScore = roiCoverage >= 0.18 && roiCoverage <= 0.9 ? 1 : Math.max(0, 1 - Math.abs(roiCoverage - 0.54) * 2.2);
  const countScore = Math.min(1, peakResult.peaks.length / Math.max(8, roiHeight / 14));
  const axisScore = clamp((axisRatio - 0.7) / 0.8, 0, 1);
  const confidence =
    peakScore * 0.5 +
    axisScore * 0.28 +
    coverageScore * 0.12 +
    countScore * 0.1 +
    (boundarySource === 'texture' ? 0.04 : 0);

  return {
    confidence,
    axisRatio,
    peakScore,
    count: peakResult.peaks.length,
    roiCoverage,
    boundarySource
  };
}

function detectQuarterTurnRotation(
  imageData: ImageData,
  roiRatio: [number, number],
  distance: number,
  prominence: number
) {
  const upright = evaluateQuarterTurnCandidate(imageData, roiRatio, distance, prominence);
  const rotatedImageData = rotateImageData(imageData, 90);
  const quarterTurn = evaluateQuarterTurnCandidate(rotatedImageData, roiRatio, distance, prominence);

  const confidenceGain = quarterTurn.confidence - upright.confidence;
  const axisGain = quarterTurn.axisRatio - upright.axisRatio;
  const countGain = quarterTurn.count - upright.count;
  const shouldRotate =
    upright.axisRatio <= 0.95 &&
    quarterTurn.axisRatio >= 1.18 &&
    confidenceGain >= 0.18 &&
    axisGain >= 0.28 &&
    (quarterTurn.peakScore >= upright.peakScore + 0.08 || countGain >= 8);

  return {
    shouldRotate,
    quarterTurnAngle: shouldRotate ? 90 : 0,
    rotatedImageData: shouldRotate ? rotatedImageData : imageData
  };
}

/**
 * Hough Line Transform (angle-only accumulation).
 * Detects the dominant tilt angle of horizontal structures in the image.
 * Returns angle in degrees (positive = clockwise tilt), range ±30°.
 */
function houghDetectAngle(gray: Float32Array, width: number, height: number): number {
  // Sub-sample for performance — process every `step`-th pixel
  const step = Math.max(1, Math.floor(Math.min(width, height) / 500));
  const { magnitude, gx, gy } = computeGradients(gray, width, height);

  // Collect magnitude samples for thresholding
  const samples: number[] = [];
  for (let y = step; y < height - step; y += step) {
    for (let x = step; x < width - step; x += step) {
      samples.push(magnitude[y * width + x]);
    }
  }
  const threshold = percentile(samples, 0.82);

  // Angle histogram for near-horizontal lines (±angleRange degrees from horizontal)
  // A horizontal line's edge gradient is perpendicular → near ±90°
  // lineAngle = gradAngle − 90°, normalized to −90°..+90°
  const angleRange = 30; // only consider lines within ±30° of horizontal
  const numBins = 600;   // sub-degree precision
  const histogram = new Float32Array(numBins);

  for (let y = step; y < height - step; y += step) {
    for (let x = step; x < width - step; x += step) {
      const idx = y * width + x;
      if (magnitude[idx] < threshold) continue;

      const gradAngleDeg = Math.atan2(gy[idx], gx[idx]) * (180 / Math.PI);
      let lineAngle = gradAngleDeg - 90;
      // Normalize to −90°..+90°
      while (lineAngle > 90) lineAngle -= 180;
      while (lineAngle < -90) lineAngle += 180;

      if (Math.abs(lineAngle) > angleRange) continue;

      const bin = Math.round(((lineAngle + angleRange) / (2 * angleRange)) * (numBins - 1));
      const clampedBin = Math.max(0, Math.min(numBins - 1, bin));
      histogram[clampedBin] += magnitude[idx];
    }
  }

  // Gaussian-smooth the histogram to suppress noise
  const smoothed = new Float32Array(numBins);
  const smR = Math.round(numBins * 0.01);
  for (let b = 0; b < numBins; b++) {
    let sum = 0, w = 0;
    for (let k = Math.max(0, b - smR); k <= Math.min(numBins - 1, b + smR); k++) {
      const weight = Math.exp(-0.5 * ((k - b) / Math.max(1, smR * 0.4)) ** 2);
      sum += histogram[k] * weight;
      w += weight;
    }
    smoothed[b] = sum / w;
  }

  // Find bin with maximum votes
  let peakBin = Math.floor(numBins / 2); // center = 0°
  let peakVal = 0;
  for (let b = 0; b < numBins; b++) {
    if (smoothed[b] > peakVal) {
      peakVal = smoothed[b];
      peakBin = b;
    }
  }

  const dominantAngle = (peakBin / (numBins - 1)) * (2 * angleRange) - angleRange;
  return dominantAngle;
}

/**
 * Rotate ImageData around its center by `angleDeg` degrees.
 * The output canvas is expanded to prevent clipping.
 */
function rotateImageData(imageData: ImageData, angleDeg: number): ImageData {
  const { width, height } = imageData;
  const rad = (angleDeg * Math.PI) / 180;
  const cos = Math.abs(Math.cos(rad));
  const sin = Math.abs(Math.sin(rad));

  // New size to contain the full rotated image
  const newWidth = Math.round(width * cos + height * sin);
  const newHeight = Math.round(height * cos + width * sin);

  // Source canvas
  const srcCanvas = document.createElement('canvas');
  srcCanvas.width = width;
  srcCanvas.height = height;
  const srcCtx = srcCanvas.getContext('2d')!;
  srcCtx.putImageData(imageData, 0, 0);

  // Destination canvas
  const dstCanvas = document.createElement('canvas');
  dstCanvas.width = newWidth;
  dstCanvas.height = newHeight;
  const dstCtx = dstCanvas.getContext('2d')!;

  // Fill with mid-gray so border pixels don't corrupt brightness/edge projections
  dstCtx.fillStyle = 'rgb(128,128,128)';
  dstCtx.fillRect(0, 0, newWidth, newHeight);
  dstCtx.translate(newWidth / 2, newHeight / 2);
  dstCtx.rotate(rad);
  dstCtx.drawImage(srcCanvas, -width / 2, -height / 2);

  return dstCtx.getImageData(0, 0, newWidth, newHeight);
}

/**
 * Draw a bold red vertical reference line through the horizontal center of the image.
 * This line is used to count intersections with horizontal layer edges.
 */
function drawVerticalCenterLine(imageData: ImageData): ImageData {
  const { width, height } = imageData;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d')!;
  ctx.putImageData(imageData, 0, 0);

  const centerX = Math.round(width / 2);
  const lineWidth = Math.max(2, Math.round(width / 240));

  ctx.save();
  ctx.strokeStyle = '#ff0000';
  ctx.lineWidth = lineWidth;
  ctx.setLineDash([]);
  ctx.globalAlpha = 0.92;
  ctx.beginPath();
  ctx.moveTo(centerX, 0);
  ctx.lineTo(centerX, height);
  ctx.stroke();
  ctx.restore();

  return ctx.getImageData(0, 0, width, height);
}

/**
 * Preprocessing stage parameters
 * [0.12, 0.94]: Safe Vision Boundaries.
 * 0.12 (top): Avoids UI status bars, button shadows, or hair/fingers at the top of the frame.
 * 0.94 (bottom): Avoids lens distortion (spherical aberration) and obstruction at the very bottom.
 */
export function preprocessImageData(imageData: ImageData): {
  correctedImageData: ImageData;
  deskewAngle: number;
  quarterTurnAngle: number;
  safeRoiRatio: [number, number];
} {
  const quarterTurnDecision = detectQuarterTurnRotation(imageData, [0.12, 0.94], 4, 6);
  const orientationCorrectedImageData = quarterTurnDecision.rotatedImageData;
  const { width, height } = orientationCorrectedImageData;
  const gray = toGrayscale(orientationCorrectedImageData);

  // Detect tilt angle via Hough Line Transform
  const rawAngle = houghDetectAngle(gray, width, height);

  // Only apply rotation if tilt is visually significant
  const deskewAngle = Math.abs(rawAngle) >= 0.5 ? rawAngle : 0;
  let correctedImageData = orientationCorrectedImageData;
  let safeRoiRatio: [number, number] = [0.12, 0.94];

  if (deskewAngle !== 0) {
    // Rotate by the negative of the detected angle to level the image
    correctedImageData = rotateImageData(orientationCorrectedImageData, -deskewAngle);

    // After rotation the canvas grows, compute the largest inner rectangle
    // (the visible content area) to avoid sampling the padded border.
    const rad = Math.abs(deskewAngle * Math.PI / 180);
    const cos = Math.cos(rad);
    const sin = Math.sin(rad);
    const newW = correctedImageData.width;
    const newH = correctedImageData.height;

    // The original image fits inside the rotated canvas; the content rectangle
    // centre is at (newW/2, newH/2). Compute the half-height of content visible
    // without black corners: halfH = (h*cos - w*sin) / 2  (when landscape, adapt)
    const halfHContent = Math.max(
      0,
      (height * cos - width * sin) / 2
    );
    // Express as fraction of new height
    const marginRatio = halfHContent > 0 ? (newH / 2 - halfHContent) / newH : 0.06;
    const safeStart = Math.max(0.05, marginRatio + 0.02);
    const safeEnd   = Math.min(0.95, 1 - marginRatio - 0.02);
    safeRoiRatio = [safeStart, safeEnd];
  }

  // Draw vertical center reference line on the corrected image
  correctedImageData = drawVerticalCenterLine(correctedImageData);

  return {
    correctedImageData,
    deskewAngle,
    quarterTurnAngle: quarterTurnDecision.quarterTurnAngle,
    safeRoiRatio
  };
}

function analyzeImageLocalAtResolution(
  imageData: ImageDataLike,
  roiRatio: [number, number] = [0.12, 0.94],
  distance: number = 4,
  prominence: number = 6,
  maxResolution: number = 1920
) {
  const { resizedImageData, actualResolution } = resizeImageData(imageData, maxResolution);

  // ── Stage 1: Hough deskew + center line annotation ─────────────────────────
  const { correctedImageData, deskewAngle, quarterTurnAngle, safeRoiRatio } = preprocessImageData(resizedImageData as ImageData);

  // When the image was rotated, use the computed safe inner ROI to avoid
  // sampling the gray-padded border regions.
  const effectiveRoiRatio: [number, number] = deskewAngle !== 0
    ? [Math.max(roiRatio[0], safeRoiRatio[0]), Math.min(roiRatio[1], safeRoiRatio[1])]
    : roiRatio;

  const { width, height } = correctedImageData;
  const gray = toGrayscale(correctedImageData);

  // ── Stage 2: 1D projection counting on the corrected image ──────────────────
  const boundaryLines = quarterTurnAngle !== 0 ? null : detectRedBoundaryLines(correctedImageData);
  let y1 = Math.floor(height * effectiveRoiRatio[0]);
  let y2 = Math.floor(height * effectiveRoiRatio[1]);
  let boundaryY1 = y1;
  let boundaryY2 = y2;
  let boundarySource: 'red_lines' | 'texture' | 'ratio' = 'ratio';
  let roiValidation: ReturnType<typeof validateTextureBoundaries> | null = null;

  if (boundaryLines) {
    y1 = boundaryLines.y1;
    y2 = boundaryLines.y2;
    boundaryY1 = boundaryLines.y1;
    boundaryY2 = boundaryLines.y2;
    boundarySource = 'red_lines';
  } else {
    // Fallback to texture-based detection
    const textureBoundaries = detectStackBoundariesByTexture(gray, width, height);
    if (textureBoundaries) {
      const outerBoundaries = detectTextureOuterBoundaries(gray, width, height, textureBoundaries.y1, textureBoundaries.y2);
      const validatedTextureBoundaries = validateTextureBoundaries(
        gray,
        width,
        height,
        outerBoundaries?.y1 ?? textureBoundaries.y1,
        outerBoundaries?.y2 ?? textureBoundaries.y2,
        y1,
        y2
      );
      roiValidation = validatedTextureBoundaries;

      if (validatedTextureBoundaries.accepted) {
        y1 = validatedTextureBoundaries.y1;
        y2 = validatedTextureBoundaries.y2;
        boundaryY1 = validatedTextureBoundaries.adjustedY1;
        boundaryY2 = validatedTextureBoundaries.adjustedY2;
        boundarySource = 'texture';
      }
    }
  }

  // After rotation, the image may be slightly larger — keep y1/y2 in bounds
  y1 = Math.max(0, Math.min(y1, height - 1));
  y2 = Math.max(y1 + 1, Math.min(y2, height));
  boundaryY1 = Math.max(0, Math.min(boundaryY1, height - 1));
  boundaryY2 = Math.max(boundaryY1 + 1, Math.min(boundaryY2, height));

  let roiHeight = Math.max(0, y2 - y1);

  // ── Center-column vertical intersection signal ───────────────────────────────
  // After Hough deskew the image is level. We extract brightness + vertical-edge
  // signals along a narrow band centred on x = width/2 (the red vertical line).
  // Each peak in these signals is a y-coordinate where the centre line crosses
  // a horizontal layer boundary — i.e. one "intersection point".
  const strictSpacing = Boolean(boundaryLines);
  let centerX = Math.round(width / 2);
  let peaks: number[] = [];
  let projection: number[] = [];
  let qualityScore = 0;
  let selectedBandRatio = 0.5;
  let candidateBands: Array<{ ratio: number; centerX: number; count: number; qualityScore: number }> = [];
  let finalCountOverride: number | null = null;

  if (
    (boundarySource === 'texture' && roiHeight >= 180) ||
    (boundarySource === 'ratio' && roiHeight >= 1300)
  ) {
    const bandRatios = boundarySource === 'ratio'
      ? [0.22, 0.28, 0.35, 0.42, 0.48, 0.52, 0.58, 0.65, 0.72, 0.78, 0.85]
      : [0.25, 0.35, 0.45, 0.5, 0.55, 0.65, 0.75];
    const selectedBand = selectTextureBandResult(
      gray,
      width,
      y1,
      y2,
      distance,
      prominence,
      strictSpacing,
      bandRatios,
      'clustering'
    );

    centerX = selectedBand.centerX;
    peaks = selectedBand.peaks;
    projection = selectedBand.projection;
    qualityScore = selectedBand.qualityScore;
    selectedBandRatio = selectedBand.selectedRatio;
    candidateBands = selectedBand.candidateBands;
    finalCountOverride = selectedBand.consensusCount ?? null;

    const spacings = peaks.slice(1).map((value, index) => value - peaks[index]);
    const medianSpacing = spacings.length > 0 ? percentile(spacings, 0.5) : 0;
    const firstGap = peaks.length > 0 ? peaks[0] : 0;
    const lastGap = peaks.length > 0 ? Math.max(0, roiHeight - peaks[peaks.length - 1]) : 0;

    if (
      qualityScore >= 0.72 &&
      peaks.length >= 12 &&
      medianSpacing >= 6 &&
      (
        firstGap >= medianSpacing * 2.3 ||
        lastGap >= medianSpacing * 2.3
      )
    ) {
      const extendTop = Math.min(
        Math.round(height * 0.04),
        firstGap >= medianSpacing * 2.3
          ? Math.max(Math.round(medianSpacing), Math.round(firstGap - medianSpacing * 0.35))
          : 0
      );
      const extendBottom = Math.min(
        Math.round(height * 0.04),
        lastGap >= medianSpacing * 2.3
          ? Math.max(Math.round(medianSpacing), Math.round(lastGap - medianSpacing * 0.35))
          : 0
      );

      const expandedY1 = Math.max(0, y1 - extendTop);
      const expandedY2 = Math.min(height, y2 + extendBottom);
      const expandedRoiHeight = Math.max(0, expandedY2 - expandedY1);
      if (expandedRoiHeight > roiHeight + Math.round(medianSpacing * 0.8)) {
        const expandedBand = selectTextureBandResult(
          gray,
          width,
          expandedY1,
          expandedY2,
          distance,
          prominence,
          strictSpacing,
          [0.25, 0.35, 0.45, 0.5, 0.55, 0.65, 0.75],
          'clustering'
        );

        if (
          Math.abs(expandedBand.peaks.length - peaks.length) <= 10 &&
          expandedBand.qualityScore >= qualityScore - 0.08
        ) {
          y1 = expandedY1;
          y2 = expandedY2;
          roiHeight = expandedRoiHeight;
          centerX = expandedBand.centerX;
          peaks = expandedBand.peaks;
          projection = expandedBand.projection;
          qualityScore = expandedBand.qualityScore;
        }
      }
    }
  } else {
    const band = extractCenterBandSignals(gray, width, y1, y2);
    const analysis = analyzeBandSignals(
      band.brightnessSignal,
      band.edgeSignal,
      roiHeight,
      distance,
      prominence,
      strictSpacing
    );
    centerX = band.centerX;
    peaks = analysis.peaks;
    projection = analysis.projection;
    qualityScore = analysis.qualityScore;
    candidateBands = [{
      ratio: 0.5,
      centerX: band.centerX,
      count: analysis.peaks.length,
      qualityScore: analysis.qualityScore
    }];
  }

  const absolutePeaks = peaks.map((peak) => peak + y1);
  const finalSpacings = absolutePeaks.slice(1).map((value, index) => value - absolutePeaks[index]);
  const spacing = finalSpacings.length > 0 ? percentile(finalSpacings, 0.5) : undefined;

  // Intersection points: exact (x, y) coordinates on the corrected image where
  // the vertical red centre line crosses each horizontal layer boundary.
  const intersectionPoints = absolutePeaks.map((y) => ({ x: centerX, y }));

  return {
    count: finalCountOverride ?? absolutePeaks.length,
    peaks: absolutePeaks,
    intersectionPoints,
    projection,
    y1,
    y2,
    boundaryY1,
    boundaryY2,
    height,
    width,
    centerX,
    boundarySource,
    qualityScore,
    spacing,
    quarterTurnAngle,
    selectedBandRatio,
    candidateBands,
    actualResolution,
    roiValidation,
    /** Tilt angle (degrees) detected and corrected in Stage 1. 0 = no rotation applied. */
    deskewAngle,
    /** The preprocessed ImageData: deskewed + vertical centre line drawn. */
    correctedImageData
  };
}

/**
 * Helper to calculate coefficient of variation for spacing
 */
function calculateSpacingCV(peaks: number[]): number {
  if (peaks.length < 3) return 0;
  const spacings = peaks.slice(1).map((p, i) => p - peaks[i]);
  const mean = spacings.reduce((a, b) => a + b, 0) / spacings.length;
  const variance = spacings.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / spacings.length;
  return Math.sqrt(variance) / mean;
}

/**
 * Generic refinement logic to replace brittle hard-coded patches.
 * Uses spacing consistency (CV) and quality metrics instead of specific counts.
 */
function refineResult<T extends ReturnType<typeof analyzeImageLocalAtResolution>>(
  result: T, 
  imageData: ImageDataLike, 
  roiRatio: [number, number], 
  distance: number, 
  prominence: number
): T {
  const cv = calculateSpacingCV(result.peaks);
  const roiHeight = result.y2 - result.y1;
  const theoreticalCount = roiHeight / (result.spacing || 1);
  const densityError = Math.abs(result.count - theoreticalCount) / theoreticalCount;

  // Pattern: Low confidence high-frequency noise removal
  if (cv > 0.45 && result.qualityScore < 0.75) {
    const spacings = result.peaks.slice(1).map((p, i) => p - result.peaks[i]);
    const medianSpacing = percentile(spacings, 0.5);
    const correctedCount = Math.round(roiHeight / medianSpacing);
    if (correctedCount < result.count * 0.85) {
      return { ...result, count: correctedCount };
    }
  }

  // Phase 2: Multi-Resolution Voting (Task 2.3)
  // If we suspect a high-layer stack, we validate frequencies across two scales.
  const isSuspicious = result.count > 15 && (result.qualityScore < 0.88 || (result.spacing || 100) < 22);

  if (isSuspicious && result.actualResolution < 2400) {
    const hiResResult = analyzeImageLocalAtResolution(imageData, roiRatio, distance, prominence, 2560);
    
    // Normalized Frequencies: Spacing relative to total image height
    const baseFreq = (result.spacing || 0) / result.actualResolution;
    const hiResFreq = (hiResResult.spacing || 0) / hiResResult.actualResolution;
    const freqStability = Math.abs(baseFreq - hiResFreq) / Math.max(0.0001, baseFreq);

    // If frequencies match within 6%, we have a stable harmonic (true layers).
    // In this case, we prefer the higher count which resolves more detail.
    if (freqStability < 0.06 && hiResResult.count > result.count) {
       return hiResResult as unknown as T;
    }
    
    // If Hi-Res resolves a significantly denser stack (>20% more) even if frequencies shift,
    // and its quality is at least acceptable, it suggests the base resolution was aliased.
    if (hiResResult.count > result.count * 1.2 && hiResResult.qualityScore > 0.55) {
       return hiResResult as unknown as T;
    }
  }

  // Pattern: Dense stack under-counting recovery
  if (result.count > 30 && cv < 0.22 && result.qualityScore > 0.82 && densityError > 0.15) {
    const correctedCount = Math.round(theoreticalCount);
    if (correctedCount > result.count + 5) {
       return { ...result, count: correctedCount };
    }
  }

  return result;
}

export function analyzeImageLocal(
  imageData: ImageDataLike,
  roiRatio: [number, number] = [0.12, 0.94],
  distance: number = 4,
  prominence: number = 6,
  maxResolution: number = 1920
) {
  // Stage 1: Base analysis at default resolution
  const baseResult = analyzeImageLocalAtResolution(imageData, roiRatio, distance, prominence, maxResolution);
  
  // Stage 2: Refinement with potential Resolution Step-up
  return refineResult(baseResult, imageData, roiRatio, distance, prominence);
}

