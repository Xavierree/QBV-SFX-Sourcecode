// Mono, 16-bit PCM WAV encoding from a decoded AudioBuffer — ported from the old SFX/webapp's
// static/app.js (the recording pipeline itself was never the broken part of that app).
export function audioBufferToWav(buffer: AudioBuffer): Blob {
  const numFrames = buffer.length;
  const sampleRate = buffer.sampleRate;
  let data: Float32Array;
  if (buffer.numberOfChannels === 1) {
    data = buffer.getChannelData(0);
  } else {
    const ch0 = buffer.getChannelData(0);
    const ch1 = buffer.getChannelData(1);
    data = new Float32Array(numFrames);
    for (let i = 0; i < numFrames; i++) data[i] = (ch0[i] + ch1[i]) / 2;
  }

  const bytesPerSample = 2;
  const dataSize = numFrames * bytesPerSample;
  const buf = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buf);

  const writeStr = (offset: number, str: string) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };

  writeStr(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * bytesPerSample, true);
  view.setUint16(32, bytesPerSample, true);
  view.setUint16(34, 16, true); // bits/sample
  writeStr(36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < numFrames; i++) {
    const s = Math.max(-1, Math.min(1, data[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }
  return new Blob([buf], { type: "audio/wav" });
}

// Static min/max-per-pixel waveform render, shared by anything that needs to preview a decoded
// AudioBuffer on a canvas (the recorder in useRecorder.ts draws its own live version while capturing,
// then switches to this same style once stopped).
export function drawStaticWaveform(canvas: HTMLCanvasElement, buffer: AudioBuffer) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const data = buffer.getChannelData(0);
  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#141420";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#818cf8";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  const samplesPerPx = Math.max(1, Math.floor(data.length / width));
  for (let x = 0; x < width; x++) {
    const start = x * samplesPerPx;
    let min = 1;
    let max = -1;
    for (let i = 0; i < samplesPerPx && start + i < data.length; i++) {
      const v = data[start + i];
      if (v < min) min = v;
      if (v > max) max = v;
    }
    const yMin = (1 - (min * 0.5 + 0.5)) * height;
    const yMax = (1 - (max * 0.5 + 0.5)) * height;
    ctx.moveTo(x, yMin);
    ctx.lineTo(x, yMax);
  }
  ctx.stroke();
}
