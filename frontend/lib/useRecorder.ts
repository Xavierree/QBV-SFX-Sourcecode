"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { audioBufferToWav } from "@/lib/wav";
import { Endpointer, rmsFromBytes } from "@/lib/endpointing";

const MAX_RECORD_MS = 10_000;

export function useRecorder(
  onStop: (wavBlob: Blob, duration: number, setStatus: (s: string) => void) => void,
  options: { autoStop?: boolean } = {}
) {
  const [recording, setRecording] = useState(false);
  const [status, setStatus] = useState("Click record, then make your sound (up to 10s).");

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const recordTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const rafRef = useRef<number | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const stopRef = useRef<() => void>(() => {});
  const autoStop = options.autoStop ?? false;
  const onStopRef = useRef(onStop);
  useEffect(() => {
    onStopRef.current = onStop;
  }, [onStop]);

  const drawWaveformFromBytes = useCallback((byteData: Uint8Array) => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    const { width, height } = canvas;
    ctx.clearRect(0, 0, width, height);
    ctx.strokeStyle = "#818cf8";
    ctx.lineWidth = 2;
    ctx.beginPath();
    const step = byteData.length / width;
    for (let x = 0; x < width; x++) {
      const v = byteData[Math.floor(x * step)] / 128.0 - 1.0;
      const y = (v * 0.5 + 0.5) * height;
      if (x === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
  }, []);

  const drawStaticWaveform = useCallback((buffer: AudioBuffer) => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
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
  }, []);

  const stopLiveMeter = useCallback(() => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    audioCtxRef.current?.close();
    audioCtxRef.current = null;
    analyserRef.current = null;
  }, []);

  const startLiveMeter = useCallback(
    (stream: MediaStream) => {
      const audioCtx = new AudioContext();
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(analyser);
      audioCtxRef.current = audioCtx;
      analyserRef.current = analyser;

      const buf = new Uint8Array(analyser.fftSize);
      const endpointer = autoStop ? new Endpointer() : null;
      const startedAt = performance.now();
      const draw = () => {
        analyser.getByteTimeDomainData(buf);
        drawWaveformFromBytes(buf);
        if (endpointer && endpointer.push(performance.now() - startedAt, rmsFromBytes(buf)) === "stop") {
          stopRef.current();
          return;
        }
        rafRef.current = requestAnimationFrame(draw);
      };
      draw();
    },
    [drawWaveformFromBytes, autoStop]
  );

  const onRecordingStop = useCallback(async () => {
    setStatus("Processing…");
    const recorder = mediaRecorderRef.current;
    const blob = new Blob(chunksRef.current, { type: recorder?.mimeType || "audio/webm" });
    try {
      const arrBuf = await blob.arrayBuffer();
      const ctx = new AudioContext();
      const audioBuffer = await ctx.decodeAudioData(arrBuf);
      await ctx.close();
      drawStaticWaveform(audioBuffer);
      const wavBlob = audioBufferToWav(audioBuffer);
      onStopRef.current(wavBlob, audioBuffer.duration, setStatus);
    } catch (err) {
      setStatus(`Could not decode recording: ${err instanceof Error ? err.message : "unknown error"}`);
    }
  }, [drawStaticWaveform]);

  const stopRecording = useCallback(() => {
    if (recordTimeoutRef.current) {
      clearTimeout(recordTimeoutRef.current);
      recordTimeoutRef.current = null;
    }
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
    }
    mediaStreamRef.current?.getTracks().forEach((t) => t.stop());
    stopLiveMeter();
    setRecording(false);
  }, [stopLiveMeter]);

  useEffect(() => {
    stopRef.current = stopRecording;
  }, [stopRecording]);

  const startRecording = useCallback(async () => {
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      setStatus(`Microphone access denied: ${err instanceof Error ? err.message : "unknown error"}`);
      return;
    }
    mediaStreamRef.current = stream;
    chunksRef.current = [];
    const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
      ? "audio/webm;codecs=opus"
      : "";
    const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data);
    };
    recorder.onstop = onRecordingStop;
    recorder.start();
    mediaRecorderRef.current = recorder;

    setRecording(true);
    setStatus(autoStop ? "Recording… it stops by itself when you go quiet, or click stop." : "Recording… click stop when done.");
    startLiveMeter(stream);
    recordTimeoutRef.current = setTimeout(() => stopRecording(), MAX_RECORD_MS);
  }, [onRecordingStop, startLiveMeter, stopRecording, autoStop]);

  const toggleRecording = useCallback(() => {
    if (recording) {
      stopRecording();
    } else {
      startRecording();
    }
  }, [recording, startRecording, stopRecording]);

  useEffect(() => () => stopRecording(), [stopRecording]);

  return { recording, status, setStatus, canvasRef, toggleRecording };
}
