import {mkdirSync, writeFileSync} from 'node:fs';
import {dirname, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const projectRoot = resolve(here, '..');
const outputPath = resolve(projectRoot, 'public/music/ambient-bed.wav');

const sampleRate = 48000;
const durationSeconds = 66;
const channels = 2;
const frameCount = sampleRate * durationSeconds;
const bytesPerSample = 2;
const dataSize = frameCount * channels * bytesPerSample;
const buffer = Buffer.alloc(44 + dataSize);

const writeString = (offset, value) => buffer.write(value, offset, 'ascii');
writeString(0, 'RIFF');
buffer.writeUInt32LE(36 + dataSize, 4);
writeString(8, 'WAVE');
writeString(12, 'fmt ');
buffer.writeUInt32LE(16, 16);
buffer.writeUInt16LE(1, 20);
buffer.writeUInt16LE(channels, 22);
buffer.writeUInt32LE(sampleRate, 24);
buffer.writeUInt32LE(sampleRate * channels * bytesPerSample, 28);
buffer.writeUInt16LE(channels * bytesPerSample, 32);
buffer.writeUInt16LE(16, 34);
writeString(36, 'data');
buffer.writeUInt32LE(dataSize, 40);

const notes = [
  {freq: 164.81, gain: 0.28},
  {freq: 246.94, gain: 0.2},
  {freq: 329.63, gain: 0.13},
  {freq: 493.88, gain: 0.08},
];

const smoothStep = (edge0, edge1, x) => {
  const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
};

for (let i = 0; i < frameCount; i++) {
  const t = i / sampleRate;
  const intro = smoothStep(0, 5, t);
  const outro = 1 - smoothStep(durationSeconds - 7, durationSeconds, t);
  const swell = 0.78 + 0.22 * Math.sin(2 * Math.PI * t / 18);
  const envelope = intro * outro * swell;

  let sample = 0;
  for (const note of notes) {
    const slowDrift = 0.18 * Math.sin(2 * Math.PI * t / 27 + note.freq);
    sample += Math.sin(2 * Math.PI * (note.freq + slowDrift) * t) * note.gain;
  }

  const shimmer = Math.sin(2 * Math.PI * 987.77 * t) * 0.018 * smoothStep(38, 52, t);
  const value = Math.max(-1, Math.min(1, (sample + shimmer) * envelope * 0.33));
  const left = Math.round(value * 32767);
  const right = Math.round(value * 0.92 * 32767);
  const offset = 44 + i * channels * bytesPerSample;
  buffer.writeInt16LE(left, offset);
  buffer.writeInt16LE(right, offset + 2);
}

mkdirSync(dirname(outputPath), {recursive: true});
writeFileSync(outputPath, buffer);
console.log(`[music] wrote ${outputPath}`);
