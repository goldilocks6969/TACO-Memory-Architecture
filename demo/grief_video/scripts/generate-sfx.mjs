import {mkdirSync, writeFileSync} from 'node:fs';
import {dirname, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const out = resolve(here, '../public/sfx/key-click.wav');
mkdirSync(dirname(out), {recursive: true});

const sampleRate = 44100;
const duration = 0.035;
const samples = Math.floor(sampleRate * duration);
const data = Buffer.alloc(samples * 2);

for (let i = 0; i < samples; i += 1) {
  const t = i / sampleRate;
  const envelope = Math.exp(-t * 120);
  const noise = (Math.random() * 2 - 1) * 0.18;
  const tick = Math.sin(2 * Math.PI * 2200 * t) * 0.12;
  const value = Math.max(-1, Math.min(1, (noise + tick) * envelope));
  data.writeInt16LE(Math.floor(value * 32767), i * 2);
}

const header = Buffer.alloc(44);
header.write('RIFF', 0);
header.writeUInt32LE(36 + data.length, 4);
header.write('WAVE', 8);
header.write('fmt ', 12);
header.writeUInt32LE(16, 16);
header.writeUInt16LE(1, 20);
header.writeUInt16LE(1, 22);
header.writeUInt32LE(sampleRate, 24);
header.writeUInt32LE(sampleRate * 2, 28);
header.writeUInt16LE(2, 32);
header.writeUInt16LE(16, 34);
header.write('data', 36);
header.writeUInt32LE(data.length, 40);

writeFileSync(out, Buffer.concat([header, data]));
console.log(`[sfx] wrote ${out}`);
