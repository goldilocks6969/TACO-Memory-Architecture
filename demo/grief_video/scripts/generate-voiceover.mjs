import {createHash} from 'node:crypto';
import {existsSync, mkdirSync, readFileSync, writeFileSync} from 'node:fs';
import {dirname, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const projectRoot = resolve(here, '..');
const repoRoot = resolve(projectRoot, '../..');
const tracePath = resolve(projectRoot, 'public/data/grief-trace.json');
const audioPath = resolve(projectRoot, 'public/voiceover/narration.mp3');
const cueDir = resolve(projectRoot, 'public/voiceover/cues');
const metaPath = resolve(projectRoot, 'public/voiceover/meta.json');

const loadEnvFile = (path) => {
  if (!existsSync(path)) {
    return;
  }
  for (const line of readFileSync(path, 'utf8').split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#') || !trimmed.includes('=')) {
      continue;
    }
    const [key, ...rest] = trimmed.split('=');
    if (!process.env[key]) {
      process.env[key] = rest.join('=').replace(/^["']|["']$/g, '');
    }
  }
};

loadEnvFile(resolve(repoRoot, '.env'));
loadEnvFile(resolve(projectRoot, '.env'));

const trace = JSON.parse(readFileSync(tracePath, 'utf8'));
const voiceId = process.env.ELEVENLABS_VOICE_ID;
const apiKey = process.env.ELEVENLABS_API_KEY;
const model = process.env.ELEVENLABS_MODEL_ID || 'eleven_multilingual_v2';
const speed = Number.parseFloat(process.env.ELEVENLABS_SPEED || '0.82');
const settings = {
  stability: 0.64,
  similarity_boost: 0.78,
  style: 0.12,
  speed,
  use_speaker_boost: true,
};
const hash = createHash('sha256')
  .update(JSON.stringify({voiceover: trace.voiceover, voiceId, model, settings}))
  .digest('hex');

const existingMeta = existsSync(metaPath)
  ? JSON.parse(readFileSync(metaPath, 'utf8'))
  : {exists: false};

if (
  existingMeta.exists &&
  existingMeta.sha256 === hash &&
  existingMeta.cues?.every((cue) => existsSync(resolve(projectRoot, 'public', cue.file)))
) {
  console.log('[voiceover] cached cue narration already matches script');
  process.exit(0);
}

const enableCaptionsOnly = (message) => {
  if (existsSync(audioPath) && existingMeta.exists) {
    console.log(`[voiceover] ${message}; keeping existing cached narration`);
    process.exit(0);
  }
  writeFileSync(
    metaPath,
    JSON.stringify({exists: false, file: 'voiceover/narration.mp3', sha256: ''}, null, 2) + '\n',
  );
  console.log(`[voiceover] ${message}; captions-only render is enabled`);
  process.exit(0);
};

if (!apiKey) {
  enableCaptionsOnly('ELEVENLABS_API_KEY missing');
}

if (!voiceId) {
  enableCaptionsOnly('ELEVENLABS_VOICE_ID missing');
}

if (/^[a-f0-9]{48,}$/i.test(voiceId)) {
  enableCaptionsOnly('ELEVENLABS_VOICE_ID looks like a key id, not a voice id');
}

const failSoft = async (response) => {
  const body = await response.text();
  let message = `${response.status} ${body}`;
  try {
    const parsed = JSON.parse(body);
    message = parsed.detail?.message || message;
  } catch {
    // Keep the raw body if ElevenLabs returns non-JSON.
  }
  enableCaptionsOnly(`ElevenLabs failed: ${message}`);
};

mkdirSync(cueDir, {recursive: true});

const cueMeta = [];
for (const [index, cue] of trace.voiceover.entries()) {
  const cueHash = createHash('sha256')
    .update(JSON.stringify({cue, voiceId, model, settings}))
    .digest('hex');
  const fileName = `cue-${String(index).padStart(2, '0')}.mp3`;
  const filePath = resolve(cueDir, fileName);
  if (existsSync(filePath) && existingMeta.cues?.some((item) => item.sha256 === cueHash)) {
    cueMeta.push({file: `voiceover/cues/${fileName}`, start: cue.start, end: cue.end, sha256: cueHash});
    continue;
  }

  const response = await fetch(
    `https://api.elevenlabs.io/v1/text-to-speech/${voiceId}?output_format=mp3_44100_128`,
    {
      method: 'POST',
      headers: {
        'xi-api-key': apiKey,
        'Content-Type': 'application/json',
        Accept: 'audio/mpeg',
      },
      body: JSON.stringify({
        text: cue.text,
        model_id: model,
        voice_settings: settings,
      }),
    },
  );

  if (!response.ok) {
    await failSoft(response);
  }

  const audio = Buffer.from(await response.arrayBuffer());
  writeFileSync(filePath, audio);
  cueMeta.push({file: `voiceover/cues/${fileName}`, start: cue.start, end: cue.end, sha256: cueHash});
  console.log(`[voiceover] wrote cue ${index + 1}/${trace.voiceover.length}`);
}
writeFileSync(
  metaPath,
  JSON.stringify({exists: true, file: 'voiceover/narration.mp3', cues: cueMeta, sha256: hash}, null, 2) + '\n',
);
console.log(`[voiceover] wrote ${cueMeta.length} timed cues`);
