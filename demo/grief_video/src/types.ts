export type Session = {
  label: string;
  gap: string;
  message: string;
  state?: {E: number; K: number; V: number; R: number};
  stored_memory?: string;
  salience: number;
  tone: string;
};

export type Candidate = {
  content: string;
  similarity: number;
  salience: number;
  tone: string;
  score: number;
  kind: 'target' | 'surface-match' | 'filler';
  lexical_overlap: number;
};

export type VoiceoverCue = {
  start: number;
  end: number;
  text: string;
};

export type GriefTrace = {
  title: string;
  subtitle: string;
  thesis: string;
  sessions: Session[];
  probe: string;
  query_kind: string;
  state: {E: number; K: number; V: number; R: number};
  current_tone: string;
  weights: Record<string, number>;
  naive: {top: string; label: string; reason: string};
  taco: {top: string; label: string; reason: string};
  candidates: Candidate[];
  proof: Record<string, boolean | number>;
  voiceover: VoiceoverCue[];
};

export type VoiceoverMeta = {
  exists: boolean;
  file?: string;
  cues?: Array<{file: string; start: number; end: number; sha256: string}>;
  sha256: string;
};
