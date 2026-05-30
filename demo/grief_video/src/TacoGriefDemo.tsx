import {loadFont as loadInter} from '@remotion/google-fonts/Inter';
import {loadFont as loadJetBrainsMono} from '@remotion/google-fonts/JetBrainsMono';
import React from 'react';
import {
  AbsoluteFill,
  Audio,
  Easing,
  interpolate,
  Sequence,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';
import traceJson from '../public/data/grief-trace.json';
import voiceoverMetaJson from '../public/voiceover/meta.json';
import type {Candidate, GriefTrace, Session, VoiceoverMeta} from './types';

const {fontFamily: inter} = loadInter('normal', {
  weights: ['400', '500', '600', '700', '800'],
  subsets: ['latin'],
});
const {fontFamily: mono} = loadJetBrainsMono('normal', {
  weights: ['400', '500', '700'],
  subsets: ['latin'],
});

const trace = traceJson as GriefTrace;
const voiceoverMeta = voiceoverMetaJson as VoiceoverMeta;

const colors = {
  bg: '#f7f7f4',
  ink: '#111111',
  muted: '#696965',
  faint: '#9a9a94',
  line: '#deded8',
  panel: '#ffffff',
  panelSoft: '#f0f0ec',
  black: '#050505',
};

const ease = Easing.bezier(0.16, 1, 0.3, 1);

const s = (seconds: number, fps: number) => seconds * fps;

const clampFade = (
  frame: number,
  fps: number,
  start: number,
  end: number,
  outStart = end + 2,
  outEnd = outStart + 1,
) => {
  const fadeIn = interpolate(frame, [s(start, fps), s(start + 1.2, fps)], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
    easing: ease,
  });
  const fadeOut = interpolate(frame, [s(outStart, fps), s(outEnd, fps)], [1, 0], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
    easing: ease,
  });
  return frame < s(end, fps) ? fadeIn : fadeOut;
};

const enterY = (frame: number, fps: number, start: number, distance = 26) =>
  interpolate(frame, [s(start, fps), s(start + 1.1, fps)], [distance, 0], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
    easing: ease,
  });

const typed = (text: string, frame: number, fps: number, start: number, duration: number) => {
  const count = Math.floor(
    interpolate(frame, [s(start, fps), s(start + duration, fps)], [0, text.length], {
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
    }),
  );
  return text.slice(0, count);
};

const useSecond = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  return frame / fps;
};

const Voiceover: React.FC = () => {
  const {fps} = useVideoConfig();
  if (!voiceoverMeta.exists) {
    return null;
  }
  if (voiceoverMeta.cues?.length) {
    return (
      <>
        {voiceoverMeta.cues.map((cue) => (
          <Sequence from={Math.round(s(cue.start, fps))} key={cue.file}>
            <Audio src={staticFile(cue.file)} volume={0.96} />
          </Sequence>
        ))}
      </>
    );
  }
  return voiceoverMeta.file ? <Audio src={staticFile(voiceoverMeta.file)} volume={0.94} /> : null;
};

const Shell: React.FC<{children: React.ReactNode}> = ({children}) => (
  <AbsoluteFill
    style={{
      backgroundColor: colors.bg,
      color: colors.ink,
      fontFamily: inter,
      overflow: 'hidden',
    }}
  >
    <div
      style={{
        position: 'absolute',
        inset: 32,
        border: `1px solid ${colors.line}`,
        borderRadius: 34,
        pointerEvents: 'none',
      }}
    />
    <div
      style={{
        position: 'absolute',
        top: 54,
        left: 72,
        fontSize: 18,
        fontWeight: 700,
      }}
    >
      TACO
    </div>
    <div
      style={{
        position: 'absolute',
        top: 54,
        right: 72,
        fontFamily: mono,
        fontSize: 13,
        color: colors.muted,
      }}
    >
      deterministic grief retrieval proof
    </div>
    {children}
  </AbsoluteFill>
);

const Caption: React.FC = () => {
  const t = useSecond();
  const active = trace.voiceover.find((cue) => t >= cue.start && t < cue.end);
  if (!active) {
    return null;
  }
  return (
    <div
      style={{
        position: 'absolute',
        left: 360,
        right: 360,
        bottom: 54,
        minHeight: 56,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        textAlign: 'center',
        fontSize: 24,
        lineHeight: 1.32,
        color: colors.muted,
      }}
    >
      {active.text}
    </div>
  );
};

const KeyboardClicks: React.FC = () => {
  const {fps} = useVideoConfig();
  const ranges = [
    {start: 6.8, end: 9.4, everyFrames: 3},
    {start: 29.7, end: 33.3, everyFrames: 3},
  ];
  const clicks: React.ReactNode[] = [];
  for (const range of ranges) {
    for (
      let frame = Math.round(s(range.start, fps));
      frame < Math.round(s(range.end, fps));
      frame += range.everyFrames
    ) {
      clicks.push(
        <Sequence from={frame} durationInFrames={8} key={`click-${frame}`}>
          <Audio src={staticFile('sfx/key-click.wav')} volume={0.055} />
        </Sequence>,
      );
    }
  }
  return <>{clicks}</>;
};

const Kicker: React.FC<{children: React.ReactNode}> = ({children}) => (
  <div
    style={{
      fontFamily: mono,
      fontSize: 14,
      textTransform: 'uppercase',
      color: colors.muted,
      fontWeight: 700,
    }}
  >
    {children}
  </div>
);

const TitleScene: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const opacity = frame < s(4.7, fps)
    ? 1
    : interpolate(frame, [s(4.7, fps), s(5.35, fps)], [1, 0], {
        extrapolateLeft: 'clamp',
        extrapolateRight: 'clamp',
        easing: ease,
      });
  return (
    <div
      style={{
        opacity,
        transform: `translateY(${enterY(frame, fps, 0.2, 20)}px)`,
        position: 'absolute',
        inset: 0,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        textAlign: 'center',
        paddingBottom: 60,
      }}
    >
      <Kicker>66 second proof of concept</Kicker>
      <div
        style={{
          marginTop: 28,
          fontSize: 98,
          lineHeight: 0.94,
          fontWeight: 800,
        }}
      >
        {trace.title}
      </div>
      <div
        style={{
          marginTop: 28,
          fontSize: 34,
          color: colors.muted,
          fontWeight: 500,
        }}
      >
        {trace.subtitle}
      </div>
    </div>
  );
};

const PhoneBubble: React.FC<{message: string; label: string; active?: boolean}> = ({
  message,
  label,
  active = false,
}) => (
  <div
    style={{
      width: 760,
      border: `1px solid ${active ? colors.black : colors.line}`,
      borderRadius: 28,
      background: colors.panel,
      padding: '30px 34px',
      boxShadow: active ? '0 24px 60px rgba(0,0,0,0.09)' : '0 14px 40px rgba(0,0,0,0.04)',
    }}
  >
    <div
      style={{
        fontFamily: mono,
        fontSize: 13,
        color: colors.muted,
        marginBottom: 14,
        fontWeight: 700,
      }}
    >
      {label}
    </div>
    <div style={{fontSize: 33, lineHeight: 1.22, fontWeight: 600}}>{message}</div>
  </div>
);

const MetricPill: React.FC<{label: string; value: string | number}> = ({label, value}) => (
  <div
    style={{
      border: `1px solid ${colors.line}`,
      borderRadius: 999,
      padding: '12px 16px',
      background: colors.panel,
      display: 'flex',
      gap: 10,
      alignItems: 'center',
      fontFamily: mono,
      fontSize: 16,
    }}
  >
    <span style={{color: colors.faint}}>{label}</span>
    <span style={{fontWeight: 700}}>{value}</span>
  </div>
);

const DisclosureScene: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const opacity = clampFade(frame, fps, 5.5, 18.5, 17.8, 18.5);
  const session = trace.sessions[0];
  const typedMessage = typed(session.message, frame, fps, 6.7, 2.8);
  return (
    <div
      style={{
        opacity,
        transform: `translateY(${enterY(frame, fps, 5.7, 24)}px)`,
        position: 'absolute',
        inset: 0,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 70,
      }}
    >
      <PhoneBubble label="Session 1 / user disclosure" message={typedMessage} active />
      <div style={{width: 560}}>
        <Kicker>write-time valuation</Kicker>
        <div style={{marginTop: 22, fontSize: 54, lineHeight: 1.02, fontWeight: 800}}>
          TACO stores the moment because the state says it matters.
        </div>
        <div style={{display: 'flex', gap: 12, flexWrap: 'wrap', marginTop: 34}}>
          <MetricPill label="E" value={session.state?.E ?? 80} />
          <MetricPill label="V" value={session.state?.V ?? 90} />
          <MetricPill label="salience" value={session.salience} />
          <MetricPill label="tone" value={session.tone} />
        </div>
        <div
          style={{
            marginTop: 34,
            borderTop: `1px solid ${colors.line}`,
            paddingTop: 24,
            color: colors.muted,
            fontSize: 24,
            lineHeight: 1.35,
          }}
        >
          Stored memory: <span style={{color: colors.ink}}>{session.stored_memory}</span>
        </div>
      </div>
    </div>
  );
};

const MiniSession: React.FC<{session: Session; index: number}> = ({session, index}) => (
  <div
    style={{
      width: 480,
      minHeight: 210,
      border: `1px solid ${colors.line}`,
      borderRadius: 26,
      padding: 26,
      background: colors.panel,
      boxShadow: '0 18px 45px rgba(0,0,0,0.045)',
    }}
  >
    <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: 18}}>
      <div style={{fontFamily: mono, color: colors.muted, fontSize: 13, fontWeight: 700}}>
        {session.label}
      </div>
      <div style={{fontFamily: mono, color: colors.faint, fontSize: 13}}>{session.gap}</div>
    </div>
    <div style={{fontSize: 25, lineHeight: 1.25, fontWeight: 600}}>{session.message}</div>
    <div style={{display: 'flex', gap: 10, marginTop: 24}}>
      <MetricPill label="sal" value={session.salience} />
      <MetricPill label="tone" value={session.tone} />
    </div>
  </div>
);

const TimePassesScene: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const opacity = clampFade(frame, fps, 18.5, 28.5, 27.8, 28.5);
  const progress = interpolate(frame, [s(18.5, fps), s(28.5, fps)], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
    easing: ease,
  });
  return (
    <div
      style={{
        opacity,
        position: 'absolute',
        inset: 0,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      <Kicker>weeks pass</Kicker>
      <div style={{marginTop: 18, fontSize: 58, fontWeight: 800}}>
        The semantic surface gets noisy.
      </div>
      <div
        style={{
          marginTop: 54,
          width: 1540,
          height: 270,
          overflow: 'hidden',
          position: 'relative',
        }}
      >
        <div
          style={{
            display: 'flex',
            gap: 24,
            transform: `translateX(${interpolate(progress, [0, 1], [260, -310])}px)`,
          }}
        >
          {trace.sessions.slice(1).map((session, index) => (
            <MiniSession key={session.label} session={session} index={index} />
          ))}
        </div>
      </div>
      <div style={{marginTop: 42, width: 860, color: colors.muted, fontSize: 25, textAlign: 'center'}}>
        Routine details enter the history. The original grief memory has no words in common
        with the moment that will trigger it later.
      </div>
    </div>
  );
};

const ProbeScene: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const opacity = clampFade(frame, fps, 28.5, 39.5, 38.8, 39.5);
  return (
    <div
      style={{
        opacity,
        transform: `translateY(${enterY(frame, fps, 28.7, 20)}px)`,
        position: 'absolute',
        inset: 0,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        textAlign: 'center',
      }}
    >
      <Kicker>session 5 / the probe</Kicker>
      <div style={{marginTop: 26, width: 1180, fontSize: 62, lineHeight: 1.08, fontWeight: 800}}>
        "{typed(trace.probe, frame, fps, 29.6, 3.8)}"
      </div>
      <div style={{display: 'flex', gap: 14, marginTop: 46}}>
        <MetricPill label="query kind" value={trace.query_kind} />
        <MetricPill label="current tone" value={trace.current_tone} />
        <MetricPill label="target lexical overlap" value="0.00" />
      </div>
    </div>
  );
};

const CandidateCard: React.FC<{
  candidate: Candidate;
  mode: 'rag' | 'taco';
  active?: boolean;
}> = ({candidate, mode, active = false}) => {
  const score = mode === 'rag' ? candidate.similarity : candidate.score;
  return (
    <div
      style={{
        border: `1px solid ${active ? colors.black : colors.line}`,
        borderRadius: 22,
        background: active ? colors.black : colors.panel,
        color: active ? colors.panel : colors.ink,
        padding: '18px 24px',
        minHeight: 126,
      }}
    >
      <div
        style={{
          fontFamily: mono,
          fontSize: 13,
          color: active ? '#d5d5d0' : colors.muted,
          marginBottom: 10,
        }}
      >
        {candidate.kind} / {mode === 'rag' ? 'semantic sim' : 'TACO score'} {score.toFixed(3)}
      </div>
      <div style={{fontSize: 23, lineHeight: 1.18, fontWeight: 650}}>{candidate.content}</div>
      <div style={{display: 'flex', gap: 10, marginTop: 14, flexWrap: 'wrap'}}>
        <MiniMetric label="sim" value={candidate.similarity.toFixed(2)} dark={active} />
        <MiniMetric label="sal" value={candidate.salience.toFixed(0)} dark={active} />
        <MiniMetric label="tone" value={candidate.tone} dark={active} />
      </div>
    </div>
  );
};

const MiniMetric: React.FC<{label: string; value: string; dark?: boolean}> = ({
  label,
  value,
  dark = false,
}) => (
  <div
    style={{
      fontFamily: mono,
      border: `1px solid ${dark ? '#40403d' : colors.line}`,
      borderRadius: 999,
      padding: '7px 10px',
      fontSize: 12,
      color: dark ? '#e8e8e2' : colors.muted,
    }}
  >
    {label} {value}
  </div>
);

const ComparisonPanel: React.FC<{
  title: string;
  subtitle: string;
  candidates: Candidate[];
  mode: 'rag' | 'taco';
}> = ({title, subtitle, candidates, mode}) => (
  <div
    style={{
      width: 780,
      minHeight: 584,
      border: `1px solid ${colors.line}`,
      borderRadius: 30,
      background: 'rgba(255,255,255,0.82)',
      padding: 28,
      boxShadow: '0 22px 70px rgba(0,0,0,0.06)',
    }}
  >
    <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'start'}}>
      <div>
        <div style={{fontSize: 32, fontWeight: 800}}>{title}</div>
        <div style={{marginTop: 7, color: colors.muted, fontSize: 19}}>{subtitle}</div>
      </div>
      <div
        style={{
          fontFamily: mono,
          fontSize: 13,
          border: `1px solid ${colors.line}`,
          borderRadius: 999,
          padding: '9px 12px',
          color: colors.muted,
        }}
      >
        top-1
      </div>
    </div>
    <div style={{display: 'flex', flexDirection: 'column', gap: 12, marginTop: 24}}>
      {candidates.map((candidate, index) => (
        <CandidateCard
          key={`${mode}-${candidate.content}`}
          candidate={candidate}
          mode={mode}
          active={index === 0}
        />
      ))}
    </div>
  </div>
);

const RetrievalScene: React.FC<{kind: 'rag' | 'taco'; start: number; end: number}> = ({
  kind,
  start,
  end,
}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const opacity = clampFade(frame, fps, start, end, end - 0.8, end + 0.2);
  const sortedForRag = [...trace.candidates].sort((a, b) => b.similarity - a.similarity);
  const sortedForTaco = [...trace.candidates].sort((a, b) => b.score - a.score);
  const tacoActive = kind === 'taco';
  return (
    <div
      style={{
        opacity,
        transform: `translateY(${enterY(frame, fps, start + 0.1, 16)}px)`,
        position: 'absolute',
        inset: 0,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      <Kicker>{tacoActive ? 'state-conditioned recall' : 'semantic retrieval'}</Kicker>
      <div style={{marginTop: 18, fontSize: 52, fontWeight: 800}}>
        {tacoActive ? trace.taco.label : trace.naive.label}
      </div>
      <div style={{display: 'flex', gap: 14, marginTop: 22}}>
        <MetricPill label="w_sem" value={trace.weights.sem.toFixed(3)} />
        <MetricPill label="w_emo" value={trace.weights.emo.toFixed(3)} />
        <MetricPill label="S" value={`E${trace.state.E} V${trace.state.V}`} />
      </div>
      <div style={{display: 'flex', gap: 28, marginTop: 26}}>
        <ComparisonPanel
          title="Naive RAG"
          subtitle="Ranks by semantic similarity"
          candidates={sortedForRag}
          mode="rag"
        />
        <ComparisonPanel
          title="TACO"
          subtitle="Reranks by state, salience, tone"
          candidates={sortedForTaco}
          mode="taco"
        />
      </div>
    </div>
  );
};

const ClosingScene: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const opacity = clampFade(frame, fps, 61.5, 66, 67, 68);
  return (
    <div
      style={{
        opacity,
        transform: `translateY(${enterY(frame, fps, 61.7, 18)}px)`,
        position: 'absolute',
        inset: 0,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        textAlign: 'center',
        paddingBottom: 30,
      }}
    >
      <Kicker>TACO Memory Architecture</Kicker>
      <div style={{marginTop: 28, width: 1280, fontSize: 84, lineHeight: 1, fontWeight: 850}}>
        {trace.thesis}
      </div>
      <div style={{marginTop: 32, width: 980, color: colors.muted, fontSize: 34, lineHeight: 1.22}}>
        What mattered, to whom, when, and in what state?
      </div>
    </div>
  );
};

export const TacoGriefDemo: React.FC = () => {
  return (
    <Shell>
      <Audio src={staticFile('music/ambient-bed.wav')} volume={0.075} />
      <Voiceover />
      <KeyboardClicks />
      <TitleScene />
      <DisclosureScene />
      <TimePassesScene />
      <ProbeScene />
      <RetrievalScene kind="rag" start={39.5} end={50.5} />
      <RetrievalScene kind="taco" start={50.5} end={61.5} />
      <ClosingScene />
      <Caption />
    </Shell>
  );
};
