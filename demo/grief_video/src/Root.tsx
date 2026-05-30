import {Composition} from 'remotion';
import {TacoGriefDemo} from './TacoGriefDemo';

export const FPS = 30;
export const DURATION_SECONDS = 66;
export const WIDTH = 1920;
export const HEIGHT = 1080;

export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="TACOGriefDemo"
      component={TacoGriefDemo}
      durationInFrames={DURATION_SECONDS * FPS}
      fps={FPS}
      width={WIDTH}
      height={HEIGHT}
    />
  );
};
