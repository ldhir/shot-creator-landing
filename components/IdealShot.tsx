'use client';

import React, { useState } from 'react';
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
} from 'recharts';

// ============ TYPES ============
interface IdealShotProps {
  player: { name: string; height: string; handedness: 'Left' | 'Right' };
  sessionConsistency: number;
  historicalConsistency: { career: number; lastWeek: number; lastMonth: number };
  radarMetrics: {
    elbowFlare: number;
    trunkLean: number;
    kneeBend: number;
    elbowExtension: number;
    wristSnap: number;
  };
  leftMetrics: {
    rhythm: { kneeTime: number; elbowTime: number; isAligned: boolean };
    footAlignment: { angleOff: number };
    footStance: { width: number };
    releaseHeight: { current: number; targetMin: number; targetMax: number };
  };
  quickTips: string[];
  workOn: Array<{
    title: string;
    description: string;
    drillName: string;
    drillTip: string;
  }>;
}

type ConsistencyPeriod = 'session' | 'lastWeek' | 'lastMonth' | 'career';

// ============ STYLES ============
const styles = `
  @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=DM+Sans:wght@400;500;600&display=swap');

  .ideal-shot {
    --bg-primary: #0D0D0F;
    --bg-card: #161619;
    --bg-card-hover: #1C1C20;
    --border: #2A2A30;
    --text-primary: #FFFFFF;
    --text-secondary: #8B8B96;
    --text-muted: #5A5A64;
    --accent-coral: #FF6B5B;
    --accent-cyan: #00D4FF;
    --accent-green: #3DD68C;
    --accent-yellow: #FFD93D;

    font-family: 'DM Sans', -apple-system, sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    min-height: 100vh;
    padding: 32px;
  }

  .ideal-shot * {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }

  /* Header */
  .is-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 40px;
    padding-bottom: 24px;
    border-bottom: 1px solid var(--border);
  }

  .is-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 56px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    background: linear-gradient(135deg, var(--text-primary) 0%, var(--text-secondary) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
    margin-bottom: 12px;
  }

  .is-title span {
    color: var(--accent-coral);
    -webkit-text-fill-color: var(--accent-coral);
  }

  .is-player-info {
    display: flex;
    gap: 24px;
    font-size: 14px;
    color: var(--text-secondary);
  }

  .is-player-info span {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .is-player-info .label {
    color: var(--text-muted);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .is-consistency-box {
    text-align: right;
  }

  .is-consistency-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--text-muted);
    margin-bottom: 8px;
  }

  .is-consistency-value {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 48px;
    font-weight: 700;
    color: var(--accent-cyan);
    line-height: 1;
  }

  .is-consistency-toggle {
    display: flex;
    gap: 4px;
    margin-top: 12px;
    background: var(--bg-card);
    border-radius: 8px;
    padding: 4px;
  }

  .is-toggle-btn {
    background: none;
    border: none;
    padding: 6px 12px;
    font-size: 11px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    color: var(--text-muted);
    cursor: pointer;
    border-radius: 6px;
    transition: all 0.2s ease;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .is-toggle-btn:hover {
    color: var(--text-secondary);
  }

  .is-toggle-btn.active {
    background: var(--accent-coral);
    color: var(--bg-primary);
  }

  /* Main Layout */
  .is-main {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 32px;
    margin-bottom: 40px;
  }

  .is-left-col {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
  }

  .is-right-col {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
  }

  /* Metric Cards */
  .is-metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px;
    transition: all 0.3s ease;
  }

  .is-metric-card:hover {
    background: var(--bg-card-hover);
    border-color: var(--accent-coral);
  }

  .is-metric-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 14px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: var(--text-muted);
    margin-bottom: 16px;
  }

  /* Rhythm Visualization */
  .is-rhythm-viz {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .is-rhythm-bar-container {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .is-rhythm-label {
    font-size: 11px;
    color: var(--text-secondary);
    width: 50px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .is-rhythm-track {
    flex: 1;
    height: 8px;
    background: var(--border);
    border-radius: 4px;
    position: relative;
    overflow: visible;
  }

  .is-rhythm-marker {
    position: absolute;
    top: 50%;
    transform: translate(-50%, -50%);
    width: 16px;
    height: 16px;
    border-radius: 50%;
    transition: left 0.4s ease;
  }

  .is-rhythm-marker.knee {
    background: var(--accent-cyan);
    box-shadow: 0 0 12px var(--accent-cyan);
  }

  .is-rhythm-marker.elbow {
    background: var(--accent-coral);
    box-shadow: 0 0 12px var(--accent-coral);
  }

  .is-rhythm-status {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 8px;
    font-size: 12px;
  }

  .is-rhythm-status.aligned {
    color: var(--accent-green);
  }

  .is-rhythm-status.offset {
    color: var(--accent-yellow);
  }

  .is-rhythm-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: currentColor;
  }

  /* Foot Alignment Arc */
  .is-arc-container {
    display: flex;
    flex-direction: column;
    align-items: center;
  }

  .is-arc-svg {
    width: 100%;
    max-width: 160px;
  }

  .is-arc-value {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 28px;
    font-weight: 700;
    margin-top: 8px;
  }

  .is-arc-unit {
    font-size: 14px;
    color: var(--text-muted);
    font-weight: 400;
  }

  /* Foot Stance */
  .is-stance-viz {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 16px;
  }

  .is-stance-track {
    width: 100%;
    height: 40px;
    background: var(--border);
    border-radius: 8px;
    position: relative;
  }

  .is-stance-zone {
    position: absolute;
    top: 0;
    height: 100%;
    background: rgba(61, 214, 140, 0.15);
    border-left: 2px dashed var(--accent-green);
    border-right: 2px dashed var(--accent-green);
  }

  .is-stance-foot {
    position: absolute;
    top: 50%;
    transform: translate(-50%, -50%);
    width: 12px;
    height: 24px;
    border-radius: 6px;
    background: var(--accent-coral);
    box-shadow: 0 0 8px rgba(255, 107, 91, 0.5);
  }

  .is-stance-label {
    font-size: 12px;
    color: var(--text-secondary);
  }

  /* Release Height */
  .is-release-viz {
    display: flex;
    align-items: flex-end;
    gap: 20px;
    height: 120px;
  }

  .is-release-bar {
    flex: 1;
    height: 100%;
    background: var(--border);
    border-radius: 8px;
    position: relative;
    overflow: hidden;
  }

  .is-release-fill {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    background: linear-gradient(180deg, var(--accent-cyan) 0%, rgba(0, 212, 255, 0.3) 100%);
    border-radius: 0 0 8px 8px;
    transition: height 0.5s ease;
  }

  .is-release-target {
    position: absolute;
    left: 0;
    right: 0;
    background: rgba(61, 214, 140, 0.2);
    border-top: 2px dashed var(--accent-green);
    border-bottom: 2px dashed var(--accent-green);
  }

  .is-release-info {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .is-release-value {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 32px;
    font-weight: 700;
    color: var(--accent-cyan);
  }

  .is-release-target-text {
    font-size: 11px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  /* Radar Chart */
  .is-radar-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 18px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: var(--text-secondary);
    margin-bottom: 16px;
    text-align: center;
  }

  .is-radar-container {
    height: 320px;
  }

  .is-radar-legend {
    display: flex;
    justify-content: center;
    gap: 24px;
    margin-top: 16px;
  }

  .is-legend-item {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    color: var(--text-secondary);
  }

  .is-legend-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
  }

  .is-legend-dot.player {
    background: var(--accent-coral);
  }

  .is-legend-dot.ideal {
    background: var(--accent-green);
    opacity: 0.5;
  }

  /* Bottom Sections */
  .is-bottom {
    display: grid;
    grid-template-columns: 1fr 2fr;
    gap: 32px;
  }

  .is-section-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 20px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 3px;
    color: var(--text-primary);
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .is-section-title::before {
    content: '';
    width: 4px;
    height: 20px;
    background: var(--accent-coral);
    border-radius: 2px;
  }

  /* Quick Tips */
  .is-tips-list {
    list-style: none;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .is-tip-item {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 16px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    font-size: 14px;
    line-height: 1.5;
    color: var(--text-secondary);
    transition: all 0.2s ease;
  }

  .is-tip-item:hover {
    border-color: var(--accent-cyan);
    background: var(--bg-card-hover);
  }

  .is-tip-bullet {
    width: 6px;
    height: 6px;
    background: var(--accent-cyan);
    border-radius: 50%;
    margin-top: 6px;
    flex-shrink: 0;
  }

  /* Work On Cards */
  .is-workon-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
  }

  .is-workon-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
  }

  .is-workon-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent-coral) 0%, var(--accent-yellow) 100%);
    opacity: 0;
    transition: opacity 0.3s ease;
  }

  .is-workon-card:hover {
    border-color: var(--accent-coral);
    transform: translateY(-4px);
  }

  .is-workon-card:hover::before {
    opacity: 1;
  }

  .is-workon-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 18px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-primary);
    margin-bottom: 8px;
  }

  .is-workon-desc {
    font-size: 13px;
    color: var(--text-secondary);
    line-height: 1.5;
    margin-bottom: 16px;
  }

  .is-workon-drill {
    background: rgba(0, 212, 255, 0.08);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 10px;
    padding: 14px;
  }

  .is-workon-drill-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--accent-cyan);
    margin-bottom: 6px;
  }

  .is-workon-drill-name {
    font-weight: 600;
    font-size: 14px;
    color: var(--text-primary);
    margin-bottom: 6px;
  }

  .is-workon-drill-tip {
    font-size: 12px;
    color: var(--text-muted);
    line-height: 1.4;
  }

  /* Responsive */
  @media (max-width: 1200px) {
    .is-main {
      grid-template-columns: 1fr;
    }

    .is-bottom {
      grid-template-columns: 1fr;
    }

    .is-workon-grid {
      grid-template-columns: 1fr;
    }
  }

  @media (max-width: 768px) {
    .ideal-shot {
      padding: 20px;
    }

    .is-header {
      flex-direction: column;
      gap: 20px;
    }

    .is-consistency-box {
      text-align: left;
    }

    .is-left-col {
      grid-template-columns: 1fr;
    }

    .is-title {
      font-size: 40px;
    }
  }
`;

// ============ SUB-COMPONENTS ============

// Rhythm Visualization
const RhythmViz: React.FC<{
  kneeTime: number;
  elbowTime: number;
  isAligned: boolean;
}> = ({ kneeTime, elbowTime, isAligned }) => {
  const maxTime = Math.max(kneeTime, elbowTime, 1);
  const kneePos = (kneeTime / maxTime) * 80 + 10;
  const elbowPos = (elbowTime / maxTime) * 80 + 10;

  return (
    <div className="is-rhythm-viz">
      <div className="is-rhythm-bar-container">
        <span className="is-rhythm-label">Knee</span>
        <div className="is-rhythm-track">
          <div
            className="is-rhythm-marker knee"
            style={{ left: `${kneePos}%` }}
          />
        </div>
      </div>
      <div className="is-rhythm-bar-container">
        <span className="is-rhythm-label">Elbow</span>
        <div className="is-rhythm-track">
          <div
            className="is-rhythm-marker elbow"
            style={{ left: `${elbowPos}%` }}
          />
        </div>
      </div>
      <div className={`is-rhythm-status ${isAligned ? 'aligned' : 'offset'}`}>
        <span className="is-rhythm-dot" />
        {isAligned ? 'Synced' : `${Math.abs(kneeTime - elbowTime).toFixed(2)}s offset`}
      </div>
    </div>
  );
};

// Foot Alignment Arc
const FootAlignmentArc: React.FC<{ angleOff: number }> = ({ angleOff }) => {
  const clampedAngle = Math.max(-45, Math.min(45, angleOff));
  const needleAngle = -90 + (clampedAngle / 45) * 90;
  const isGood = Math.abs(angleOff) < 5;

  return (
    <div className="is-arc-container">
      <svg className="is-arc-svg" viewBox="0 0 100 60">
        {/* Background arc */}
        <path
          d="M 10 55 A 40 40 0 0 1 90 55"
          fill="none"
          stroke="#2A2A30"
          strokeWidth="8"
          strokeLinecap="round"
        />
        {/* Good zone */}
        <path
          d="M 45 17.5 A 40 40 0 0 1 55 17.5"
          fill="none"
          stroke="rgba(61, 214, 140, 0.3)"
          strokeWidth="10"
          strokeLinecap="round"
        />
        {/* Needle */}
        <g transform={`rotate(${needleAngle}, 50, 55)`}>
          <line
            x1="50"
            y1="55"
            x2="50"
            y2="20"
            stroke={isGood ? '#3DD68C' : '#FF6B5B'}
            strokeWidth="3"
            strokeLinecap="round"
          />
          <circle cx="50" cy="55" r="6" fill={isGood ? '#3DD68C' : '#FF6B5B'} />
        </g>
        {/* Labels */}
        <text x="8" y="58" fill="#5A5A64" fontSize="8">-45°</text>
        <text x="46" y="12" fill="#5A5A64" fontSize="8">0°</text>
        <text x="82" y="58" fill="#5A5A64" fontSize="8">+45°</text>
      </svg>
      <div className="is-arc-value" style={{ color: isGood ? '#3DD68C' : '#FF6B5B' }}>
        {angleOff > 0 ? '+' : ''}{angleOff}
        <span className="is-arc-unit">°</span>
      </div>
    </div>
  );
};

// Foot Stance Visualization
const FootStanceViz: React.FC<{ width: number }> = ({ width }) => {
  // width is in inches, shoulder width zone is typically 16-20 inches
  const minZone = 16;
  const maxZone = 20;
  const trackWidth = 100;
  const scale = trackWidth / 30; // 30 inches max display

  const leftFoot = 50 - (width / 2) * scale;
  const rightFoot = 50 + (width / 2) * scale;
  const zoneLeft = 50 - (maxZone / 2) * scale;
  const zoneWidth = (maxZone - minZone + maxZone - minZone) * scale / 2 + (maxZone - minZone) * scale;
  const isInZone = width >= minZone && width <= maxZone;

  return (
    <div className="is-stance-viz">
      <div className="is-stance-track">
        <div
          className="is-stance-zone"
          style={{
            left: `${50 - (maxZone / 2) * scale}%`,
            width: `${maxZone * scale}%`,
          }}
        />
        <div className="is-stance-foot" style={{ left: `${leftFoot}%` }} />
        <div className="is-stance-foot" style={{ left: `${rightFoot}%` }} />
      </div>
      <span className="is-stance-label" style={{ color: isInZone ? '#3DD68C' : '#FFD93D' }}>
        {width}" width {isInZone ? '(optimal)' : width < minZone ? '(narrow)' : '(wide)'}
      </span>
    </div>
  );
};

// Release Height Visualization
const ReleaseHeightViz: React.FC<{
  current: number;
  targetMin: number;
  targetMax: number;
}> = ({ current, targetMin, targetMax }) => {
  const maxHeight = 120; // inches
  const fillPercent = (current / maxHeight) * 100;
  const targetMinPercent = (targetMin / maxHeight) * 100;
  const targetMaxPercent = (targetMax / maxHeight) * 100;
  const isInTarget = current >= targetMin && current <= targetMax;

  return (
    <div className="is-release-viz">
      <div className="is-release-bar">
        <div className="is-release-fill" style={{ height: `${fillPercent}%` }} />
        <div
          className="is-release-target"
          style={{
            bottom: `${targetMinPercent}%`,
            height: `${targetMaxPercent - targetMinPercent}%`,
          }}
        />
      </div>
      <div className="is-release-info">
        <span className="is-release-value">{current}"</span>
        <span className="is-release-target-text">
          Target: {targetMin}-{targetMax}"
        </span>
        <span
          className="is-release-target-text"
          style={{ color: isInTarget ? '#3DD68C' : '#FFD93D' }}
        >
          {isInTarget ? 'In range' : current < targetMin ? 'Too low' : 'Too high'}
        </span>
      </div>
    </div>
  );
};

// ============ MAIN COMPONENT ============
export default function IdealShot({
  player,
  sessionConsistency,
  historicalConsistency,
  radarMetrics,
  leftMetrics,
  quickTips,
  workOn,
}: IdealShotProps) {
  const [consistencyPeriod, setConsistencyPeriod] = useState<ConsistencyPeriod>('session');

  const getConsistencyValue = () => {
    switch (consistencyPeriod) {
      case 'session':
        return sessionConsistency;
      case 'lastWeek':
        return historicalConsistency.lastWeek;
      case 'lastMonth':
        return historicalConsistency.lastMonth;
      case 'career':
        return historicalConsistency.career;
      default:
        return sessionConsistency;
    }
  };

  // Radar chart data
  const radarData = [
    { metric: 'Elbow Flare', value: radarMetrics.elbowFlare, ideal: 85 },
    { metric: 'Trunk Lean', value: radarMetrics.trunkLean, ideal: 80 },
    { metric: 'Knee Bend', value: radarMetrics.kneeBend, ideal: 75 },
    { metric: 'Elbow Ext.', value: radarMetrics.elbowExtension, ideal: 90 },
    { metric: 'Wrist Snap', value: radarMetrics.wristSnap, ideal: 85 },
  ];

  return (
    <>
      <style>{styles}</style>
      <div className="ideal-shot">
        {/* Header */}
        <header className="is-header">
          <div>
            <h1 className="is-title">
              IDEAL <span>SHOT</span>
            </h1>
            <div className="is-player-info">
              <span>
                <span className="label">Player</span>
                {player.name}
              </span>
              <span>
                <span className="label">Height</span>
                {player.height}
              </span>
              <span>
                <span className="label">Hand</span>
                {player.handedness}
              </span>
            </div>
          </div>
          <div className="is-consistency-box">
            <div className="is-consistency-label">Session Consistency</div>
            <div className="is-consistency-value">{getConsistencyValue()}</div>
            <div className="is-consistency-toggle">
              {(['session', 'lastWeek', 'lastMonth', 'career'] as const).map((period) => (
                <button
                  key={period}
                  className={`is-toggle-btn ${consistencyPeriod === period ? 'active' : ''}`}
                  onClick={() => setConsistencyPeriod(period)}
                >
                  {period === 'lastWeek' ? 'Week' : period === 'lastMonth' ? 'Month' : period}
                </button>
              ))}
            </div>
          </div>
        </header>

        {/* Main Content */}
        <div className="is-main">
          {/* Left Column - Individual Visualizations */}
          <div className="is-left-col">
            <div className="is-metric-card">
              <div className="is-metric-title">Rhythm</div>
              <RhythmViz
                kneeTime={leftMetrics.rhythm.kneeTime}
                elbowTime={leftMetrics.rhythm.elbowTime}
                isAligned={leftMetrics.rhythm.isAligned}
              />
            </div>

            <div className="is-metric-card">
              <div className="is-metric-title">Foot Alignment</div>
              <FootAlignmentArc angleOff={leftMetrics.footAlignment.angleOff} />
            </div>

            <div className="is-metric-card">
              <div className="is-metric-title">Foot Stance</div>
              <FootStanceViz width={leftMetrics.footStance.width} />
            </div>

            <div className="is-metric-card">
              <div className="is-metric-title">Release Height</div>
              <ReleaseHeightViz
                current={leftMetrics.releaseHeight.current}
                targetMin={leftMetrics.releaseHeight.targetMin}
                targetMax={leftMetrics.releaseHeight.targetMax}
              />
            </div>
          </div>

          {/* Right Column - Radar Chart */}
          <div className="is-right-col">
            <div className="is-radar-title">Mechanics Breakdown</div>
            <div className="is-radar-container">
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="75%">
                  <PolarGrid stroke="#2A2A30" />
                  <PolarAngleAxis
                    dataKey="metric"
                    tick={{ fill: '#8B8B96', fontSize: 11 }}
                  />
                  <PolarRadiusAxis
                    angle={90}
                    domain={[0, 100]}
                    tick={{ fill: '#5A5A64', fontSize: 10 }}
                    axisLine={false}
                  />
                  {/* Ideal range - filled area */}
                  <Radar
                    name="Ideal"
                    dataKey="ideal"
                    stroke="#3DD68C"
                    fill="#3DD68C"
                    fillOpacity={0.15}
                    strokeWidth={1}
                    strokeDasharray="4 4"
                  />
                  {/* Player values */}
                  <Radar
                    name="Player"
                    dataKey="value"
                    stroke="#FF6B5B"
                    fill="#FF6B5B"
                    fillOpacity={0.3}
                    strokeWidth={2}
                  />
                </RadarChart>
              </ResponsiveContainer>
            </div>
            <div className="is-radar-legend">
              <div className="is-legend-item">
                <span className="is-legend-dot player" />
                Your Form
              </div>
              <div className="is-legend-item">
                <span className="is-legend-dot ideal" />
                Ideal Range
              </div>
            </div>
          </div>
        </div>

        {/* Bottom Section */}
        <div className="is-bottom">
          {/* Quick Tips */}
          <div>
            <h2 className="is-section-title">Quick Tips</h2>
            <ul className="is-tips-list">
              {quickTips.map((tip, index) => (
                <li key={index} className="is-tip-item">
                  <span className="is-tip-bullet" />
                  {tip}
                </li>
              ))}
            </ul>
          </div>

          {/* Work On */}
          <div>
            <h2 className="is-section-title">Work On</h2>
            <div className="is-workon-grid">
              {workOn.slice(0, 3).map((item, index) => (
                <div key={index} className="is-workon-card">
                  <div className="is-workon-title">{item.title}</div>
                  <div className="is-workon-desc">{item.description}</div>
                  <div className="is-workon-drill">
                    <div className="is-workon-drill-label">Recommended Drill</div>
                    <div className="is-workon-drill-name">{item.drillName}</div>
                    <div className="is-workon-drill-tip">{item.drillTip}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

// ============ DEMO/EXAMPLE USAGE ============
export function IdealShotDemo() {
  const demoProps: IdealShotProps = {
    player: {
      name: 'Marcus Chen',
      height: "6'2\"",
      handedness: 'Right',
    },
    sessionConsistency: 78,
    historicalConsistency: {
      career: 72,
      lastWeek: 75,
      lastMonth: 74,
    },
    radarMetrics: {
      elbowFlare: 82,
      trunkLean: 68,
      kneeBend: 71,
      elbowExtension: 88,
      wristSnap: 79,
    },
    leftMetrics: {
      rhythm: {
        kneeTime: 0.42,
        elbowTime: 0.48,
        isAligned: false,
      },
      footAlignment: {
        angleOff: 8,
      },
      footStance: {
        width: 17,
      },
      releaseHeight: {
        current: 96,
        targetMin: 92,
        targetMax: 100,
      },
    },
    quickTips: [
      'Focus on syncing your knee drive with elbow lift for smoother rhythm',
      'Square your feet to the basket before each shot',
      'Your release point is solid - maintain that consistency',
    ],
    workOn: [
      {
        title: 'Trunk Stability',
        description: 'Your torso shows slight forward lean at release, reducing shot consistency.',
        drillName: 'Wall Alignment Shots',
        drillTip: 'Stand 1ft from wall, shoot without touching. Builds upright muscle memory.',
      },
      {
        title: 'Knee-Elbow Sync',
        description: 'Your elbow extends 0.06s after knee peak, creating timing inconsistency.',
        drillName: 'Rhythm Counting',
        drillTip: 'Count "1-2-shoot" aloud. Knee bends on 1, elbow rises on 2, release on shoot.',
      },
      {
        title: 'Foot Alignment',
        description: "Feet are angled 8° off center, affecting accuracy on longer shots.",
        drillName: 'Tape Line Drill',
        drillTip: 'Place tape on floor pointing at rim. Align feet parallel before each shot.',
      },
    ],
  };

  return <IdealShot {...demoProps} />;
}
