import React, { useState, useMemo } from 'react';
import {
  Eye,
  EyeOff,
  Shield,
  ShieldAlert,
  ShieldCheck,
  AlertTriangle,
  Info,
  Clock,
  ChevronDown,
  ChevronUp,
  Cpu,
  Lock,
  Layers,
  Sparkles,
  Zap,
} from 'lucide-react';
import { evaluatePassword, LEVEL_NAMES, LEVEL_COLORS, StrengthResult } from '../utils/strength';
// @ts-ignore
import { FEATURE_NAMES } from '../../web/features.js';

interface PasswordMeterProps {
  onSelectSample?: (pwd: string) => void;
}

const PRESET_PASSWORDS = [
  { label: 'Weak: Walk & Year', pwd: 'qwerty1995', type: 'weak' },
  { label: 'Weak: L33t Sub', pwd: 'P@ssw0rd!', type: 'weak' },
  { label: 'Breach Blocklist', pwd: 'password123', type: 'block' },
  { label: 'Moderate: Name+Num', pwd: 'Michael#2024', type: 'moderate' },
  { label: 'Strong: Passphrase', pwd: 'correct-horse-battery-staple', type: 'strong' },
  { label: 'Strong: High-Entropy', pwd: 'xK9#m$L2!zQ8*vR', type: 'strong' },
];

export const PasswordMeter: React.FC<PasswordMeterProps> = () => {
  const [password, setPassword] = useState('P@ssw0rd2024!');
  const [showPassword, setShowPassword] = useState(false);
  const [showFeatureDrawer, setShowFeatureDrawer] = useState(false);

  const result: StrengthResult = useMemo(() => {
    return evaluatePassword(password);
  }, [password]);

  const levelColor = LEVEL_COLORS[result.displayedLevel] || '#64748b';

  return (
    <div className="space-y-6">
      {/* Input Card */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 shadow-xl backdrop-blur-sm">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-4">
          <div>
            <h2 className="text-xl font-bold text-slate-100 flex items-center gap-2">
              <Lock className="w-5 h-5 text-blue-500" />
              Interactive Guessability Meter
            </h2>
            <p className="text-xs text-slate-400">
              Evaluates true adversarial guess work $G^*$ and runs real-time occlusion attribution.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              <ShieldCheck className="w-3.5 h-3.5" />
              100% Client-Side Private
            </span>
          </div>
        </div>

        {/* Input Field with Show/Hide */}
        <div className="relative mb-3">
          <input
            type={showPassword ? 'text' : 'password'}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Type a password to test..."
            className="w-full bg-slate-950/80 border border-slate-700/80 rounded-xl px-4 py-3.5 pr-28 text-slate-100 font-mono text-base placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500 transition-all"
            autoComplete="off"
            spellCheck="false"
          />
          <div className="absolute right-2.5 top-1/2 -translate-y-1/2 flex items-center gap-1">
            {password && (
              <button
                type="button"
                onClick={() => setPassword('')}
                className="px-2 py-1 text-xs text-slate-400 hover:text-slate-200 transition-colors"
                title="Clear input"
              >
                Clear
              </button>
            )}
            <button
              type="button"
              onClick={() => setShowPassword(!showPassword)}
              className="p-1.5 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
              aria-label={showPassword ? 'Hide password' : 'Show password'}
            >
              {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
            </button>
          </div>
        </div>

        {/* Quick Presets */}
        <div className="flex flex-wrap items-center gap-2 mb-6">
          <span className="text-xs text-slate-400 flex items-center gap-1 mr-1">
            <Sparkles className="w-3 h-3 text-amber-400" /> Test Presets:
          </span>
          {PRESET_PASSWORDS.map((preset) => (
            <button
              key={preset.label}
              type="button"
              onClick={() => setPassword(preset.pwd)}
              className="text-xs px-2.5 py-1 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-slate-300 hover:text-white border border-slate-700/50 transition-colors"
            >
              {preset.label}
            </button>
          ))}
        </div>

        {/* 5-Segment Strength Bar */}
        <div className="space-y-2 mb-6">
          <div className="grid grid-cols-5 gap-2">
            {[0, 1, 2, 3, 4].map((idx) => {
              const isActive = password.length > 0 && idx <= result.displayedLevel;
              return (
                <div
                  key={idx}
                  className="h-2.5 rounded-full transition-all duration-300"
                  style={{
                    backgroundColor: isActive ? levelColor : '#1e293b',
                    boxShadow: isActive ? `0 0 10px ${levelColor}40` : 'none',
                  }}
                />
              );
            })}
          </div>

          <div className="flex flex-wrap items-center justify-between gap-2 pt-1">
            <div className="flex items-center gap-3">
              <span
                className="px-3 py-1 rounded-md text-xs font-bold uppercase tracking-wider transition-colors shadow-sm"
                style={{
                  backgroundColor: result.displayedLevel === 2 ? '#eab308' : levelColor,
                  color: result.displayedLevel === 2 ? '#000000' : '#ffffff',
                }}
              >
                Level {result.displayedLevel}: {result.levelName}
              </span>
              <span className="text-xs text-slate-400 font-mono">
                {'y = log₁₀(G*) ≈'} <strong className="text-slate-200">{result.yHat.toFixed(2)}</strong>
              </span>
            </div>

            <div className="text-xs font-mono text-slate-300 bg-slate-950/60 px-2.5 py-1 rounded-lg border border-slate-800">
              Estimated effort: ~<strong>10<sup>{result.yHat.toFixed(1)}</sup></strong> guesses
            </div>
          </div>
        </div>

        {/* Conservative Floor Policy Box */}
        <div
          className={`p-3.5 rounded-xl border text-xs leading-relaxed transition-all mb-6 ${
            result.isBlocklisted
              ? 'bg-red-950/20 border-red-500/40 text-red-300'
              : result.rawModelCls > result.zxcvbnScore
              ? 'bg-amber-950/20 border-amber-500/40 text-amber-200'
              : 'bg-blue-950/20 border-blue-500/30 text-blue-200'
          }`}
        >
          <div className="flex items-start gap-2.5">
            {result.isBlocklisted ? (
              <ShieldAlert className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
            ) : result.rawModelCls > result.zxcvbnScore ? (
              <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
            ) : (
              <Shield className="w-4 h-4 text-blue-400 shrink-0 mt-0.5" />
            )}
            <div>
              <div className="font-semibold mb-0.5">
                {result.isBlocklisted
                  ? 'Breach Blocklist Override Engaged'
                  : result.rawModelCls > result.zxcvbnScore
                  ? 'Conservative Floor Protection Active'
                  : 'Multi-Model Consensus'}
              </div>
              <div>
                {result.isBlocklisted ? (
                  <span>
                    This password appears in known breach corpora (e.g. RockYou top list). To avoid dangerous
                    false security, the meter strictly pins the displayed score to <strong>Level 0 (Very Weak)</strong>.
                  </span>
                ) : result.rawModelCls > result.zxcvbnScore ? (
                  <span>
                    Model predicted Level {result.rawModelCls}, but reference <code>zxcvbn</code> scored Level{' '}
                    {result.zxcvbnScore}. Under the paper's conservative policy (min(Model, zxcvbn)),
                    displayed score is <strong>Level {result.displayedLevel}</strong> to prevent overestimation.
                  </span>
                ) : (
                  <span>
                    Model prediction (Level {result.rawModelCls}) is bounded conservatively by reference baseline (Level{' '}
                    {result.zxcvbnScore}).
                  </span>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Why this score? (Key Indicators) */}
        <div className="mb-6">
          <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2.5 flex items-center gap-1.5">
            <Info className="w-3.5 h-3.5 text-blue-400" /> Top Structural Indicators
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
            {result.reasons.map((reason, i) => (
              <div
                key={i}
                className="bg-slate-950/50 border border-slate-800 rounded-lg p-2.5 text-xs text-slate-300 flex items-start gap-2"
              >
                <span className="w-1.5 h-1.5 rounded-full bg-blue-400 mt-1.5 shrink-0" />
                <span>{reason}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Which characters hurt? (Occlusion Strip) */}
        <div className="mb-4">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
              <Zap className="w-3.5 h-3.5 text-amber-400" />
              Which characters hurt? (Occlusion Sensitivity Strip)
            </h3>
            <span className="text-[11px] text-slate-500">
              Red/Amber = removing raises score (weak pattern)
            </span>
          </div>

          <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 flex flex-wrap gap-1.5 min-h-[52px] items-center">
            {result.occlusion.length > 0 ? (
              result.occlusion.map((item, idx) => {
                let badgeClass = 'bg-slate-800 text-slate-200 border-slate-700';
                if (item.severity === 'hurts-most') {
                  badgeClass = 'bg-red-950/80 text-red-200 border-red-500 shadow-sm shadow-red-900/50';
                } else if (item.severity === 'hurts-moderate') {
                  badgeClass = 'bg-amber-950/80 text-amber-200 border-amber-500/80';
                }

                return (
                  <div
                    key={idx}
                    title={item.tooltip}
                    className={`group relative flex flex-col items-center justify-center min-w-[28px] h-9 px-1.5 rounded border font-mono text-sm cursor-help transition-transform hover:scale-110 ${badgeClass}`}
                  >
                    <span>{item.char}</span>
                    <span className="text-[9px] opacity-70">
                      {item.delta > 0 ? `+${item.delta.toFixed(1)}` : item.delta.toFixed(1)}
                    </span>
                  </div>
                );
              })
            ) : (
              <span className="text-xs text-slate-500 italic px-1">
                Type at least 2 characters to activate sensitivity analysis strip.
              </span>
            )}
          </div>
        </div>

        {/* Drawer Toggle: 22-Feature Decomposition */}
        <div className="pt-2">
          <button
            type="button"
            onClick={() => setShowFeatureDrawer(!showFeatureDrawer)}
            className="w-full flex items-center justify-between px-4 py-2.5 rounded-xl bg-slate-950/60 hover:bg-slate-950 border border-slate-800 text-xs font-medium text-slate-300 transition-colors"
          >
            <span className="flex items-center gap-2">
              <Layers className="w-4 h-4 text-blue-400" />
              Inspect 22-Dimensional Extracted Feature Vector (Paper Section IV-B)
            </span>
            {showFeatureDrawer ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>

          {showFeatureDrawer && (
            <div className="mt-3 p-4 bg-slate-950 rounded-xl border border-slate-800 space-y-3 animate-fadeIn">
              <div className="text-xs text-slate-400">
                These 22 deterministic features feed into the model&apos;s Feature MLP branch $[B, 64]$ alongside the
                BiLSTM token sequence branch.
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2.5">
                {FEATURE_NAMES.map((name: string, i: number) => {
                  const val = result.features[i] ?? 0;
                  const displayVal = Number.isInteger(val) ? val : val.toFixed(3);
                  return (
                    <div key={name} className="p-2 rounded-lg bg-slate-900 border border-slate-800/80">
                      <div className="text-[10px] text-slate-400 truncate" title={name}>
                        #{i + 1} {name}
                      </div>
                      <div className="font-mono text-xs font-bold text-slate-100 mt-0.5">{displayVal}</div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {/* Footer Info */}
        <div className="mt-6 pt-4 border-t border-slate-800/80 flex flex-col sm:flex-row items-center justify-between gap-2 text-[11px] text-slate-500">
          <div className="flex items-center gap-2">
            <Cpu className="w-3.5 h-3.5 text-blue-400" />
            <span>Architecture: HybridStrengthNet (BiLSTM + Additive Attention + Feature MLP)</span>
          </div>
          <div className="flex items-center gap-1.5 font-mono">
            <Clock className="w-3.5 h-3.5 text-slate-400" />
            <span>Client Inference: {result.latencyMs.toFixed(2)} ms</span>
          </div>
        </div>
      </div>
    </div>
  );
};
