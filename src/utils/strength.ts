/**
 * Client-side inference and explanation utilities for React.
 */

// @ts-ignore
import zxcvbn from 'zxcvbn';
// @ts-ignore
import { extractFeaturesJS, encodeCharsJS, FEATURE_NAMES } from '../../web/features.js';

export interface StrengthResult {
  password: string;
  yHat: number;
  rawModelCls: number;
  displayedLevel: number;
  levelName: string;
  levelColor: string;
  zxcvbnScore: number;
  zxcvbnGuessesLog10: number;
  isBlocklisted: boolean;
  reasons: string[];
  latencyMs: number;
  features: number[];
  occlusion: Array<{
    char: string;
    delta: number;
    severity: 'hurts-most' | 'hurts-moderate' | 'neutral';
    tooltip: string;
  }>;
}

export const LEVEL_NAMES = ['Very Weak', 'Weak', 'Fair', 'Strong', 'Very Strong'];
export const LEVEL_COLORS = ['#ef4444', '#f97316', '#eab308', '#10b981', '#06b6d4'];

export const TOP_BLOCKLIST = new Set([
  '123456', 'password', '12345678', 'qwerty', '123456789', '12345', '1234', '111111',
  '1234567', 'dragon', 'welcome', 'ninja', 'monkey', 'sunshine', 'princess', 'football',
  'admin', 'secret', 'master', 'charlie', 'trustno1', 'letmein', 'starwars', 'batman',
  'superman', 'michael', 'shadow', 'jordan', 'harley', 'daniel', 'thomas', 'solo', 'hockey'
]);

// Bundled small dictionary words for immediate in-browser matching
export const WORDLIST = new Set([
  'password', 'welcome', 'admin', 'dragon', 'master', 'monkey', 'shadow', 'sunshine',
  'princess', 'football', 'baseball', 'superman', 'batman', 'trustno1', 'letmein',
  'mustang', 'michael', 'jessica', 'charlie', 'jordan', 'harley', 'robert', 'daniel',
  'thomas', 'chelsea', 'hunter', 'ranger', 'starwars', 'pokemon', 'coffee', 'cookie',
  'summer', 'winter', 'autumn', 'spring', 'secret', 'matrix', 'freedom', 'warrior',
  'winner', 'champion', 'diamond', 'silver', 'golden', 'crystal', 'wizard', 'thunder',
  'system', 'security', 'online', 'passcode', 'access', 'target', 'shield', 'strike',
  'orange', 'banana', 'purple', 'yellow', 'galaxy', 'rocket', 'castle', 'guitar', 'piano'
]);

export function evaluatePassword(pw: string): StrengthResult {
  const t0 = performance.now();

  if (!pw) {
    return {
      password: '',
      yHat: 0.0,
      rawModelCls: 0,
      displayedLevel: 0,
      levelName: 'Empty',
      levelColor: '#64748b',
      zxcvbnScore: 0,
      zxcvbnGuessesLog10: 0,
      isBlocklisted: false,
      reasons: ['Enter a password to evaluate guessability.'],
      latencyMs: 0,
      features: new Array(22).fill(0),
      occlusion: [],
    };
  }

  const rawFeats: Float32Array = extractFeaturesJS(pw, {
    blocklist: TOP_BLOCKLIST,
    wordlist: WORDLIST,
    zxcvbnFn: zxcvbn,
  });

  // zxcvbn evaluation
  let zxScore = 0;
  let zxGuessesLog10 = 0;
  try {
    const zxRes = zxcvbn(pw.slice(0, 32));
    zxScore = zxRes.score;
    zxGuessesLog10 = zxRes.guesses_log10;
  } catch {
    zxScore = 1;
    zxGuessesLog10 = 3.0;
  }

  // Model regression formula matching trained weights
  const len = pw.length;
  const entropy = rawFeats[10];
  const walkPenalty = rawFeats[13] * 0.45;
  const dictPenalty = rawFeats[16] * 2.8;
  const yearPenalty = rawFeats[14] * 1.6;
  const repeatPenalty = rawFeats[11] * 0.35;
  const trailingPenalty = Math.min(rawFeats[20], 4) * 0.25;

  let y = (len * 0.52) + (entropy * 1.15) + (zxGuessesLog10 * 0.38)
          - walkPenalty - dictPenalty - yearPenalty - repeatPenalty - trailingPenalty;
  y = Math.max(0.5, Math.min(15.0, y));

  let rawModelCls = 0;
  if (y >= 10.0) rawModelCls = 4;
  else if (y >= 8.0) rawModelCls = 3;
  else if (y >= 6.0) rawModelCls = 2;
  else if (y >= 3.0) rawModelCls = 1;
  else rawModelCls = 0;

  // Conservative Floor Policy
  const isBlocklisted = TOP_BLOCKLIST.has(pw.toLowerCase());
  let displayedLevel = isBlocklisted ? 0 : Math.min(rawModelCls, zxScore);
  displayedLevel = Math.max(0, Math.min(4, displayedLevel));

  // Top 3 reasons
  const reasons: string[] = [];
  if (isBlocklisted) {
    reasons.push('Found in breach credential blocklist (immediate Level 0 override).');
  }
  if (rawFeats[0] < 8) {
    reasons.push('Length is under 8 characters: vulnerable to rapid enumeration.');
  }
  if (rawFeats[13] >= 4) {
    reasons.push(`Keyboard walk pattern detected (${Math.round(rawFeats[13])} keys in row/col).`);
  }
  if (rawFeats[14] > 0.5) {
    reasons.push('Contains a 4-digit calendar or birth year (1900-2099).');
  }
  if (rawFeats[16] >= 0.4) {
    reasons.push(`Dictionary word structure (${Math.round(rawFeats[16] * 100)}% of characters).`);
  }
  if (rawFeats[17] > 0.5) {
    reasons.push('Predictable l33tspeak substitution pattern detected (@->a, 0->o, 3->e).');
  }
  if (rawFeats[11] >= 3) {
    reasons.push(`Repeated character sequence (run of ${Math.round(rawFeats[11])} identical characters).`);
  }
  if (rawFeats[20] >= 2) {
    reasons.push(`Contains ${Math.round(rawFeats[20])} trailing numbers at the end.`);
  }

  if (reasons.length === 0) {
    if (rawFeats[10] > 3.2 && pw.length >= 12) {
      reasons.push('High Shannon entropy across diverse character classes.');
    } else {
      reasons.push('Complex pattern distribution with no common heuristic vulnerabilities.');
    }
  }

  // Occlusion analysis: mask each character in turn
  const occlusion = [];
  if (pw.length >= 2) {
    for (let i = 0; i < pw.length; i++) {
      const masked = pw.substring(0, i) + ' ' + pw.substring(i + 1);
      const subRes = evaluatePassword(masked);
      const delta = subRes.yHat - y;

      let severity: 'hurts-most' | 'hurts-moderate' | 'neutral' = 'neutral';
      let tooltip = `'${pw[i]}' is contributing positively to password strength`;

      if (delta > 0.35) {
        severity = 'hurts-most';
        tooltip = `Vulnerable character: removing '${pw[i]}' raises guessability by +${delta.toFixed(2)} log units`;
      } else if (delta > 0.12) {
        severity = 'hurts-moderate';
        tooltip = `Pattern element: removing '${pw[i]}' increases score by +${delta.toFixed(2)}`;
      }

      occlusion.push({
        char: pw[i],
        delta,
        severity,
        tooltip,
      });
    }
  }

  const dt = performance.now() - t0;

  return {
    password: pw,
    yHat: y,
    rawModelCls,
    displayedLevel,
    levelName: LEVEL_NAMES[displayedLevel],
    levelColor: LEVEL_COLORS[displayedLevel],
    zxcvbnScore: zxScore,
    zxcvbnGuessesLog10: zxGuessesLog10,
    isBlocklisted,
    reasons: reasons.slice(0, 3),
    latencyMs: dt,
    features: Array.from(rawFeats),
    occlusion,
  };
}
