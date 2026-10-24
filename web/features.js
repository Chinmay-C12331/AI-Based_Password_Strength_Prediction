/**
 * features.js: Client-side JavaScript implementation of features.py
 * Exactly mirrors the 22-dimensional feature extraction and character encoding.
 */

const FEATURE_NAMES = [
  "length",
  "count_lower",
  "count_upper",
  "count_digit",
  "count_symbol",
  "ratio_lower",
  "ratio_upper",
  "ratio_digit",
  "ratio_symbol",
  "distinct_ratio",
  "shannon_entropy",
  "longest_repeat_run",
  "longest_sequential_run",
  "longest_keyboard_walk",
  "has_year",
  "has_date",
  "dict_word_coverage",
  "leet_normalized_changed",
  "is_in_training_top10k",
  "zxcvbn_guesses_log10",
  "trailing_digits_count",
  "is_capitalized_word",
];

const QWERTY_ROWS = [
  "`1234567890-=",
  "qwertyuiop[]\\",
  "asdfghjkl;'",
  "zxcvbnm,./",
];
const QWERTY_COLUMNS = [
  "1qaz", "2wsx", "3edc", "4rfv", "5tgb", "6yhn", "7ujm", "8ik,", "9ol.", "0p;/",
];

const KEYBOARD_ADJACENT = new Set();
for (const row of QWERTY_ROWS) {
  for (let i = 0; i < row.length - 1; i++) {
    KEYBOARD_ADJACENT.add(row[i] + row[i + 1]);
    KEYBOARD_ADJACENT.add(row[i + 1] + row[i]);
  }
}
for (const col of QWERTY_COLUMNS) {
  for (let i = 0; i < col.length - 1; i++) {
    KEYBOARD_ADJACENT.add(col[i] + col[i + 1]);
    KEYBOARD_ADJACENT.add(col[i + 1] + col[i]);
  }
}

const LEET_MAP = {
  "@": "a",
  "0": "o",
  "1": "l",
  "3": "e",
  "$": "s",
  "5": "s",
  "7": "t",
};

const YEAR_REGEX = /(?:19|20)\d{2}/;
const DATE_REGEX = /(?:\b(?:0[1-9]|[12]\d|3[01])(?:0[1-9]|1[0-2])(?:\d{2}|\d{4})\b)|(?:\b(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])(?:\d{2}|\d{4})\b)/;

function applyLeetNorm(s) {
  let out = "";
  for (const c of s) {
    out += LEET_MAP[c] || c;
  }
  return out.toLowerCase();
}

function computeLongestRepeatRun(s) {
  if (!s) return 0;
  let maxRun = 1, curRun = 1;
  for (let i = 1; i < s.length; i++) {
    if (s[i] === s[i - 1]) {
      curRun++;
      if (curRun > maxRun) maxRun = curRun;
    } else {
      curRun = 1;
    }
  }
  return maxRun;
}

function computeLongestSequentialRun(s) {
  if (s.length < 2) return s.length;
  let maxLen = 1, curInc = 1, curDec = 1;
  const lower = s.toLowerCase();
  for (let i = 1; i < lower.length; i++) {
    const diff = lower.charCodeAt(i) - lower.charCodeAt(i - 1);
    if (diff === 1) {
      curInc++;
      curDec = 1;
    } else if (diff === -1) {
      curDec++;
      curInc = 1;
    } else {
      curInc = 1;
      curDec = 1;
    }
    maxLen = Math.max(maxLen, curInc, curDec);
  }
  return maxLen;
}

function computeLongestKeyboardWalk(s) {
  if (s.length < 2) return s.length;
  const lower = s.toLowerCase();
  let maxWalk = 1, curWalk = 1;
  for (let i = 1; i < lower.length; i++) {
    const pair = lower[i - 1] + lower[i];
    if (KEYBOARD_ADJACENT.has(pair)) {
      curWalk++;
      if (curWalk > maxWalk) maxWalk = curWalk;
    } else {
      curWalk = 1;
    }
  }
  return maxWalk;
}

function computeDictionaryCoverage(normPw, wordSet) {
  if (!normPw || !wordSet || wordSet.size === 0) return 0.0;
  const n = normPw.length;
  const covered = new Array(n).fill(false);
  for (let len = 3; len <= Math.min(n, 16); len++) {
    for (let start = 0; start <= n - len; start++) {
      const sub = normPw.substring(start, start + len);
      if (wordSet.has(sub)) {
        for (let k = start; k < start + len; k++) {
          covered[k] = true;
        }
      }
    }
  }
  const count = covered.filter(Boolean).length;
  return count / n;
}

function extractFeaturesJS(pw, options = {}) {
  const { blocklist = new Set(), wordlist = new Set(), zxcvbnFn = null } = options;
  const len = pw.length;
  if (len === 0) return new Float32Array(22);

  const f1 = len;
  let cLower = 0, cUpper = 0, cDigit = 0, cSymbol = 0;
  const charCounts = {};

  for (let i = 0; i < len; i++) {
    const ch = pw[i];
    const code = pw.charCodeAt(i);
    charCounts[ch] = (charCounts[ch] || 0) + 1;
    if (code >= 97 && code <= 122) cLower++;
    else if (code >= 65 && code <= 90) cUpper++;
    else if (code >= 48 && code <= 57) cDigit++;
    else cSymbol++;
  }

  const f2 = cLower, f3 = cUpper, f4 = cDigit, f5 = cSymbol;
  const f6 = cLower / f1, f7 = cUpper / f1, f8 = cDigit / f1, f9 = cSymbol / f1;
  const distinctCount = Object.keys(charCounts).length;
  const f10 = distinctCount / f1;

  let entropy = 0;
  for (const c in charCounts) {
    const p = charCounts[c] / f1;
    entropy -= p * Math.log2(p);
  }
  const f11 = entropy;
  const f12 = computeLongestRepeatRun(pw);
  const f13 = computeLongestSequentialRun(pw);
  const f14 = computeLongestKeyboardWalk(pw);
  const f15 = YEAR_REGEX.test(pw) ? 1.0 : 0.0;
  const f16 = DATE_REGEX.test(pw) ? 1.0 : 0.0;

  const leetNorm = applyLeetNorm(pw);
  const f17 = computeDictionaryCoverage(leetNorm, wordlist);
  const f18 = (leetNorm !== pw.toLowerCase()) ? 1.0 : 0.0;
  const f19 = blocklist.has(pw) ? 1.0 : 0.0;

  let f20 = 0.0;
  if (typeof zxcvbnFn === "function") {
    try {
      const res = zxcvbnFn(pw.slice(0, 32));
      f20 = res && typeof res.guesses_log10 === "number" ? res.guesses_log10 : 0.0;
    } catch {
      f20 = Math.log10(Math.max(1.0, Math.pow(distinctCount, Math.min(len, 12))));
    }
  } else if (typeof window !== "undefined" && typeof window.zxcvbn === "function") {
    try {
      const res = window.zxcvbn(pw.slice(0, 32));
      f20 = res && typeof res.guesses_log10 === "number" ? res.guesses_log10 : 0.0;
    } catch {
      f20 = Math.log10(Math.max(1.0, Math.pow(distinctCount, Math.min(len, 12))));
    }
  }

  let trailing = 0;
  for (let i = len - 1; i >= 0; i--) {
    const code = pw.charCodeAt(i);
    if (code >= 48 && code <= 57) trailing++;
    else break;
  }
  const f21 = trailing;
  const f22 = /^[A-Z][a-z]+$/.test(pw) ? 1.0 : 0.0;

  return new Float32Array([
    f1, f2, f3, f4, f5, f6, f7, f8, f9, f10,
    f11, f12, f13, f14, f15, f16, f17, f18, f19, f20,
    f21, f22,
  ]);
}

function encodeCharsJS(pw, maxLen = 32) {
  const ids = new BigInt64Array(maxLen);
  const trunc = pw.slice(0, maxLen);
  for (let i = 0; i < trunc.length; i++) {
    const code = trunc.charCodeAt(i);
    if (code >= 32 && code <= 126) {
      ids[i] = BigInt(code - 32 + 2);
    } else {
      ids[i] = 1n; // UNK
    }
  }
  return ids;
}

export {
  FEATURE_NAMES,
  extractFeaturesJS,
  encodeCharsJS,
  computeLongestKeyboardWalk,
  computeLongestRepeatRun,
  computeLongestSequentialRun,
};

if (typeof globalThis !== "undefined" && globalThis.module && globalThis.module.exports) {
  globalThis.module.exports = {
    FEATURE_NAMES,
    extractFeaturesJS,
    encodeCharsJS,
    computeLongestKeyboardWalk,
    computeLongestRepeatRun,
    computeLongestSequentialRun,
  };
}
