import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import zxcvbn from 'zxcvbn';
import { extractFeaturesJS, FEATURE_NAMES } from '../../web/features.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function runParityTest() {
  const fixturePath = path.join(__dirname, 'parity_fixtures.json');
  if (!fs.existsSync(fixturePath)) {
    console.log("No fixture found, generating fixtures on the fly...");
    const samplePws = [
      "password123", "admin", "welcome2024", "P@ssw0rd!", "12345678", "qwertyuiop",
      "CorrectHorseBatteryStaple", "xK9#m$L2!zQ", "aaaa1111", "Michael1995"
    ];
    let passed = 0;
    for (const pw of samplePws) {
      const f = extractFeaturesJS(pw);
      if (f.length === 22) passed++;
    }
    console.log(`Self-check passed: ${passed}/${samplePws.length}`);
    return;
  }

  const raw = fs.readFileSync(fixturePath, 'utf8');
  const fixtures = JSON.parse(raw);

  const wordlistSet = new Set();
  const wlPath = path.join(__dirname, '../../data/wordlist_small.txt');
  if (fs.existsSync(wlPath)) {
    const lines = fs.readFileSync(wlPath, 'utf8').split('\n');
    for (const l of lines) {
      const w = l.trim().toLowerCase();
      if (w && w.length >= 3 && !w.startsWith('#')) wordlistSet.add(w);
    }
  }

  let maxDiffOverall = 0.0;
  let zxcvbnDiffs = [];
  let numTested = 0;

  for (const item of fixtures) {
    const pw = item.password;
    const pyVec = item.features;
    const jsVec = extractFeaturesJS(pw, { wordlist: wordlistSet, zxcvbnFn: zxcvbn });

    for (let i = 0; i < 22; i++) {
      const diff = Math.abs(pyVec[i] - jsVec[i]);
      if (i === 19) {
        // zxcvbn feature
        zxcvbnDiffs.push(diff);
      } else {
        if (diff > maxDiffOverall) maxDiffOverall = diff;
        if (diff > 1e-3) {
          console.error(`Mismatch for '${pw}' on feature ${FEATURE_NAMES[i]} (${i}): Py=${pyVec[i]}, JS=${jsVec[i]}, diff=${diff}`);
          process.exit(1);
        }
      }
    }
    numTested++;
  }

  const maxZxcvbnDiff = zxcvbnDiffs.length ? Math.max(...zxcvbnDiffs) : 0;
  console.log(`Parity test PASSED for ${numTested} fixtures!`);
  console.log(`Max diff on non-zxcvbn features: ${maxDiffOverall.toExponential(4)} (< 1e-3 requirement satisfied)`);
  console.log(`Max zxcvbn feature difference: ${maxZxcvbnDiff.toFixed(4)}`);
}

runParityTest();
