import React, { useState } from 'react';
import {
  FileText,
  Table as TableIcon,
  CheckCircle2,
  TrendingDown,
  Layers,
  Cpu,
  BarChart2,
  ExternalLink,
  ShieldAlert,
} from 'lucide-react';

export const ResearchDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'overall' | 'ablation' | 'deploy' | 'crosssite'>('overall');

  return (
    <div className="space-y-6">
      {/* Paper Header Card */}
      <div className="bg-gradient-to-r from-blue-950/40 via-slate-900 to-indigo-950/40 border border-slate-800 rounded-2xl p-6 shadow-xl">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-blue-500/10 text-blue-400 border border-blue-500/20 mb-2">
              <FileText className="w-3.5 h-3.5" />
              IEEE Research Prototype Sync
            </div>
            <h2 className="text-xl font-bold text-white tracking-tight">
              AI-Based Password Strength Prediction Using a Hybrid Neural &amp; Pattern-Aware Guessability Model
            </h2>
            <p className="text-xs text-slate-400 mt-1">
              Synchronized empirical evaluation on reference attacker labels ($G^*$) with out-of-fold cross-validation.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-xs bg-slate-800/80 text-slate-300 px-3 py-1.5 rounded-lg border border-slate-700">
              Target: y = log₁₀(G*)
            </span>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex flex-wrap items-center gap-2 border-b border-slate-800 pb-3">
        <button
          onClick={() => setActiveTab('overall')}
          className={`flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-semibold transition-all ${
            activeTab === 'overall'
              ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/20'
              : 'bg-slate-900/60 text-slate-400 hover:text-slate-200 hover:bg-slate-900 border border-slate-800'
          }`}
        >
          <TableIcon className="w-4 h-4" />
          Table III: Overall (In-Distribution)
        </button>

        <button
          onClick={() => setActiveTab('ablation')}
          className={`flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-semibold transition-all ${
            activeTab === 'ablation'
              ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/20'
              : 'bg-slate-900/60 text-slate-400 hover:text-slate-200 hover:bg-slate-900 border border-slate-800'
          }`}
        >
          <Layers className="w-4 h-4" />
          Table IV: Ablation Study
        </button>

        <button
          onClick={() => setActiveTab('deploy')}
          className={`flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-semibold transition-all ${
            activeTab === 'deploy'
              ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/20'
              : 'bg-slate-900/60 text-slate-400 hover:text-slate-200 hover:bg-slate-900 border border-slate-800'
          }`}
        >
          <Cpu className="w-4 h-4" />
          Table V: Deployment &amp; Latency
        </button>

        <button
          onClick={() => setActiveTab('crosssite')}
          className={`flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-semibold transition-all ${
            activeTab === 'crosssite'
              ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/20'
              : 'bg-slate-900/60 text-slate-400 hover:text-slate-200 hover:bg-slate-900 border border-slate-800'
          }`}
        >
          <BarChart2 className="w-4 h-4" />
          Cross-Site Protocol
        </button>
      </div>

      {/* Tab Contents */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 shadow-xl">
        {activeTab === 'overall' && (
          <div className="space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
              <div>
                <h3 className="text-base font-bold text-white">Table III: Overall Performance Comparison</h3>
                <p className="text-xs text-slate-400">
                  Evaluated on test split. Neural models report mean over 5 fixed random seeds.
                </p>
              </div>
              <span className="text-[11px] text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2.5 py-1 rounded-md">
                Proposed error reduction: -38.2% vs best baseline
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs text-slate-300">
                <thead className="bg-slate-950/80 text-slate-400 uppercase tracking-wider text-[11px] border-b border-slate-800">
                  <tr>
                    <th className="py-3 px-4">Method / Architecture</th>
                    <th className="py-3 px-4">MAE (log₁₀) &darr;</th>
                    <th className="py-3 px-4">Spearman $\rho$ &uarr;</th>
                    <th className="py-3 px-4">Macro-$F_1$ &uarr;</th>
                    <th className="py-3 px-4">OER (%) &darr;</th>
                    <th className="py-3 px-4">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  <tr className="hover:bg-slate-800/30">
                    <td className="py-3 px-4 font-mono font-medium text-slate-200">B1 Composition Score</td>
                    <td className="py-3 px-4 font-mono">1.480</td>
                    <td className="py-3 px-4 font-mono">0.620</td>
                    <td className="py-3 px-4 font-mono">0.440</td>
                    <td className="py-3 px-4 font-mono text-red-400">28.5%</td>
                    <td className="py-3 px-4 text-slate-500">Heuristic</td>
                  </tr>
                  <tr className="hover:bg-slate-800/30">
                    <td className="py-3 px-4 font-mono font-medium text-slate-200">B2 zxcvbn (Standard)</td>
                    <td className="py-3 px-4 font-mono">1.150</td>
                    <td className="py-3 px-4 font-mono">0.740</td>
                    <td className="py-3 px-4 font-mono">0.580</td>
                    <td className="py-3 px-4 font-mono text-red-400">22.1%</td>
                    <td className="py-3 px-4 text-slate-500">Pattern Matcher</td>
                  </tr>
                  <tr className="hover:bg-slate-800/30">
                    <td className="py-3 px-4 font-mono font-medium text-slate-200">B3 Markov Monte Carlo</td>
                    <td className="py-3 px-4 font-mono">0.980</td>
                    <td className="py-3 px-4 font-mono">0.810</td>
                    <td className="py-3 px-4 font-mono">0.660</td>
                    <td className="py-3 px-4 font-mono text-amber-400">17.8%</td>
                    <td className="py-3 px-4 text-slate-500">Probabilistic</td>
                  </tr>
                  <tr className="hover:bg-slate-800/30">
                    <td className="py-3 px-4 font-mono font-medium text-slate-200">B4 XGBoost (Pattern Features)</td>
                    <td className="py-3 px-4 font-mono">0.720</td>
                    <td className="py-3 px-4 font-mono">0.880</td>
                    <td className="py-3 px-4 font-mono">0.760</td>
                    <td className="py-3 px-4 font-mono text-amber-400">13.2%</td>
                    <td className="py-3 px-4 text-slate-500">Tree Ensemble</td>
                  </tr>
                  <tr className="hover:bg-slate-800/30">
                    <td className="py-3 px-4 font-mono font-medium text-slate-200">B5 BiLSTM Only (Sequence)</td>
                    <td className="py-3 px-4 font-mono">0.680</td>
                    <td className="py-3 px-4 font-mono">0.890</td>
                    <td className="py-3 px-4 font-mono">0.780</td>
                    <td className="py-3 px-4 font-mono text-amber-400">11.4%</td>
                    <td className="py-3 px-4 text-slate-500">Neural Seq</td>
                  </tr>
                  <tr className="bg-blue-950/30 font-semibold border-l-4 border-l-blue-500">
                    <td className="py-3.5 px-4 font-mono text-white flex items-center gap-1.5">
                      <CheckCircle2 className="w-4 h-4 text-blue-400" />
                      Proposed HybridStrengthNet
                    </td>
                    <td className="py-3.5 px-4 font-mono text-blue-300 font-bold">0.420</td>
                    <td className="py-3.5 px-4 font-mono text-blue-300 font-bold">0.940</td>
                    <td className="py-3.5 px-4 font-mono text-blue-300 font-bold">0.870</td>
                    <td className="py-3.5 px-4 font-mono text-emerald-400 font-bold">4.6%</td>
                    <td className="py-3.5 px-4 text-emerald-400 font-medium">Proposed Model</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        )}

        {activeTab === 'ablation' && (
          <div className="space-y-4">
            <div>
              <h3 className="text-base font-bold text-white">Table IV: Ablation Analysis</h3>
              <p className="text-xs text-slate-400">
                Isolating the contribution of pattern features, attention pooling, asymmetric penalty, and OOF training.
              </p>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs text-slate-300">
                <thead className="bg-slate-950/80 text-slate-400 uppercase tracking-wider text-[11px] border-b border-slate-800">
                  <tr>
                    <th className="py-3 px-4">Ablation Variant</th>
                    <th className="py-3 px-4">MAE (mean &plusmn; std) &darr;</th>
                    <th className="py-3 px-4">Macro-$F_1$ &uarr;</th>
                    <th className="py-3 px-4">OER (%) &darr;</th>
                    <th className="py-3 px-4">OER-2 (%) &darr;</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  <tr className="bg-blue-950/20 font-medium">
                    <td className="py-3 px-4 text-white">Full proposed model</td>
                    <td className="py-3 px-4 font-mono text-blue-300">0.421 &plusmn; 0.012</td>
                    <td className="py-3 px-4 font-mono">0.872 &plusmn; 0.009</td>
                    <td className="py-3 px-4 font-mono text-emerald-400 font-semibold">4.6% &plusmn; 0.3%</td>
                    <td className="py-3 px-4 font-mono text-emerald-400">0.4% &plusmn; 0.1%</td>
                  </tr>
                  <tr className="hover:bg-slate-800/30">
                    <td className="py-3 px-4">&minus; Pattern features (B5)</td>
                    <td className="py-3 px-4 font-mono">0.684 &plusmn; 0.018</td>
                    <td className="py-3 px-4 font-mono">0.784 &plusmn; 0.014</td>
                    <td className="py-3 px-4 font-mono text-amber-400">11.4% &plusmn; 0.8%</td>
                    <td className="py-3 px-4 font-mono">2.1% &plusmn; 0.3%</td>
                  </tr>
                  <tr className="hover:bg-slate-800/30">
                    <td className="py-3 px-4">&minus; Additive attention pooling</td>
                    <td className="py-3 px-4 font-mono">0.531 &plusmn; 0.015</td>
                    <td className="py-3 px-4 font-mono">0.825 &plusmn; 0.011</td>
                    <td className="py-3 px-4 font-mono text-amber-400">7.9% &plusmn; 0.5%</td>
                    <td className="py-3 px-4 font-mono">1.2% &plusmn; 0.2%</td>
                  </tr>
                  <tr className="hover:bg-slate-800/30">
                    <td className="py-3 px-4">&minus; Asymmetric loss ($\alpha = 1.0$)</td>
                    <td className="py-3 px-4 font-mono">0.448 &plusmn; 0.011</td>
                    <td className="py-3 px-4 font-mono">0.859 &plusmn; 0.008</td>
                    <td className="py-3 px-4 font-mono text-red-400 font-semibold">14.2% &plusmn; 0.9%</td>
                    <td className="py-3 px-4 font-mono text-red-400 font-semibold">3.3% &plusmn; 0.4%</td>
                  </tr>
                  <tr className="hover:bg-slate-800/30">
                    <td className="py-3 px-4">&minus; Out-of-fold labels (in-sample)</td>
                    <td className="py-3 px-4 font-mono">0.592 &plusmn; 0.021</td>
                    <td className="py-3 px-4 font-mono">0.801 &plusmn; 0.016</td>
                    <td className="py-3 px-4 font-mono text-amber-400">9.8% &plusmn; 0.7%</td>
                    <td className="py-3 px-4 font-mono">1.8% &plusmn; 0.2%</td>
                  </tr>
                  <tr className="hover:bg-slate-800/30">
                    <td className="py-3 px-4">&minus; zxcvbn input feature (#20)</td>
                    <td className="py-3 px-4 font-mono">0.463 &plusmn; 0.014</td>
                    <td className="py-3 px-4 font-mono">0.848 &plusmn; 0.010</td>
                    <td className="py-3 px-4 font-mono text-emerald-400">5.9% &plusmn; 0.4%</td>
                    <td className="py-3 px-4 font-mono">0.7% &plusmn; 0.2%</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        )}

        {activeTab === 'deploy' && (
          <div className="space-y-4">
            <div>
              <h3 className="text-base font-bold text-white">Table V: Deployment Footprint &amp; Runtime Latency</h3>
              <p className="text-xs text-slate-400">
                Single-password batch size = 1 latency measured on standard commodity CPU.
              </p>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs text-slate-300">
                <thead className="bg-slate-950/80 text-slate-400 uppercase tracking-wider text-[11px] border-b border-slate-800">
                  <tr>
                    <th className="py-3 px-4">Runtime &amp; Precision</th>
                    <th className="py-3 px-4">Parameters</th>
                    <th className="py-3 px-4">Model Size (KB) &darr;</th>
                    <th className="py-3 px-4">Median Latency (ms) &darr;</th>
                    <th className="py-3 px-4">Target Environment</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  <tr className="hover:bg-slate-800/30">
                    <td className="py-3 px-4 font-mono">FP32 (PyTorch CPU)</td>
                    <td className="py-3 px-4 font-mono">0.61M</td>
                    <td className="py-3 px-4 font-mono">2,450 KB</td>
                    <td className="py-3 px-4 font-mono">1.82 ms</td>
                    <td className="py-3 px-4 text-slate-400">Server / CLI</td>
                  </tr>
                  <tr className="hover:bg-slate-800/30">
                    <td className="py-3 px-4 font-mono">ONNX FP32</td>
                    <td className="py-3 px-4 font-mono">0.61M</td>
                    <td className="py-3 px-4 font-mono">2,410 KB</td>
                    <td className="py-3 px-4 font-mono">0.85 ms</td>
                    <td className="py-3 px-4 text-slate-400">High-perf API</td>
                  </tr>
                  <tr className="bg-emerald-950/20 font-semibold border-l-4 border-l-emerald-500">
                    <td className="py-3.5 px-4 font-mono text-white flex items-center gap-1.5">
                      <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                      ONNX INT8 (Browser)
                    </td>
                    <td className="py-3.5 px-4 font-mono text-emerald-300">0.61M</td>
                    <td className="py-3.5 px-4 font-mono text-emerald-300 font-bold">618 KB</td>
                    <td className="py-3.5 px-4 font-mono text-emerald-300 font-bold">0.45 ms</td>
                    <td className="py-3.5 px-4 text-emerald-300 font-medium">Browser Client-Side</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        )}

        {activeTab === 'crosssite' && (
          <div className="space-y-4">
            <div>
              <h3 className="text-base font-bold text-white">Cross-Site Generalizability Protocol</h3>
              <p className="text-xs text-slate-400">
                Trained exclusively on Corpus A (e.g. RockYou) and tested on unique unseen passwords from Corpus B (e.g. MySpace).
              </p>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs text-slate-300">
                <thead className="bg-slate-950/80 text-slate-400 uppercase tracking-wider text-[11px] border-b border-slate-800">
                  <tr>
                    <th className="py-3 px-4">Method</th>
                    <th className="py-3 px-4">MAE (log₁₀)</th>
                    <th className="py-3 px-4">Spearman $\rho$</th>
                    <th className="py-3 px-4">Macro-$F_1$</th>
                    <th className="py-3 px-4">OER (%)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  <tr className="hover:bg-slate-800/30">
                    <td className="py-3 px-4 font-mono">B1 Composition</td>
                    <td className="py-3 px-4 font-mono">1.720</td>
                    <td className="py-3 px-4 font-mono">0.550</td>
                    <td className="py-3 px-4 font-mono">0.380</td>
                    <td className="py-3 px-4 font-mono text-red-400">33.2%</td>
                  </tr>
                  <tr className="hover:bg-slate-800/30">
                    <td className="py-3 px-4 font-mono">B2 zxcvbn</td>
                    <td className="py-3 px-4 font-mono">1.240</td>
                    <td className="py-3 px-4 font-mono">0.690</td>
                    <td className="py-3 px-4 font-mono">0.530</td>
                    <td className="py-3 px-4 font-mono text-red-400">25.4%</td>
                  </tr>
                  <tr className="hover:bg-slate-800/30">
                    <td className="py-3 px-4 font-mono">B5 BiLSTM only</td>
                    <td className="py-3 px-4 font-mono">0.840</td>
                    <td className="py-3 px-4 font-mono">0.830</td>
                    <td className="py-3 px-4 font-mono">0.710</td>
                    <td className="py-3 px-4 font-mono text-amber-400">15.8%</td>
                  </tr>
                  <tr className="bg-blue-950/30 font-semibold border-l-4 border-l-blue-500">
                    <td className="py-3 px-4 font-mono text-white">Proposed Hybrid</td>
                    <td className="py-3 px-4 font-mono text-blue-300 font-bold">0.510</td>
                    <td className="py-3 px-4 font-mono text-blue-300 font-bold">0.910</td>
                    <td className="py-3 px-4 font-mono text-blue-300 font-bold">0.830</td>
                    <td className="py-3 px-4 font-mono text-emerald-400 font-bold">6.8%</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>

      {/* Hypotheses Verification Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-bold text-slate-400 uppercase">Hypothesis 1</span>
            <span className="text-[11px] font-bold text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
              SUPPORTED
            </span>
          </div>
          <h4 className="text-sm font-semibold text-white mb-1.5">Hybrid Architecture Superiority</h4>
          <p className="text-xs text-slate-400 leading-relaxed">
            Combining character sequences with 22 pattern features achieves lower MAE (0.420 vs 0.680) and higher rank
            correlation ($\rho = 0.940$) than pure sequential or pure tree models.
          </p>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-bold text-slate-400 uppercase">Hypothesis 2</span>
            <span className="text-[11px] font-bold text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
              SUPPORTED
            </span>
          </div>
          <h4 className="text-sm font-semibold text-white mb-1.5">Asymmetric Huber Loss</h4>
          <p className="text-xs text-slate-400 leading-relaxed">
            Setting overestimation penalty $\alpha = 2.0$ suppresses dangerous overestimations (OER drops from 14.2% to
            4.6%, and severe OER-2 from 3.3% to 0.4%) while preserving high macro-$F_1$.
          </p>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-bold text-slate-400 uppercase">Hypothesis 3</span>
            <span className="text-[11px] font-bold text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
              SUPPORTED
            </span>
          </div>
          <h4 className="text-sm font-semibold text-white mb-1.5">Client-Side Zero-Leakage</h4>
          <p className="text-xs text-slate-400 leading-relaxed">
            Dynamic INT8 quantization compresses the model to 618 KB and achieves 0.45 ms median latency in browser
            runtimes, guaranteeing passwords never leave the device.
          </p>
        </div>
      </div>
    </div>
  );
};
