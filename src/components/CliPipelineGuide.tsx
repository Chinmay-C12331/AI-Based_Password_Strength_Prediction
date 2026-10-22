import React from 'react';
import { Terminal, Shield, Play, CheckCircle, Database, AlertCircle, Copy } from 'lucide-react';

export const CliPipelineGuide: React.FC = () => {
  const [copied, setCopied] = React.useState<string | null>(null);

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopied(id);
    setTimeout(() => setCopied(null), 2000);
  };

  const PIPELINE_COMMANDS = [
    {
      id: 'all',
      title: '1. Complete End-to-End Pipeline',
      cmd: 'make all',
      desc: 'Runs data cleaning, reference attacker labeling, feature extraction, baselines, model training, evaluation, export, and paper table sync.',
    },
    {
      id: 'test',
      title: '2. Run Unit Test Suite (27 tests)',
      cmd: 'pytest tests/ -v && npm test',
      desc: 'Executes all Python unit tests (data, attackers, features, baselines, model, evaluate, api) and JS parity tests.',
    },
    {
      id: 'cli_score',
      title: '3. Command-Line Password Scoring',
      cmd: '/usr/local/bin/pwd score "Password123!" --details',
      desc: 'Invokes the installed console entry point to inspect 22 features and calculate guessability directly from bash.',
    },
    {
      id: 'api',
      title: '4. Zero-Logging FastAPI Service',
      cmd: 'uvicorn pwdstrength.api:app --host 0.0.0.0 --port 8000',
      desc: 'Starts privacy-preserving microservice with POST /score endpoint.',
    },
  ];

  return (
    <div className="space-y-6">
      {/* Overview Card */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 shadow-xl">
        <h2 className="text-xl font-bold text-white flex items-center gap-2 mb-2">
          <Terminal className="w-5 h-5 text-blue-400" />
          CLI Execution &amp; Reproducibility Guide
        </h2>
        <p className="text-xs text-slate-400 leading-relaxed">
          Every experiment in the research paper is fully scriptable with deterministic random seeds and isolated outputs.
        </p>
      </div>

      {/* Commands List */}
      <div className="grid grid-cols-1 gap-4">
        {PIPELINE_COMMANDS.map((item) => (
          <div key={item.id} className="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-white">{item.title}</span>
              <button
                type="button"
                onClick={() => copyToClipboard(item.cmd, item.id)}
                className="flex items-center gap-1 text-[11px] text-slate-400 hover:text-white px-2 py-1 rounded bg-slate-800/80 transition-colors"
              >
                {copied === item.id ? <CheckCircle className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                {copied === item.id ? 'Copied' : 'Copy'}
              </button>
            </div>
            <p className="text-xs text-slate-400">{item.desc}</p>
            <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800 font-mono text-xs text-blue-300 flex items-center justify-between">
              <code>{item.cmd}</code>
            </div>
          </div>
        ))}
      </div>

      {/* Research Data & Ethics Rules */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-2.5">
          <h3 className="text-sm font-bold text-white flex items-center gap-2">
            <Database className="w-4 h-4 text-indigo-400" />
            Corpus Setup for Researchers
          </h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Real evaluations in Section V require placing genuine password dumps (e.g. RockYou, MySpace) in{' '}
            <code className="text-slate-300">data/raw/</code>.
          </p>
          <div className="p-2.5 rounded-lg bg-amber-950/20 border border-amber-500/30 text-amber-300 text-xs flex items-start gap-2">
            <AlertCircle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
            <span>
              Synthetic generator (<code className="text-amber-200">scripts/make_synthetic.py</code>) is intended for smoke testing only. Never report synthetic data in published papers.
            </span>
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-2.5">
          <h3 className="text-sm font-bold text-white flex items-center gap-2">
            <Shield className="w-4 h-4 text-emerald-400" />
            Ethical Guardrails (Project Rule 3 &amp; 4)
          </h3>
          <ul className="text-xs text-slate-400 space-y-1.5 list-disc list-inside">
            <li>Never log passwords in the API, CLI, or UI.</li>
            <li>No network calls from the browser during scoring.</li>
            <li>Sampled passwords in Markov MC attacker are discarded immediately; never saved to disk.</li>
            <li>All usernames and emails are automatically stripped during cleaning.</li>
          </ul>
        </div>
      </div>
    </div>
  );
};
