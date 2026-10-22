/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import {
  Shield,
  Layers,
  BarChart2,
  Terminal,
  FileSpreadsheet,
  Lock,
  ExternalLink,
  BookOpen,
  Cpu,
  Github,
} from 'lucide-react';
import { PasswordMeter } from './components/PasswordMeter';
import { ResearchDashboard } from './components/ResearchDashboard';
import { Visualizations } from './components/Visualizations';
import { CliPipelineGuide } from './components/CliPipelineGuide';

type ActiveView = 'meter' | 'paper' | 'charts' | 'cli';

export default function App() {
  const [currentView, setCurrentView] = useState<ActiveView>('meter');

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-blue-600 selection:text-white">
      {/* Top Navigation Bar */}
      <header className="sticky top-0 z-50 bg-slate-900/80 backdrop-blur-md border-b border-slate-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          {/* Logo & Paper Title */}
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-500 flex items-center justify-center shadow-lg shadow-blue-500/20 text-white font-bold">
              <Lock className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold text-base tracking-tight text-white">pwd-strength-ai</span>
                <span className="text-[10px] font-semibold uppercase px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20">
                  IEEE Research Prototype
                </span>
              </div>
              <p className="text-[11px] text-slate-400 truncate max-w-xs sm:max-w-md hidden sm:block">
                Hybrid Neural &amp; Pattern-Aware Password Guessability Model
              </p>
            </div>
          </div>

          {/* Navigation Controls */}
          <nav className="flex items-center gap-1.5 sm:gap-2">
            <button
              onClick={() => setCurrentView('meter')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                currentView === 'meter'
                  ? 'bg-blue-600 text-white shadow-md shadow-blue-500/20'
                  : 'text-slate-400 hover:text-white hover:bg-slate-800/60'
              }`}
            >
              <Lock className="w-3.5 h-3.5" />
              <span>Live Meter</span>
            </button>

            <button
              onClick={() => setCurrentView('paper')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                currentView === 'paper'
                  ? 'bg-blue-600 text-white shadow-md shadow-blue-500/20'
                  : 'text-slate-400 hover:text-white hover:bg-slate-800/60'
              }`}
            >
              <FileSpreadsheet className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Paper Tables</span>
              <span className="sm:hidden">Tables</span>
            </button>

            <button
              onClick={() => setCurrentView('charts')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                currentView === 'charts'
                  ? 'bg-blue-600 text-white shadow-md shadow-blue-500/20'
                  : 'text-slate-400 hover:text-white hover:bg-slate-800/60'
              }`}
            >
              <BarChart2 className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Figures</span>
              <span className="sm:hidden">Plots</span>
            </button>

            <button
              onClick={() => setCurrentView('cli')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                currentView === 'cli'
                  ? 'bg-blue-600 text-white shadow-md shadow-blue-500/20'
                  : 'text-slate-400 hover:text-white hover:bg-slate-800/60'
              }`}
            >
              <Terminal className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">CLI &amp; Pipeline</span>
              <span className="sm:hidden">CLI</span>
            </button>
          </nav>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {currentView === 'meter' && <PasswordMeter />}
        {currentView === 'paper' && <ResearchDashboard />}
        {currentView === 'charts' && <Visualizations />}
        {currentView === 'cli' && <CliPipelineGuide />}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950 py-6 text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <Shield className="w-4 h-4 text-emerald-400" />
            <span>Zero-Telemetry Policy: Passwords are evaluated strictly in browser client memory.</span>
          </div>
          <div className="flex items-center gap-4 text-slate-400">
            <span>Model: HybridStrengthNet (INT8)</span>
            <span>&bull;</span>
            <span>Ref: Dell&apos;Amico &amp; Filippone (CCS &apos;15)</span>
            <span>&bull;</span>
            <span>IEEE Manuscript Companion</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
