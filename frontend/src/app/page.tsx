'use client';

import { useState, useRef } from 'react';
import { 
  Upload, Music, Play, BarChart3, Loader2, Search, Zap, 
  AlertCircle, CheckCircle2, Disc, Waveform, 
  Github, Twitter, Mail, TrendingUp
} from 'lucide-react';

/* --- TYPES --- */
interface PredictionDetail {
  artist: string;
  score: number;
  views: number;
  popularity: number;
}

interface ApiResponse {
  prediction: string;
  confidence: number;
  details: PredictionDetail[];
}

/* --- COMPONENTS --- */

const Navbar = () => (
  <nav className="fixed top-0 w-full z-50 border-b border-white/5 bg-zinc-950/80 backdrop-blur-md">
    <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
      <div className="flex items-center gap-2">
        <div className="w-8 h-8 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-lg flex items-center justify-center">
          <Waveform className="text-white w-5 h-5" />
        </div>
        <span className="font-bold text-lg tracking-tight text-white">TypeBeat<span className="text-indigo-400">Finder</span></span>
      </div>
      <div className="flex items-center gap-6 text-sm font-medium text-zinc-400">
        <div className="hidden md:flex gap-6">
            <a href="#" className="hover:text-white transition-colors">Pricing</a>
            <a href="#" className="hover:text-white transition-colors">API</a>
        </div>
        <a href="#" className="px-4 py-2 bg-white/5 hover:bg-white/10 text-white rounded-full border border-white/5 transition-all text-sm">
          Sign In
        </a>
      </div>
    </div>
  </nav>
);

const Footer = () => (
    <footer className="border-t border-white/5 mt-20 bg-zinc-950">
        <div className="max-w-7xl mx-auto px-6 py-12 flex flex-col md:flex-row justify-between items-center gap-6">
            <div className="text-zinc-500 text-sm">
                © 2026 TypeBeatFinder AI. All rights reserved.
            </div>
            <div className="flex gap-6">
                <Github className="w-5 h-5 text-zinc-600 hover:text-white cursor-pointer transition-colors" />
                <Twitter className="w-5 h-5 text-zinc-600 hover:text-white cursor-pointer transition-colors" />
                <Mail className="w-5 h-5 text-zinc-600 hover:text-white cursor-pointer transition-colors" />
            </div>
        </div>
    </footer>
);

/* --- MAIN PAGE --- */

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [status, setStatus] = useState<'idle' | 'uploading' | 'analyzing' | 'success' | 'error'>('idle');
  const [result, setResult] = useState<ApiResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // URL API Handling
  const envUrl = process.env.NEXT_PUBLIC_API_URL || 'https://type-beat-suggestions-ai.onrender.com';
  const API_URL = envUrl.endsWith('/predict') ? envUrl : `${envUrl}/predict`;

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndSetFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      validateAndSetFile(e.target.files[0]);
    }
  };

  const validateAndSetFile = (selectedFile: File) => {
    const validTypes = ['audio/mpeg', 'audio/wav', 'audio/mp3', 'audio/x-wav'];
    const validExtensions = ['.mp3', '.wav'];
    
    // Check type or extension
    const isValidType = validTypes.includes(selectedFile.type);
    const isValidExt = validExtensions.some(ext => selectedFile.name.toLowerCase().endsWith(ext));

    if (isValidType || isValidExt) {
      setFile(selectedFile);
      setErrorMessage(null);
      setStatus('idle');
      setResult(null);
    } else {
      setErrorMessage("Unsupported format. Please upload MP3 or WAV.");
    }
  };

  const handleSubmit = async () => {
    if (!file) return;

    setStatus('uploading');
    setErrorMessage(null);

    const formData = new FormData();
    formData.append('file', file);
    
    try {
      setStatus('analyzing');

      const response = await fetch(API_URL, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        let errorMsg = `Server Error (${response.status})`;
        try {
          const errorData = await response.json();
          if (errorData.detail) errorMsg = errorData.detail;
        } catch (e) {}
        throw new Error(errorMsg);
      }

      const rawData = await response.json();
      
      const recommendations = rawData.recommendations || [];
      const topMatch = recommendations[0] || {};
      const toScore = (dist: number) => Math.max(0, 1 - dist); 

      const adaptedResult: ApiResponse = {
        prediction: topMatch.label || "Unknown",
        confidence: toScore(topMatch.distance || 0),
        details: recommendations.map((rec: any) => ({
          artist: rec.label,
          score: toScore(rec.distance || 0),
          views: rec.views || 0,
          popularity: rec.popularity || 0
        }))
      };

      try {
         console.log("Prediction Details:", adaptedResult.details);
      } catch (e) {}

      setResult(adaptedResult);
      setStatus('success');

    } catch (err: any) {
      console.error("Error:", err);
      setStatus('error');
      setErrorMessage(err.message || "Connection failed. Please try again.");
    }
  };

  const formatProbability = (score: number) => Math.round(score * 100);
  
  // Fonction pour déterminer si un artiste est "Populaire" (Green Border)
  const isPopular = (detail: PredictionDetail) => {
      // Seuil arbitraire basé sur le CSV (ex: popularity > 70 ou views > 2000)
      return detail.popularity >= 70;
  };

  return (
    <div className="min-h-screen bg-zinc-950 flex flex-col font-sans selection:bg-indigo-500/30">
      <Navbar />

      {/* BACKGROUND FX */}
      <div className="fixed inset-0 z-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[-10%] right-[-5%] w-[800px] h-[800px] bg-indigo-600/10 rounded-full blur-[120px]" />
        <div className="absolute bottom-[-10%] left-[-10%] w-[600px] h-[600px] bg-purple-600/10 rounded-full blur-[100px]" />
        <div className="absolute top-[20%] left-[20%] w-[400px] h-[400px] bg-blue-600/5 rounded-full blur-[80px]" />
      </div>

      <main className="relative z-10 flex-grow pt-32 px-6">
        <div className="max-w-6xl mx-auto w-full">
            
            {/* HERO SECTION */}
            <div className="text-center mb-16 animate-fade-in-up">
                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/5 border border-white/10 text-indigo-300 text-xs font-medium backdrop-blur-sm mb-6 hover:bg-white/10 transition-colors cursor-default">
                    <span className="flex h-2 w-2 rounded-full bg-indigo-500 animate-pulse"></span>
                    V2.0 Engine Live
                </div>
                
                <h1 className="text-5xl md:text-7xl font-bold tracking-tight text-white mb-6 text-glow">
                    Identify any <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 via-purple-400 to-indigo-400">Type Beat</span> style.
                </h1>
                
                <p className="text-lg md:text-xl text-zinc-400 max-w-2xl mx-auto leading-relaxed">
                    Upload your raw audio. Our AI analyzes the spectral fingerprint to match it against 50+ professional artist styles instantly.
                </p>
            </div>

            {/* MAIN INTERFACE GRID */}
            <div className="grid grid-cols-1 gap-12 max-w-4xl mx-auto">
                
                {/* UPLOAD CARD */}
                <div className={`
                    relative group animate-fade-in-up delay-100
                    glass-panel rounded-3xl p-1 transition-all duration-300
                    ${isDragOver ? 'ring-2 ring-indigo-500 scale-[1.01]' : 'hover:border-white/20'}
                `}>
                    <div 
                        className="bg-zinc-900/80 rounded-[22px] p-10 text-center border border-white/5 h-full flex flex-col items-center justify-center cursor-pointer relative overflow-hidden"
                        onDragOver={handleDragOver}
                        onDragLeave={handleDragLeave}
                        onDrop={handleDrop}
                        onClick={() => fileInputRef.current?.click()}
                    >
                         {/* Grid Pattern Background */}
                         <div className="absolute inset-0 opacity-[0.03]" 
                              style={{ backgroundImage: 'radial-gradient(#fff 1px, transparent 1px)', backgroundSize: '30px 30px' }}>
                         </div>

                        <input 
                            type="file" 
                            ref={fileInputRef} 
                            className="hidden" 
                            accept=".mp3,.wav" 
                            onChange={handleFileSelect} 
                        />

                        {/* Icon State */}
                        <div className={`
                            w-20 h-20 rounded-2xl flex items-center justify-center mb-6 transition-all duration-300
                            ${status === 'analyzing' || status === 'uploading' 
                                ? 'bg-indigo-500/20 text-indigo-400' 
                                : isDragOver || file 
                                    ? 'bg-indigo-500 text-white shadow-lg shadow-indigo-500/50 scale-110' 
                                    : 'bg-zinc-800 text-zinc-500 group-hover:bg-zinc-700 group-hover:text-zinc-300'}
                        `}>
                            {status === 'analyzing' || status === 'uploading' ? (
                                <Loader2 className="w-8 h-8 animate-spin" />
                            ) : file ? (
                                <Music className="w-8 h-8" />
                            ) : (
                                <Upload className="w-8 h-8" />
                            )}
                        </div>

                        {/* Text Content */}
                        <div className="space-y-2 relative z-10 w-full flex flex-col items-center">
                            {status === 'analyzing' ? (
                                <div className="space-y-4 w-full max-w-xs mx-auto">
                                    <h3 className="text-xl font-semibold text-white">Analyzing Audio Spectrum...</h3>
                                    <div className="h-1.5 w-full bg-zinc-800 rounded-full overflow-hidden">
                                        <div className="h-full bg-indigo-500 animate-progress"></div>
                                    </div>
                                    <p className="text-sm text-zinc-500 font-mono">Extracting features...</p>
                                </div>
                            ) : file ? (
                                <>
                                    <h3 className="text-xl font-semibold text-white break-all line-clamp-1 px-4">{file.name}</h3>
                                    <p className="text-indigo-400 font-medium flex items-center justify-center gap-2">
                                        <CheckCircle2 size={16} /> Ready to analyze
                                    </p>
                                    <button 
                                        onClick={(e) => { e.stopPropagation(); handleSubmit(); }}
                                        className="mt-6 w-full max-w-xs py-4 bg-white text-black font-bold rounded-xl hover:bg-zinc-200 transition-colors shadow-lg active:scale-95 flex items-center justify-center gap-2"
                                    >
                                        <Zap size={18} className="fill-current" />
                                        Launch Analysis
                                    </button>
                                </>
                            ) : (
                                <>
                                    <h3 className="text-xl font-semibold text-white group-hover:text-indigo-300 transition-colors">Drag & drop your beat</h3>
                                    <p className="text-sm text-zinc-500">Supports MP3, WAV (Max 20MB)</p>
                                </>
                            )}
                        </div>

                        {errorMessage && (
                            <div className="absolute bottom-4 left-0 right-0 mx-auto px-4 w-fit">
                                <div className="px-4 py-2 bg-red-500/10 border border-red-500/20 text-red-400 text-xs rounded-full flex items-center gap-2 animate-in fade-in slide-in-from-bottom-2">
                                    <AlertCircle size={12} /> {errorMessage}
                                </div>
                            </div>
                        )}
                    </div>
                </div>

                {/* RESULTS AREA */}
                {result && status === 'success' && (
                    <div className="space-y-12 animate-fade-in-up">
                        
                        {/* FEATURE MATCH */}
                        <div className="relative overflow-hidden rounded-3xl bg-indigo-600 p-8 md:p-12 text-white shadow-2xl shadow-indigo-900/50">
                             <div className="absolute top-0 right-0 -mt-10 -mr-10 opacity-20">
                                <Disc size={300} className="animate-spin-slow" />
                             </div>
                             
                             <div className="relative z-10 grid md:grid-cols-2 gap-8 items-center">
                                <div>
                                    <div className="inline-block px-3 py-1 bg-white/20 backdrop-blur-md rounded-full text-xs font-bold uppercase tracking-wider mb-4 border border-white/20">
                                        Top Prediction
                                    </div>
                                    <h2 className="text-5xl md:text-7xl font-black mb-2 tracking-tighter">
                                        {result.prediction}
                                    </h2>
                                    <div className="flex items-center gap-3 opacity-90">
                                        <span className="text-2xl font-light">Type Beat</span>
                                        <span className="h-1 w-1 bg-white rounded-full"></span>
                                        <span className="text-lg font-medium">Confidence: {formatProbability(result.confidence)}%</span>
                                    </div>
                                </div>

                                <div className="space-y-4 bg-white/10 backdrop-blur-md p-6 rounded-2xl border border-white/10">
                                    <div className="flex justify-between items-center text-sm font-medium opacity-80 mb-2">
                                        <span>Match Confidence</span>
                                        <span>{formatProbability(result.confidence)}%</span>
                                    </div>
                                    <div className="h-4 bg-black/20 rounded-full overflow-hidden backdrop-blur-sm">
                                        <div 
                                            className="h-full bg-white rounded-full transition-all duration-1000 ease-out"
                                            style={{ width: `${formatProbability(result.confidence)}%` }}
                                        />
                                    </div>
                                    <p className="text-xs opacity-70 mt-2 leading-relaxed">
                                        Our model identified strong spectral similarities with {result.prediction}'s production style.
                                    </p>
                                </div>
                             </div>
                        </div>

                        {/* OTHER SUGGESTIONS - GRID 8 */}
                        <div>
                            <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
                                <BarChart3 className="text-indigo-500" />
                                Top 8 Alternatives
                            </h3>
                            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
                                {result.details.slice(1, 9).map((item, idx) => {
                                    const popular = isPopular(item);
                                    return (
                                        <div 
                                            key={idx} 
                                            className={`
                                                relative glass-card p-5 rounded-xl border flex flex-col justify-between group
                                                ${popular 
                                                    ? 'border-green-500/30 bg-green-500/5 hover:bg-green-500/10' 
                                                    : 'border-white/5 hover:border-white/10'}
                                            `}
                                        >
                                            <div className="flex justify-between items-start mb-4">
                                                <div className="flex items-center gap-3">
                                                    <div className="w-8 h-8 rounded-full bg-zinc-800 flex items-center justify-center text-zinc-400 font-bold text-xs group-hover:bg-indigo-500 group-hover:text-white transition-colors">
                                                        {idx + 2}
                                                    </div>
                                                </div>
                                                {popular && (
                                                    <div className="text-[10px] font-bold uppercase tracking-wider text-green-400 bg-green-500/10 px-2 py-1 rounded-full border border-green-500/20 flex items-center gap-1">
                                                        <TrendingUp size={10} /> Popular
                                                    </div>
                                                )}
                                            </div>
                                            
                                            <div>
                                                <div className="font-bold text-white text-lg mb-1 truncate" title={item.artist}>
                                                    {item.artist}
                                                </div>
                                                <div className="flex items-center justify-between">
                                                    <div className="text-xs text-zinc-500">Match score</div>
                                                    <div className={`font-mono font-bold ${popular ? 'text-green-400' : 'text-indigo-400'}`}>
                                                        {formatProbability(item.score)}%
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
      </main>

      <Footer />
    </div>
  );
}
