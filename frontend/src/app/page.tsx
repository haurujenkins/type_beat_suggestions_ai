'use client';

import { useState, useRef } from 'react';
import { 
  Upload, Loader2, Music, BarChart3, AlertCircle, 
  CheckCircle2, Zap, CloudLightning
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
  <nav className="fixed top-0 w-full z-50 border-b border-zinc-200 bg-white/80 backdrop-blur-sm">
    <div className="max-w-5xl mx-auto px-6 h-16 flex items-center justify-between">
      <div className="flex items-center gap-2">
        <span className="text-2xl font-black tracking-tighter text-zinc-900">
          TypeBeat<span className="text-red-500">Suggest</span>
        </span>
      </div>
      <div className="text-sm font-medium text-zinc-500">
        AI v2.0
      </div>
    </div>
  </nav>
);

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
      setErrorMessage("Format non supporté. Merci d'uploader un MP3 ou WAV.");
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
        let errorMsg = `Erreur Serveur (${response.status})`;
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
        prediction: topMatch.label || "Inconnu",
        confidence: toScore(topMatch.distance || 0),
        details: recommendations.map((rec: any) => ({
          artist: rec.label,
          score: toScore(rec.distance || 0),
          views: rec.views || 0,
          popularity: rec.popularity || 0
        }))
      };

      setResult(adaptedResult);
      setStatus('success');

    } catch (err: any) {
      console.error("Error:", err);
      setStatus('error');
      setErrorMessage(err.message || "Erreur de connexion. Veuillez réessayer.");
    }
  };

  const formatPercentage = (score: number) => Math.round(score * 100);

  // Seuil de popularité (ex: > 70/100) pour encadrer en vert
  const isPopular = (detail: PredictionDetail) => detail.popularity >= 65;

  return (
    <div className="min-h-screen bg-zinc-50 text-zinc-800 font-sans">
      <Navbar />

      <main className="pt-28 pb-20 px-4 md:px-0">
        <div className="max-w-2xl mx-auto space-y-10">
          
          {/* HEADER */}
          <div className="text-center space-y-2">
             <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-red-100 text-red-600 font-bold text-xs uppercase tracking-wider mb-2">
                <CloudLightning size={14} /> Power V2
             </div>
             <h1 className="text-4xl md:text-5xl font-black text-zinc-900 tracking-tight">
                Analyse ton Beat
             </h1>
             <p className="text-lg text-zinc-500 max-w-lg mx-auto">
                Upload ton fichier audio et découvre quel artiste matcherait le mieux avec ton instru.
             </p>
          </div>

          {/* UPLOAD BOX */}
          <div 
            className={`
                relative group cursor-pointer bg-white
                border-2 border-dashed rounded-xl p-10 text-center transition-all duration-200
                ${isDragOver 
                    ? 'border-red-500 bg-red-50' 
                    : 'border-zinc-300 hover:border-zinc-400 hover:bg-zinc-50/50'}
                ${status === 'uploading' || status === 'analyzing' ? 'opacity-90 pointer-events-none' : ''}
            `}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
             <input 
                type="file" 
                ref={fileInputRef} 
                className="hidden" 
                accept=".mp3,.wav" 
                onChange={handleFileSelect} 
             />
             
             <div className="flex flex-col items-center gap-4">
                <div className={`p-4 rounded-full bg-zinc-100 text-zinc-400 transition-colors group-hover:bg-white group-hover:text-red-500 group-hover:shadow-md`}>
                    {status === 'analyzing' || status === 'uploading' ? (
                        <Loader2 size={32} className="animate-spin text-red-500" />
                    ) : file ? (
                        <Music size={32} className="text-red-500" />
                    ) : (
                        <Upload size={32} />
                    )}
                </div>

                <div className="space-y-1">
                    {status === 'analyzing' ? (
                         <h3 className="text-lg font-bold text-zinc-800">Analyse spectrale en cours...</h3>
                    ) : file ? (
                         <h3 className="text-lg font-bold text-zinc-800">{file.name}</h3>
                    ) : (
                         <h3 className="text-lg font-bold text-zinc-800">Glisse ton fichier ici</h3>
                    )}
                    
                    {!status.includes('analyzing') && (
                        <p className="text-sm text-zinc-500">MP3 ou WAV (Max 20MB)</p>
                    )}
                </div>

                {file && status !== 'analyzing' && status !== 'success' && (
                    <button 
                        onClick={(e) => { e.stopPropagation(); handleSubmit(); }}
                        className="mt-4 px-8 py-3 bg-red-500 hover:bg-red-600 text-white font-bold rounded-lg shadow-lg shadow-red-500/30 active:scale-95 transition-all flex items-center gap-2"
                    >
                        <Zap size={18} fill="currentColor" /> Lancer l'analyse
                    </button>
                )}
             </div>
             
             {errorMessage && (
                <div className="absolute -bottom-14 left-0 right-0 p-3 mx-auto w-fit bg-red-50 text-red-500 text-sm font-medium rounded-lg border border-red-100 flex items-center gap-2">
                    <AlertCircle size={16} /> {errorMessage}
                </div>
             )}
          </div>

          {/* RESULTS SECTION */}
          {result && status === 'success' && (
            <div className="animate-fade-in-up space-y-8 pt-6">
                
                {/* 1. WINNER CARD (Streamlit Style) */}
                <div className="winner-container text-left relative overflow-hidden bg-white">
                    <div className="relative z-10">
                        <div className="winner-label flex items-center gap-2">
                             <CheckCircle2 size={16} /> TOP MATCH
                        </div>
                        <h2 className="winner-name text-zinc-900 tracking-tighter">
                            {result.prediction}
                        </h2>
                        <div className="winner-stat mt-2 flex items-center gap-2 text-zinc-500">
                             <span>Confiance IA :</span>
                             <span className="font-bold text-zinc-900">{formatPercentage(result.confidence)}%</span>
                        </div>
                    </div>
                </div>

                {/* 2. SUGGESTIONS GRID (8 items) */}
                <div>
                   <h3 className="text-zinc-400 font-bold uppercase text-xs tracking-wider mb-4 flex items-center gap-2 ml-1">
                      <BarChart3 size={14} />
                      8 Meilleures Suggestions
                   </h3>
                   
                   <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      {result.details.slice(0, 8).map((item, idx) => {
                          const popular = isPopular(item);
                          return (
                            <div 
                                key={idx} 
                                className={`
                                    relative flex items-center justify-between p-4 rounded-xl border transition-all duration-200 group
                                    ${popular 
                                        ? 'bg-green-50/50 border-green-500 shadow-sm' 
                                        : 'bg-white border-zinc-100 hover:border-red-200 hover:shadow-md'}
                                `}
                            >
                                <div className="flex items-center gap-4">
                                    <div className={`
                                        w-10 h-10 rounded-full flex items-center justify-center font-bold text-sm
                                        ${popular 
                                            ? 'bg-green-100 text-green-700' 
                                            : 'bg-zinc-100 text-zinc-500 group-hover:bg-red-50 group-hover:text-red-500 transition-colors'}
                                    `}>
                                        {idx + 1}
                                    </div>
                                    <div>
                                        <div className="font-bold text-zinc-900 text-lg leading-tight">
                                            {item.artist}
                                        </div>
                                        <div className="text-xs text-zinc-400 text-medium">
                                            {popular ? 'Artiste Tendance 🔥' : `${formatPercentage(item.score)}% Similarity`}
                                        </div>
                                    </div>
                                </div>

                                <div className={`font-mono font-bold ${popular ? 'text-green-600' : 'text-zinc-300 group-hover:text-red-500'}`}>
                                    {formatPercentage(item.score)}%
                                </div>
                            </div>
                          );
                      })}
                   </div>
                </div>
            </div>
          )}

        </div>
      </main>
    </div>
  );
}
