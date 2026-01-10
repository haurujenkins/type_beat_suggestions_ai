'use client';

import { useState, useRef } from 'react';
import { Upload, Music, Play, BarChart3, Loader2, Search, Zap, AlertCircle } from 'lucide-react';

// Types pour la réponse API
interface PredictionDetail {
  artist: string;
  score: number;
  views: number;
}

interface ApiResponse {
  prediction: string;
  confidence: number;
  details: PredictionDetail[];
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [status, setStatus] = useState<'idle' | 'uploading' | 'analyzing' | 'success' | 'error'>('idle');
  const [result, setResult] = useState<ApiResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // URL API depuis variable d'environnement (avec fallback localhost pour le dev)
  const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/predict';

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
    if (selectedFile.type === 'audio/mpeg' || selectedFile.type === 'audio/wav' || selectedFile.name.endsWith('.mp3') || selectedFile.name.endsWith('.wav')) {
      setFile(selectedFile);
      setErrorMessage(null);
      setStatus('idle');
      setResult(null);
    } else {
      setErrorMessage("Format non supporté. Merci d'uploader un fichier MP3 ou WAV.");
    }
  };

  const handleSubmit = async () => {
    if (!file) return;

    setStatus('uploading');
    setErrorMessage(null);

    const formData = new FormData();
    formData.append('file', file);
    
    // URL cible (variable d'env en PROD, hardcodée ici par sécurité suite à la demande)
    const targetUrl = 'https://type-beat-suggestions-ai.onrender.com/predict';

    try {
      console.log("🚀 Envoi vers l'URL :", targetUrl);

      // Simulation d'UX pour voir le loader (optionnel)
      // await new Promise(r => setTimeout(r, 500)); 
      
      setStatus('analyzing');

      const response = await fetch(targetUrl, {
        method: 'POST',
        body: formData,
        // PAS DE CONTENT-TYPE MANUEL AVEC FORMDATA !
      });

      if (!response.ok) {
        // Tentative de lire le message d'erreur JSON du backend
        let errorMsg = `Erreur serveur (${response.status})`;
        try {
          const errorData = await response.json();
          if (errorData.detail) errorMsg = errorData.detail;
        } catch (e) {
             // Si le body n'est pas json (ex: erreur système Render)
        }
        throw new Error(errorMsg);
      }

      // 1. Récupération des données brutes
      const rawData = await response.json();
      console.log("📦 Réponse Backend reçue :", rawData);

      // 2. Mapping de la structure Backend (backend/main.py) vers l'interface Frontend (ApiResponse)
      // Backend: { input_filename, recommendations: [{ rank, filename, label, distance, preview_path }] }
      // Frontend: { prediction, confidence, details: [{ artist, score, views }] }
      
      const recommendations = rawData.recommendations || [];
      const topMatch = recommendations[0] || {};
      
      // Conversion Distance -> Score de confiance (Distance cosine : 0 = identique, 1 = opposé)
      // Une distance de 0.1 est très proche. On inverse pour afficher un score.
      const toScore = (dist: number) => Math.max(0, 1 - dist); 

      const adaptedResult: ApiResponse = {
        prediction: topMatch.label || "Inconnu",
        confidence: toScore(topMatch.distance || 0),
        details: recommendations.map((rec: any) => ({
          artist: rec.label,
          score: toScore(rec.distance || 0),
          views: 0 // Donnée pas encore dispo dans le backend V2 light
        }))
      };

      setResult(adaptedResult);
      setStatus('success');

    } catch (err: any) {
      console.error("❌ Erreur catchée :", err);
      setStatus('error');
      // Gestion des erreurs spécifiques (timeout, network, etc)
      if (err.message.includes('422')) {
        setErrorMessage("Le fichier envoyé n'est pas valide (422).");
      } else if (err.message.includes('503')) {
        setErrorMessage("Le modèle est encore en cours chargement. Réessayez dans 10 secondes.");
      } else {
        setErrorMessage(err.message || "Erreur de connexion au serveur.");
      }
    }
  };

  const formatPercentage = (score: number) => Math.round(score * 100);

  return (
    <main className="min-h-screen bg-slate-950 text-slate-200 font-sans selection:bg-indigo-500/30">
      
      {/* BACKGROUND GRADIENTS */}
      <div className="fixed inset-0 z-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[-10%] right-[-5%] w-[500px] h-[500px] bg-indigo-900/20 rounded-full blur-[120px]" />
        <div className="absolute bottom-[-10%] left-[-10%] w-[600px] h-[600px] bg-purple-900/10 rounded-full blur-[100px]" />
      </div>

      <div className="relative z-10 max-w-5xl mx-auto px-6 py-12 md:py-20">
        
        {/* HERO SECTION */}
        <div className="text-center mb-16 space-y-6">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-slate-900/50 border border-slate-700/50 text-indigo-400 text-sm font-medium backdrop-blur-sm animate-fade-in">
            <Zap size={14} fill="currentColor" />
            <span>AI Powered Audio Analysis v2.0</span>
          </div>
          
          <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight text-white mb-4">
            Type Beat <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-purple-500">Finder</span>
          </h1>
          
          <p className="text-lg md:text-xl text-slate-400 max-w-2xl mx-auto leading-relaxed">
            Upload ton mp3, l'IA trouve les sons similaires parmi +50 artistes pros.
          </p>
        </div>

        {/* UPLOAD SECTION */}
        <div className="max-w-xl mx-auto mb-20">
          <div 
            className={`
              relative group cursor-pointer
              border-2 border-dashed rounded-3xl p-10 text-center transition-all duration-300 ease-out
              ${isDragOver 
                ? 'border-indigo-500 bg-indigo-500/10 scale-[1.02]' 
                : 'border-slate-700 bg-slate-900/40 hover:border-slate-500 hover:bg-slate-800/40'}
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
              <div className={`p-4 rounded-full bg-slate-800 transition-colors ${isDragOver ? 'text-indigo-400' : 'text-slate-400 group-hover:text-indigo-400'}`}>
                {status === 'analyzing' || status === 'uploading' ? (
                   <Loader2 size={40} className="animate-spin" />
                ) : (
                   <Upload size={40} />
                )}
              </div>
              
              <div className="space-y-2">
                <h3 className="text-xl font-semibold text-white">
                  {file ? file.name : "Glisse ton fichier ici"}
                </h3>
                <p className="text-sm text-slate-500">
                  {file ? (
                    <span className="text-indigo-400 font-medium">Prêt pour l'analyse</span>
                  ) : (
                    "MP3 ou WAV (Max 10Mo)"
                  )}
                </p>
              </div>
            </div>
            
            {/* PROGRESS BAR SIMULATION */}
            {status === 'analyzing' && (
              <div className="absolute bottom-0 left-0 h-1 bg-indigo-500/20 w-full rounded-b-3xl overflow-hidden">
                <div className="h-full bg-indigo-500 animate-progress-indeterminate"></div>
              </div>
            )}
          </div>

          <button
            onClick={handleSubmit}
            disabled={!file || status === 'uploading' || status === 'analyzing'}
            className={`
              w-full mt-6 py-4 px-6 rounded-xl font-bold text-lg shadow-lg flex items-center justify-center gap-3 transition-all
              ${!file 
                ? 'bg-slate-800 text-slate-500 cursor-not-allowed' 
                : 'bg-indigo-600 hover:bg-indigo-500 text-white hover:scale-[1.02] active:scale-[0.98] shadow-indigo-500/25'}
            `}
          >
            {status === 'analyzing' ? (
              <>
                <Loader2 className="animate-spin" /> Analyse spectrale en cours...
              </>
            ) : (
               <>
                <Search size={20} /> Lancer la recherche
               </>
            )}
          </button>

          {errorMessage && (
            <div className="mt-4 p-4 bg-red-500/10 border border-red-500/20 rounded-xl flex items-center gap-3 text-red-400 text-sm animate-in fade-in slide-in-from-top-2">
              <AlertCircle size={18} />
              {errorMessage}
            </div>
          )}
        </div>

        {/* RESULTS SECTION */}
        {result && status === 'success' && (
          <div className="animate-in fade-in slide-in-from-bottom-8 duration-700">
            <h2 className="text-2xl font-bold text-white mb-8 flex items-center gap-2">
              <BarChart3 className="text-indigo-400" /> Résultats de l'analyse
            </h2>

            {/* TOP MATCH */}
            <div className="mb-12 cursor-pointer group relative overflow-hidden rounded-3xl bg-gradient-to-br from-indigo-900/50 to-slate-900 border border-indigo-500/30 p-8 md:p-10 transition-all hover:border-indigo-500/50">
                <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                    <Music size={200} />
                </div>
                
                <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-6">
                    <div>
                        <div className="text-indigo-300 font-medium tracking-wider text-sm mb-2 uppercase">Match Principal</div>
                        <h3 className="text-4xl md:text-6xl font-black text-white tracking-tight mb-2">
                            {result.prediction}
                        </h3>
                        <p className="text-lg text-slate-400 flex items-center gap-2">
                            Type Beat • {formatPercentage(result.confidence)}% de similarité
                        </p>
                    </div>
                    
                    <div className="flex items-center gap-4">
                         <div className="text-right hidden md:block">
                            <div className="text-3xl font-bold text-white">{formatPercentage(result.confidence)}%</div>
                            <div className="text-xs text-slate-500">Confidence Score</div>
                         </div>
                         <div className="h-16 w-16 md:h-20 md:w-20 rounded-full bg-indigo-500 flex items-center justify-center text-white shadow-lg shadow-indigo-500/30 group-hover:scale-110 transition-transform">
                             <Play fill="currentColor" className="ml-1" />
                         </div>
                    </div>
                </div>
            </div>

            {/* GRID OF SUGGESTIONS */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {result.details.slice(1, 5).map((detail, idx) => (
                <div 
                    key={idx}
                    className="group bg-slate-900/40 hover:bg-slate-800/60 border border-slate-800 hover:border-slate-600 p-5 rounded-2xl transition-all hover:-translate-y-1"
                >
                    <div className="flex justify-between items-start mb-4">
                        <div className="w-10 h-10 rounded-full bg-slate-800 flex items-center justify-center text-slate-400 group-hover:text-white group-hover:bg-indigo-600 transition-colors">
                            <Music size={18} />
                        </div>
                        <span className="text-xs font-mono text-slate-500 bg-slate-900 px-2 py-1 rounded">
                            {formatPercentage(detail.score)}% MATCH
                        </span>
                    </div>
                    
                    <h4 className="text-xl font-bold text-slate-100 mb-1 truncate">{detail.artist}</h4>
                    <p className="text-sm text-slate-500 mb-4">Type Beat Style</p>
                    
                    <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                        <div 
                            className="bg-indigo-500 h-full rounded-full" 
                            style={{ width: `${formatPercentage(detail.score)}%` }}
                        />
                    </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
