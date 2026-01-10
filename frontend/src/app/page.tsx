'use client';

import { useState } from 'react';

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  // Remplacez cette URL par votre URL Render (ex: https://mon-api.onrender.com/predict)
  // Pour le test local, utilisez http://localhost:8000/predict
  const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/predict';

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setResult(null);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Erreur lors de l\'analyse');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError('Impossible de contacter le serveur. (Il peut être en cours de démarrage sur Render, réessayez dans 30s)');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-gray-900 text-white flex flex-col items-center justify-center p-4 font-sans">
      <div className="w-full max-w-md bg-gray-800 rounded-2xl shadow-2xl p-8 border border-gray-700">
        <h1 className="text-3xl font-extrabold text-center mb-2 text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-600">
          TYPE BEAT AI
        </h1>
        <p className="text-gray-400 text-center mb-8 text-sm">
          Découvrez quel artiste matcherait avec votre instru
        </p>

        {/* Upload Zone */}
        <div className="mb-6">
          <label 
            htmlFor="audio-upload" 
            className="flex flex-col items-center justify-center w-full h-32 border-2 border-gray-600 border-dashed rounded-lg cursor-pointer hover:bg-gray-700 transition-colors bg-gray-750"
          >
            <div className="flex flex-col items-center justify-center pt-5 pb-6">
              <svg className="w-8 h-8 mb-3 text-gray-400" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 20 16">
                <path stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 13h3a3 3 0 0 0 0-6h-.025A5.56 5.56 0 0 0 16 6.5 5.5 5.5 0 0 0 5.207 5.021C5.137 5.017 5.071 5 5 5a4 4 0 0 0 0 8h2.167M10 15V6m0 0L8 8m2-2 2 2"/>
              </svg>
              <p className="text-sm text-gray-400">
                {file ? <span className="text-green-400 font-semibold">{file.name}</span> : "Click to upload or drag and drop"}
              </p>
              <p className="text-xs text-gray-500">MP3, WAV (Max 10MB)</p>
            </div>
            <input id="audio-upload" type="file" className="hidden" accept=".mp3, .wav" onChange={handleFileChange} />
          </label>
        </div>

        {/* Action Button */}
        <button
          onClick={handleUpload}
          disabled={!file || loading}
          className={`w-full py-3 px-4 rounded-xl font-bold text-lg transition-all transform ${
            !file || loading 
              ? 'bg-gray-600 cursor-not-allowed opacity-50' 
              : 'bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 hover:scale-[1.02] shadow-lg'
          }`}
        >
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Analyse en cours...
            </span>
          ) : (
            'Analyser le Style'
          )}
        </button>

        {/* Results */}
        {error && (
            <div className="mt-4 p-3 bg-red-900/50 border border-red-700 rounded-lg text-red-200 text-sm text-center">
                {error}
            </div>
        )}

        {result && (
          <div className="mt-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <div className="bg-gray-700/50 rounded-xl p-4 border border-gray-600 text-center">
              <div className="text-sm text-gray-400 uppercase tracking-widest mb-1">Résultat Principal</div>
              <div className="text-4xl font-black text-white mb-2">{result.prediction}</div>
              <div className="inline-block px-3 py-1 bg-green-500/20 text-green-400 rounded-full text-sm font-bold">
                {(result.confidence * 100).toFixed(1)}% de correspondance
              </div>
            </div>

            <div className="mt-4">
              <h3 className="text-sm font-semibold text-gray-400 mb-2 ml-1">Autres possibilités :</h3>
              <div className="space-y-2">
                {result.details.slice(1, 4).map((item: any, idx: number) => (
                  <div key={idx} className="flex justify-between items-center bg-gray-800/80 p-3 rounded-lg border border-gray-700">
                    <span className="font-medium">{item.artist}</span>
                    <span className="text-gray-400 text-xs">{(item.score * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
