"use client";

import { useState } from "react";
import { UploadCard, type Handedness } from "@/components/UploadCard";
import { Results } from "@/components/Results";
import type { AnalysisResult } from "@/lib/types";

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export default function Home() {
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [stage, setStage] = useState<string>("");

  async function handleUpload(file: File, handedness: Handedness) {
    setUploading(true);
    setError(null);
    setResult(null);
    setStage("Uploading video…");

    const form = new FormData();
    form.append("video", file);
    form.append("handedness", handedness);

    try {
      // Stage timer — switches the visible label as time passes.
      const stages = [
        [0, "Uploading video…"],
        [3000, "Detecting body landmarks (MediaPipe)…"],
        [12000, "Segmenting swing phases…"],
        [16000, "Computing biomechanics…"],
        [22000, "Generating coaching feedback…"],
        [40000, "Rendering annotated video…"],
      ] as const;
      const timers = stages.map(([ms, label]) =>
        setTimeout(() => setStage(label), ms),
      );

      const res = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        body: form,
      });

      timers.forEach(clearTimeout);

      if (!res.ok) {
        const detail = await res.text();
        throw new Error(`Analysis failed: ${res.status} ${detail.slice(0, 200)}`);
      }
      const data: AnalysisResult = await res.json();
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setUploading(false);
      setStage("");
    }
  }

  return (
    <main className="min-h-screen px-4 py-10 sm:py-16">
      <div className="max-w-6xl mx-auto">
        <header className="text-center mb-10 sm:mb-14">
          <div className="inline-block text-[10px] font-semibold tracking-[0.2em] uppercase text-sand-500 mb-3">
            Tour-level biomechanics
          </div>
          <h1 className="font-display text-5xl sm:text-6xl text-fairway-900 mb-3 leading-tight">
            Swing like a +5.
          </h1>
          <p className="max-w-xl mx-auto text-fairway-900/70">
            Upload a video of your swing. We measure the eight metrics that
            define a tour-level swing, then tell you exactly what to work on.
          </p>
        </header>

        {!result && (
          <UploadCard onUpload={handleUpload} uploading={uploading} />
        )}

        {uploading && (
          <div className="mt-8 text-center text-fairway-700">
            <div className="inline-flex items-center gap-3">
              <span className="inline-block w-2 h-2 rounded-full bg-fairway-600 animate-pulse" />
              <span className="text-sm">{stage}</span>
            </div>
            <p className="mt-2 text-xs text-fairway-700/60">
              First analysis takes 30–60s. The MediaPipe model is heavy.
            </p>
          </div>
        )}

        {error && (
          <div className="mt-8 max-w-xl mx-auto rounded-xl bg-red-50 border border-red-200 p-4 text-sm text-red-800">
            {error}
          </div>
        )}

        {result && (
          <div className="mt-2">
            <div className="flex justify-end mb-4">
              <button
                onClick={() => {
                  setResult(null);
                  setError(null);
                }}
                className="text-sm text-fairway-700 hover:text-fairway-600 underline"
              >
                Analyze another swing
              </button>
            </div>
            <Results result={result} apiBase={API_BASE} />
          </div>
        )}

        <footer className="mt-16 text-center text-xs text-fairway-700/50">
          MediaPipe BlazePose · Claude Opus 4.7 · benchmarks from McLean, TPI,
          Cheetham et al.
        </footer>
      </div>
    </main>
  );
}
