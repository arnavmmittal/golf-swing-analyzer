"use client";

import { useRef, useState } from "react";
import clsx from "clsx";

export type Handedness = "auto" | "right" | "left";
export type View = "dtl" | "face-on";

type Props = {
  onUpload: (file: File, handedness: Handedness, view: View) => void;
  uploading: boolean;
};

export function UploadCard({ onUpload, uploading }: Props) {
  const [handedness, setHandedness] = useState<Handedness>("auto");
  const [view, setView] = useState<View>("dtl");
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function handleFile(file: File) {
    if (!file.type.startsWith("video/") && !/\.(mp4|mov|m4v|avi)$/i.test(file.name)) {
      setError("Please upload a video file (mp4, mov, m4v, avi).");
      return;
    }
    if (file.size > 200 * 1024 * 1024) {
      setError("File too large (200MB max).");
      return;
    }
    setError(null);
    onUpload(file, handedness, view);
  }

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        const file = e.dataTransfer.files?.[0];
        if (file) handleFile(file);
      }}
      className={clsx(
        "rounded-2xl border-2 border-dashed bg-white/70 backdrop-blur p-12 text-center transition shadow-sm",
        dragOver
          ? "border-fairway-600 bg-fairway-50"
          : "border-fairway-500/40 hover:border-fairway-600/70",
      )}
    >
      <div className="mx-auto max-w-md">
        <div className="text-5xl font-display text-fairway-700 mb-3">↑</div>
        <h3 className="font-display text-2xl text-fairway-900 mb-2">
          Drop your swing video
        </h3>
        <p className="text-sm text-fairway-700/80 mb-6">
          Side-on or down-the-line works best. Up to 60 seconds, max 200MB. We
          measure pose with MediaPipe and grade against tour benchmarks.
        </p>
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          disabled={uploading}
          className={clsx(
            "px-6 py-3 rounded-full font-medium transition",
            uploading
              ? "bg-fairway-500/40 text-white cursor-wait"
              : "bg-fairway-700 hover:bg-fairway-600 text-white",
          )}
        >
          {uploading ? "Analyzing…" : "Choose video"}
        </button>
        <input
          ref={inputRef}
          type="file"
          accept="video/*,.mp4,.mov,.m4v,.avi"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) handleFile(file);
          }}
        />
        <div className="mt-6 space-y-3">
          <div className="flex items-center justify-center gap-2 text-sm">
            <span className="text-fairway-700/70 mr-1">I am</span>
            {(["auto", "right", "left"] as const).map((opt) => (
              <button
                key={opt}
                type="button"
                onClick={() => setHandedness(opt)}
                className={clsx(
                  "px-3 py-1 rounded-full border transition",
                  handedness === opt
                    ? "bg-fairway-700 text-white border-fairway-700"
                    : "bg-white border-fairway-500/30 text-fairway-700 hover:border-fairway-500/60",
                )}
              >
                {opt === "auto" ? "auto-detect" : opt === "right" ? "right-handed" : "left-handed"}
              </button>
            ))}
          </div>
          <div className="flex items-center justify-center gap-2 text-sm">
            <span className="text-fairway-700/70 mr-1">camera</span>
            {(["dtl", "face-on"] as const).map((opt) => (
              <button
                key={opt}
                type="button"
                onClick={() => setView(opt)}
                className={clsx(
                  "px-3 py-1 rounded-full border transition",
                  view === opt
                    ? "bg-fairway-700 text-white border-fairway-700"
                    : "bg-white border-fairway-500/30 text-fairway-700 hover:border-fairway-500/60",
                )}
              >
                {opt === "dtl" ? "down the line" : "face on"}
              </button>
            ))}
          </div>
          <p className="text-xs text-fairway-700/60 max-w-md mx-auto">
            {view === "dtl"
              ? "From down-the-line we measure tempo, X-Factor, hip rotation, spine stability, and lead arm. Head sway, weight transfer, and shaft lean need a face-on view."
              : "From face-on we measure tempo, head sway, weight transfer, shaft lean, and spine stability. X-Factor, hip rotation, and lead arm need down-the-line."}
          </p>
        </div>
        {error && (
          <p className="mt-4 text-sm text-red-600">{error}</p>
        )}
      </div>
    </div>
  );
}
