import type { AnalysisResult } from "@/lib/types";
import { MetricCard } from "./MetricCard";
import { FaultCard } from "./FaultCard";

export function Results({
  result,
  apiBase,
}: {
  result: AnalysisResult;
  apiBase: string;
}) {
  const videoSrc = `${apiBase}${result.annotated_video_url}`;
  return (
    <div className="space-y-8">
      <Hero result={result} videoSrc={videoSrc} />
      <section>
        <h2 className="font-display text-2xl text-fairway-900 mb-4">
          The eight tour benchmarks
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {result.scores.map((s) => (
            <MetricCard key={s.name} metric={s} />
          ))}
        </div>
      </section>
      <section>
        <h2 className="font-display text-2xl text-fairway-900 mb-4">
          Your three highest-leverage fixes
        </h2>
        <div className="space-y-4">
          {result.feedback.faults.map((f, i) => (
            <FaultCard
              key={f.metric}
              fault={f}
              headline={f.metric === result.feedback.headline_fault || i === 0}
            />
          ))}
        </div>
      </section>
    </div>
  );
}

function Hero({
  result,
  videoSrc,
}: {
  result: AnalysisResult;
  videoSrc: string;
}) {
  const score = result.overall_score;
  const grade =
    score >= 90 ? "Tour level" :
    score >= 75 ? "Single-digit" :
    score >= 60 ? "Mid-handicap" :
    "Building blocks";

  return (
    <div className="grid grid-cols-1 lg:grid-cols-5 gap-6 items-stretch">
      <div className="lg:col-span-3 rounded-2xl overflow-hidden bg-fairway-900 shadow-md">
        <video
          src={videoSrc}
          controls
          autoPlay
          loop
          muted
          className="w-full h-full object-contain bg-black"
        />
      </div>
      <div className="lg:col-span-2 rounded-2xl bg-white/90 backdrop-blur p-7 shadow-sm border border-fairway-500/10 flex flex-col">
        <div className="text-[10px] font-semibold tracking-wider uppercase text-fairway-700/60">
          Overall swing score
        </div>
        <div className="flex items-baseline gap-3 mt-1">
          <span className="font-display text-7xl text-fairway-900 leading-none">
            {score}
          </span>
          <span className="text-sand-500 font-semibold">{grade}</span>
        </div>
        <p className="mt-4 text-sm text-fairway-900/85 leading-relaxed">
          {result.feedback.overall_summary}
        </p>
        <div className="mt-auto pt-6 grid grid-cols-2 gap-3 text-center">
          <div className="rounded-xl bg-fairway-50 p-4">
            <div className="font-display text-3xl text-fairway-700">
              +{result.feedback.estimated_distance_gain_yards}
            </div>
            <div className="text-[11px] uppercase tracking-wider text-fairway-700/70 mt-1">
              yards if fixed
            </div>
          </div>
          <div className="rounded-xl bg-fairway-50 p-4">
            <div className="font-display text-3xl text-fairway-700">
              −{result.feedback.estimated_handicap_improvement_strokes}
            </div>
            <div className="text-[11px] uppercase tracking-wider text-fairway-700/70 mt-1">
              strokes
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
