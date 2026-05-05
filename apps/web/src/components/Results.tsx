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
        <h2 className="font-display text-2xl text-fairway-900 mb-1">
          Measured from this view
        </h2>
        <p className="text-sm text-fairway-700/70 mb-4">
          {result.video.view === "dtl"
            ? "Down-the-line. Five metrics this angle can validly measure."
            : "Face-on. Five metrics this angle can validly measure."}
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {result.scores.map((s) => (
            <MetricCard key={s.name} metric={s} />
          ))}
        </div>
        {result.metrics_not_measured_from_this_view.length > 0 && (
          <div className="mt-5 rounded-xl bg-sand-50 border border-sand-100 p-4">
            <div className="text-[10px] font-semibold tracking-wider uppercase text-sand-500 mb-2">
              Not measurable from this view
            </div>
            <ul className="text-sm text-fairway-900/80 space-y-1">
              {result.metrics_not_measured_from_this_view.map((m) => (
                <li key={m.name}>
                  <span className="font-medium">{m.label}</span>{" "}
                  <span className="text-fairway-700/70">
                    — needs {m.valid_views.join(" or ")}
                  </span>
                </li>
              ))}
            </ul>
            <p className="mt-2 text-xs text-fairway-700/60">
              For a complete audit, also upload a {result.video.view === "dtl" ? "face-on" : "down-the-line"} video.
            </p>
          </div>
        )}
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
