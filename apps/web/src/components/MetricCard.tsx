import clsx from "clsx";
import type { MetricScore } from "@/lib/types";

export function MetricCard({ metric }: { metric: MetricScore }) {
  const tone = scoreTone(metric.score);
  return (
    <div className="rounded-xl bg-white/80 backdrop-blur p-5 shadow-sm border border-fairway-500/10">
      <div className="flex items-center justify-between gap-3">
        <h4 className="font-medium text-fairway-900 text-sm">{metric.label}</h4>
        <span
          className={clsx(
            "text-xs font-semibold px-2 py-0.5 rounded-full",
            tone.badge,
          )}
        >
          {metric.score}
        </span>
      </div>
      <div className="mt-3 flex items-baseline gap-2">
        <span className="font-display text-3xl text-fairway-900">
          {formatValue(metric.value, metric.name)}
        </span>
        <span className="text-xs text-fairway-700/70">
          target {metric.ideal_range[0]}–{metric.ideal_range[1]}
        </span>
      </div>
      <div className="mt-3 h-1.5 rounded-full bg-fairway-50 overflow-hidden">
        <div
          className={clsx("h-full transition-all", tone.bar)}
          style={{ width: `${metric.score}%` }}
        />
      </div>
    </div>
  );
}

function scoreTone(score: number) {
  if (score >= 85) return { badge: "bg-fairway-100 text-fairway-700", bar: "bg-fairway-600" };
  if (score >= 60) return { badge: "bg-sand-100 text-sand-500", bar: "bg-sand-500" };
  return { badge: "bg-red-100 text-red-700", bar: "bg-red-500" };
}

function formatValue(v: number, name: string): string {
  if (name === "tempo_ratio") return `${v.toFixed(2)}:1`;
  if (name === "head_sway") return v.toFixed(3);
  if (name === "weight_transfer") return `${Math.round(v * 100)}%`;
  return Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2);
}
