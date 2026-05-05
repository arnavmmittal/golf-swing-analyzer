import type { Fault } from "@/lib/types";

export function FaultCard({ fault, headline }: { fault: Fault; headline: boolean }) {
  return (
    <article className="rounded-2xl bg-white/90 backdrop-blur p-6 shadow-sm border border-fairway-500/10">
      <header className="flex items-center justify-between gap-4 mb-3">
        <div>
          {headline && (
            <span className="inline-block text-[10px] font-semibold tracking-wider uppercase text-sand-500 mb-1">
              Headline fix
            </span>
          )}
          <h3 className="font-display text-2xl text-fairway-900">
            {prettyMetric(fault.metric)}
          </h3>
          <p className="text-xs text-fairway-700/70 mt-0.5">
            current {fault.current_value} · target {fault.target_range}
          </p>
        </div>
      </header>

      <Section label="Cause" body={fault.cause} />
      <Section label="What you're seeing" body={fault.ball_flight_impact} />
      <Section label="The feel" body={fault.correction} emphasis />

      <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-3">
        {fault.drills.map((d) => (
          <div
            key={d.name}
            className="rounded-xl bg-fairway-50/60 p-4 border border-fairway-500/10"
          >
            <div className="flex items-center justify-between gap-2 mb-1">
              <h5 className="font-semibold text-fairway-900 text-sm">{d.name}</h5>
              <span className="text-[10px] uppercase tracking-wider text-fairway-700/70">
                {d.frequency}
              </span>
            </div>
            <p className="text-sm text-fairway-900/80 leading-relaxed">
              {d.instructions}
            </p>
          </div>
        ))}
      </div>
    </article>
  );
}

function Section({
  label,
  body,
  emphasis = false,
}: {
  label: string;
  body: string;
  emphasis?: boolean;
}) {
  return (
    <div className="mt-3">
      <div className="text-[10px] font-semibold tracking-wider uppercase text-fairway-700/60 mb-1">
        {label}
      </div>
      <p
        className={
          emphasis
            ? "font-display text-lg text-fairway-900 leading-snug"
            : "text-sm text-fairway-900/80 leading-relaxed"
        }
      >
        {body}
      </p>
    </div>
  );
}

function prettyMetric(name: string): string {
  return name
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}
