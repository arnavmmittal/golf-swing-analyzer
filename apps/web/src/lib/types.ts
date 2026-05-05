export type MetricScore = {
  name: string;
  label: string;
  value: number;
  score: number;
  ideal_range: [number, number];
  fault: "low" | "high" | null;
};

export type Drill = {
  name: string;
  instructions: string;
  frequency: string;
};

export type Fault = {
  metric: string;
  current_value: number;
  target_range: string;
  cause: string;
  ball_flight_impact: string;
  correction: string;
  drills: Drill[];
};

export type Feedback = {
  overall_summary: string;
  headline_fault: string;
  faults: Fault[];
  estimated_distance_gain_yards: number;
  estimated_handicap_improvement_strokes: number;
};

export type Phases = {
  address_end_frame: number;
  top_frame: number;
  impact_frame: number;
  finish_frame: number;
  handedness: "right" | "left";
};

export type AnalysisResult = {
  analysis_id: string;
  video_meta: { fps: number; width: number; height: number; total_frames: number };
  phases: Phases;
  metrics: Record<string, number>;
  scores: MetricScore[];
  overall_score: number;
  feedback: Feedback;
  annotated_video_url: string;
};
