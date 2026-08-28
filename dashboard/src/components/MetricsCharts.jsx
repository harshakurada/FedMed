import React from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";

// Shared dark-theme styling for every chart tooltip -- Recharts' own default tooltip is
// a light box, which would look out of place against this dashboard's dark background.
const TOOLTIP_CONTENT_STYLE = {
  background: "rgba(15, 23, 42, 0.95)",
  border: "1px solid rgba(148, 163, 184, 0.25)",
  borderRadius: "0.6rem",
  boxShadow: "0 8px 24px rgba(0,0,0,0.4)",
  color: "#e2e8f0",
};
const TOOLTIP_LABEL_STYLE = { color: "#aab4c8", fontWeight: 600, marginBottom: 4 };
const TOOLTIP_ITEM_STYLE = { color: "#e7ebf3" };
const AXIS_TICK = { fill: "#aab4c8", fontSize: 12 };
const AXIS_LABEL = { fill: "#8994ac" };

function EmptyChart({ label }) {
  return (
    <div className="fm-chart-empty" role="img" aria-label={`${label}: no data yet`}>
      No data yet
    </div>
  );
}

function LineChartPanel({ title, data, dataKey, color, unit }) {
  return (
    <div className="fm-chart-card">
      <h3>{title}</h3>
      {data.length === 0 ? (
        <EmptyChart label={title} />
      ) : (
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--fm-border)" />
            <XAxis dataKey="round" tick={AXIS_TICK} label={{ value: "Round", position: "insideBottom", offset: -4, style: AXIS_LABEL }} />
            <YAxis tick={AXIS_TICK} label={{ value: unit, angle: -90, position: "insideLeft", style: AXIS_LABEL }} />
            <Tooltip contentStyle={TOOLTIP_CONTENT_STYLE} labelStyle={TOOLTIP_LABEL_STYLE} itemStyle={TOOLTIP_ITEM_STYLE} />
            <Line type="monotone" dataKey={dataKey} stroke={color} strokeWidth={2.5} dot={{ r: 3, fill: color }} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}

export default React.memo(function MetricsCharts({ state }) {
  const metricsHistory = state.metricsHistory || [];
  const roundHistory = state.roundHistory || [];

  const privacyHistory = React.useMemo(() => {
    if (!state.dpEnabled || state.epsilon == null) {
      return [];
    }
    return metricsHistory.map((m) => ({ round: m.round, epsilon: state.epsilon }));
  }, [metricsHistory, state.dpEnabled, state.epsilon]);

  return (
    <section className="fm-panel" aria-label="Model performance charts">
      <h2>Model Performance</h2>
      <div className="fm-chart-grid">
        <LineChartPanel title="Global Dice vs. Round" data={metricsHistory} dataKey="globalDice" color="#60a5fa" unit="Dice" />
        <LineChartPanel title="Global IoU vs. Round" data={metricsHistory} dataKey="globalIou" color="#22d3ee" unit="IoU" />
        <LineChartPanel title="Global Loss vs. Round" data={metricsHistory} dataKey="globalLoss" color="#f87171" unit="Loss" />
        <LineChartPanel title="Privacy Budget (ε) vs. Round" data={privacyHistory} dataKey="epsilon" color="#c084fc" unit="Epsilon" />

        <div className="fm-chart-card">
          <h3>Round Duration vs. Round</h3>
          {roundHistory.length === 0 ? (
            <EmptyChart label="Round Duration" />
          ) : (
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={roundHistory} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--fm-border)" />
                <XAxis dataKey="round" tick={AXIS_TICK} label={{ value: "Round", position: "insideBottom", offset: -4, style: AXIS_LABEL }} />
                <YAxis tick={AXIS_TICK} label={{ value: "Seconds", angle: -90, position: "insideLeft", style: AXIS_LABEL }} />
                <Tooltip contentStyle={TOOLTIP_CONTENT_STYLE} labelStyle={TOOLTIP_LABEL_STYLE} itemStyle={TOOLTIP_ITEM_STYLE} />
                <Line type="monotone" dataKey="durationSeconds" stroke="#fb923c" strokeWidth={2.5} dot={{ r: 3, fill: "#fb923c" }} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>

        <div className="fm-chart-card">
          <h3>Client Participation by Round</h3>
          {roundHistory.length === 0 ? (
            <EmptyChart label="Client Participation" />
          ) : (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={roundHistory} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--fm-border)" />
                <XAxis dataKey="round" tick={AXIS_TICK} label={{ value: "Round", position: "insideBottom", offset: -4, style: AXIS_LABEL }} />
                <YAxis allowDecimals={false} tick={AXIS_TICK} />
                <Tooltip contentStyle={TOOLTIP_CONTENT_STYLE} labelStyle={TOOLTIP_LABEL_STYLE} itemStyle={TOOLTIP_ITEM_STYLE} />
                <Legend wrapperStyle={{ color: "#94a3b8", fontSize: 13 }} />
                <Bar dataKey="clientsCompleted" name="Completed" fill="#4ade80" isAnimationActive={false} radius={[4, 4, 0, 0]} />
                <Bar dataKey="clientsFailed" name="Failed" fill="#f87171" isAnimationActive={false} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>
    </section>
  );
});
