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
            <XAxis dataKey="round" label={{ value: "Round", position: "insideBottom", offset: -4 }} />
            <YAxis label={{ value: unit, angle: -90, position: "insideLeft" }} />
            <Tooltip />
            <Line type="monotone" dataKey={dataKey} stroke={color} strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
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
    return metricsHistory
      .map((m) => ({ round: m.round, epsilon: state.dpEnabled ? state.epsilon : null }))
      .filter((m) => m.epsilon != null);
  }, [metricsHistory, state.dpEnabled, state.epsilon]);

  return (
    <section className="fm-panel" aria-label="Model performance charts">
      <h2>Model Performance</h2>
      <div className="fm-chart-grid">
        <LineChartPanel title="Global Dice vs. Round" data={metricsHistory} dataKey="globalDice" color="#2563eb" unit="Dice" />
        <LineChartPanel title="Global IoU vs. Round" data={metricsHistory} dataKey="globalIou" color="#0891b2" unit="IoU" />
        <LineChartPanel title="Global Loss vs. Round" data={metricsHistory} dataKey="globalLoss" color="#dc2626" unit="Loss" />
        <LineChartPanel title="Privacy Budget (ε) vs. Round" data={privacyHistory} dataKey="epsilon" color="#7c3aed" unit="Epsilon" />

        <div className="fm-chart-card">
          <h3>Round Duration vs. Round</h3>
          {roundHistory.length === 0 ? (
            <EmptyChart label="Round Duration" />
          ) : (
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={roundHistory} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--fm-border)" />
                <XAxis dataKey="round" label={{ value: "Round", position: "insideBottom", offset: -4 }} />
                <YAxis label={{ value: "Seconds", angle: -90, position: "insideLeft" }} />
                <Tooltip />
                <Line type="monotone" dataKey="durationSeconds" stroke="#ea580c" strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
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
                <XAxis dataKey="round" label={{ value: "Round", position: "insideBottom", offset: -4 }} />
                <YAxis allowDecimals={false} />
                <Tooltip />
                <Legend />
                <Bar dataKey="clientsCompleted" name="Completed" fill="#16a34a" isAnimationActive={false} />
                <Bar dataKey="clientsFailed" name="Failed" fill="#dc2626" isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>
    </section>
  );
});
