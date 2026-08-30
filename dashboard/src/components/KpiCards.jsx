import React from "react";

function formatNumber(value, digits = 3) {
  return value === null || value === undefined || Number.isNaN(value) ? "N/A" : value.toFixed(digits);
}

function Card({ label, value, hint }) {
  return (
    <div className="fm-card">
      <span className="fm-card-label">{label}</span>
      <span className="fm-card-value">{value}</span>
      {hint && <span className="fm-card-hint">{hint}</span>}
    </div>
  );
}

export default React.memo(function KpiCards({ state }) {
  const connectedHospitals = React.useMemo(
    () => Object.values(state.hospitals).filter(
      (h) => h.connectionStatus === "Online" || h.connectionStatus === "Training" || h.connectionStatus === "Completed"
    ).length,
    [state.hospitals]
  );

  return (
    <section className="fm-kpi-grid" aria-label="System overview">
      <Card label="Current Round" value={state.currentRound != null ? `${state.currentRound}${state.numRounds ? ` / ${state.numRounds}` : ""}` : "N/A"} />
      <Card label="Global Dice" value={formatNumber(state.globalDice)} />
      <Card label="Global IoU" value={formatNumber(state.globalIou)} />
      <Card label="Global Loss" value={formatNumber(state.globalLoss)} />
      <Card
        label="Privacy Budget (ε)"
        value={state.dpEnabled ? formatNumber(state.cumulativeEpsilon ?? state.epsilon, 4) : "N/A"}
        hint={state.dpEnabled ? state.budgetStatus : "DP disabled"}
      />
      <Card label="Encryption" value={state.ckksEnabled == null ? "N/A" : state.ckksEnabled ? "CKKS Active" : "Inactive"} />
      <Card label="Connected Hospitals" value={`${connectedHospitals} / ${Object.keys(state.hospitals).length || "N/A"}`} />
      <Card label="System Status" value={state.systemStatus || "N/A"} />
    </section>
  );
});
