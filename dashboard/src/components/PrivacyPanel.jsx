import React from "react";

function Row({ label, value }) {
  return (
    <div className="fm-info-row">
      <dt>{label}</dt>
      <dd>{value === null || value === undefined || value === "" ? "N/A" : String(value)}</dd>
    </div>
  );
}

export default function PrivacyPanel({ state }) {
  return (
    <section className="fm-panel" aria-label="Privacy status">
      <h2>Privacy</h2>
      <dl className="fm-info-list">
        <Row label="Differential Privacy" value={state.dpEnabled == null ? null : state.dpEnabled ? "Enabled" : "Disabled"} />
        <Row label="Privacy Unit" value={state.privacyUnit} />
        <Row label="Epsilon (ε)" value={state.epsilon != null ? Number(state.epsilon).toFixed(4) : null} />
        <Row label="Delta (δ)" value={state.delta} />
        <Row label="Noise Multiplier" value={state.noiseMultiplier} />
        <Row label="Clipping Norm" value={state.clipNorm} />
        <Row label="Budget Status" value={state.budgetStatus} />
        <Row label="Cumulative Epsilon" value={state.cumulativeEpsilon != null ? Number(state.cumulativeEpsilon).toFixed(4) : null} />
      </dl>
    </section>
  );
}
