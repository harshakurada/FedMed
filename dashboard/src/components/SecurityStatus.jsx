import React from "react";

function StatusChip({ label, active }) {
  const text = active == null ? "N/A" : active ? "Active" : "Inactive";
  const cls = active == null ? "fm-status-muted" : active ? "fm-status-ok" : "fm-status-warn";
  return (
    <div className="fm-security-chip">
      <span className="fm-security-chip-label">{label}</span>
      <span className={`fm-status-pill ${cls}`}>{text}</span>
    </div>
  );
}

export default function SecurityStatus({ state }) {
  const tlsActive = state.tlsStatus == null ? null : state.tlsStatus.toLowerCase().includes("active");
  return (
    <section className="fm-panel" aria-label="Security summary">
      <h2>Security Summary</h2>
      <div className="fm-security-grid">
        <StatusChip label="Transport (TLS)" active={tlsActive} />
        <StatusChip label="Aggregation (CKKS)" active={state.ckksEnabled} />
        <StatusChip label="Privacy (DP)" active={state.dpEnabled} />
      </div>
      <p className="fm-security-note">
        This summary reflects reported component status only. It is not a certification of overall system
        security.
      </p>
    </section>
  );
}
