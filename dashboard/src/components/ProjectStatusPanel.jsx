import React from "react";

// Module 13 Phase 4: "architecture status" panel. Every value below is derived from
// actual reported state -- never hard-coded to ACTIVE/CONNECTED. Before any event has
// arrived, everything correctly reads "Not started" / "N/A" / 0, not a fabricated ACTIVE.
function StatusRow({ label, value, tone }) {
  return (
    <div className="fm-security-chip">
      <span className="fm-security-chip-label">{label}</span>
      <span className={`fm-status-pill ${tone}`}>{value}</span>
    </div>
  );
}

function toneFor(active) {
  if (active == null) return "fm-status-muted";
  return active ? "fm-status-ok" : "fm-status-warn";
}

export default function ProjectStatusPanel({ state, connectionStatus }) {
  const flStarted = state.systemStatus && state.systemStatus !== "No data yet" && state.systemStatus !== "Not started";
  const tlsActive = state.tlsStatus == null ? null : state.tlsStatus.toLowerCase().includes("active");
  const hospitalCount = Object.keys(state.hospitals).length;
  const wsConnected = connectionStatus === "Connected";

  return (
    <section className="fm-panel" aria-label="Project status">
      <h2>Project Status</h2>
      <div className="fm-security-grid">
        <StatusRow
          label="Federated Learning"
          value={flStarted ? "ACTIVE" : "Not started"}
          tone={toneFor(flStarted ? true : null)}
        />
        <StatusRow
          label="Differential Privacy"
          value={state.dpEnabled == null ? "N/A" : state.dpEnabled ? "ACTIVE" : "INACTIVE"}
          tone={toneFor(state.dpEnabled)}
        />
        <StatusRow
          label="CKKS Encryption"
          value={state.ckksEnabled == null ? "N/A" : state.ckksEnabled ? "ACTIVE" : "INACTIVE"}
          tone={toneFor(state.ckksEnabled)}
        />
        <StatusRow
          label="TLS"
          value={tlsActive == null ? "N/A" : tlsActive ? "ACTIVE" : "INACTIVE"}
          tone={toneFor(tlsActive)}
        />
        <StatusRow
          label="WebSocket"
          value={wsConnected ? "CONNECTED" : "DISCONNECTED"}
          tone={toneFor(wsConnected)}
        />
        <StatusRow label="Hospitals" value={hospitalCount > 0 ? String(hospitalCount) : "N/A"} tone="fm-status-muted" />
      </div>
      <p className="fm-security-note">
        Reflects actual reported backend/connection state only -- never hard-coded.
      </p>
    </section>
  );
}
