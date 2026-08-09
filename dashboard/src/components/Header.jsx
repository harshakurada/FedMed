import React from "react";

const STATUS_LABEL = {
  Connected: "Connected",
  Connecting: "Connecting…",
  Reconnecting: "Disconnected — Attempting to reconnect…",
  Disconnected: "Disconnected",
};

// Module 13: the dashboard must never be mistaken for a real experiment when it isn't
// one. `mode` comes straight from the backend's own SYSTEM_READY event (server/demo/
// run_demo.py sets it honestly based on which data source is actually configured) --
// never guessed or defaulted to "LIVE" on the frontend.
const MODE_CLASS = {
  "LIVE MODE": "fm-mode-live",
  "DEMO MODE": "fm-mode-demo",
  "SIMULATION MODE": "fm-mode-demo",
};

function ModeBadge({ mode }) {
  if (!mode) return null;
  return (
    <div className={`fm-mode-badge ${MODE_CLASS[mode] || "fm-mode-demo"}`} role="status">
      {mode}
    </div>
  );
}

export default function Header({ connectionStatus, mode }) {
  const isConnected = connectionStatus === "Connected";
  return (
    <header className="fm-header">
      <div>
        <h1>FedMed</h1>
        <p className="fm-subtitle">Privacy-Preserving Federated Medical Imaging Platform</p>
      </div>
      <div className="fm-header-badges">
        <ModeBadge mode={mode} />
        <div
          className={`fm-connection-badge ${isConnected ? "fm-status-ok" : "fm-status-warn"}`}
          role="status"
          aria-live="polite"
        >
          <span className="fm-status-dot" aria-hidden="true" />
          {STATUS_LABEL[connectionStatus] || connectionStatus}
        </div>
      </div>
    </header>
  );
}
