import React from "react";

const STATUS_LABEL = {
  Connected: "Connected",
  Connecting: "Connecting…",
  Reconnecting: "Disconnected — Attempting to reconnect…",
  Disconnected: "Disconnected",
};

export default function Header({ connectionStatus }) {
  const isConnected = connectionStatus === "Connected";
  return (
    <header className="fm-header">
      <div>
        <h1>FedMed</h1>
        <p className="fm-subtitle">Privacy-Preserving Federated Medical Imaging Platform</p>
      </div>
      <div
        className={`fm-connection-badge ${isConnected ? "fm-status-ok" : "fm-status-warn"}`}
        role="status"
        aria-live="polite"
      >
        <span className="fm-status-dot" aria-hidden="true" />
        {STATUS_LABEL[connectionStatus] || connectionStatus}
      </div>
    </header>
  );
}
