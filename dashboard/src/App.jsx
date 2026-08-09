import React from "react";
import { useDashboardSocket } from "./hooks/useDashboardSocket";
import Header from "./components/Header";
import KpiCards from "./components/KpiCards";
import HospitalStatus from "./components/HospitalStatus";
import RoundStatus from "./components/RoundStatus";
import MetricsCharts from "./components/MetricsCharts";
import PrivacyPanel from "./components/PrivacyPanel";
import EncryptionPanel from "./components/EncryptionPanel";
import SecurityStatus from "./components/SecurityStatus";
import ProjectStatusPanel from "./components/ProjectStatusPanel";
import EventLog from "./components/EventLog";
import "./App.css";

const WS_URL = process.env.REACT_APP_DASHBOARD_WS_URL || "ws://127.0.0.1:8765";

export default function App() {
  const { connectionStatus, state } = useDashboardSocket(WS_URL);

  return (
    <div className="fm-app">
      <Header connectionStatus={connectionStatus} mode={state.mode} />

      {connectionStatus !== "Connected" && state.recentEvents.length === 0 ? (
        <p className="fm-empty fm-top-notice" role="status">
          {connectionStatus === "Connecting" && "Connecting to the FedMed backend…"}
          {connectionStatus === "Reconnecting" && "Disconnected — attempting to reconnect…"}
          {connectionStatus === "Disconnected" && "Disconnected from the FedMed backend."}
        </p>
      ) : null}

      <main>
        <KpiCards state={state} />
        <HospitalStatus hospitals={state.hospitals} />
        <RoundStatus state={state} />
        <MetricsCharts state={state} />
        <div className="fm-security-row">
          <PrivacyPanel state={state} />
          <EncryptionPanel state={state} />
          <SecurityStatus state={state} />
          <ProjectStatusPanel state={state} connectionStatus={connectionStatus} />
        </div>
        <EventLog events={state.recentEvents} />
      </main>

      <footer className="fm-footer">
        FedMed dashboard is a local development monitoring interface. It performs no training, model,
        encryption, or privacy computation itself — see docs/dashboard.md.
      </footer>
    </div>
  );
}
