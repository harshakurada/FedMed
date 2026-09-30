import React from "react";
import { useDashboardSocket } from "./hooks/useDashboardSocket";
import { useDashboardReplay } from "./hooks/useDashboardReplay";
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
// Set at build time for static hosting (e.g. Vercel), where no Python backend exists.
const REPLAY_URL = process.env.REACT_APP_REPLAY_URL;

function LiveNotice({ connectionStatus }) {
  if (connectionStatus === "Connected") return null;
  return (
    <p className="fm-empty fm-top-notice" role="status">
      {connectionStatus === "Connecting" && "Connecting to the FedMed backend…"}
      {connectionStatus === "Reconnecting" && "Disconnected — attempting to reconnect…"}
      {connectionStatus === "Disconnected" && "Disconnected from the FedMed backend."}
    </p>
  );
}

function ReplayNotice({ connectionStatus, restart }) {
  return (
    <p className="fm-empty fm-top-notice" role="status">
      {connectionStatus === "Replay unavailable"
        ? "Could not load the recorded demo run."
        : "Replay of a recorded DEMO MODE run (real DP, CKKS and mutual TLS on synthetic data). " +
          "Run scripts/run_demo.py locally for a live session."}{" "}
      {connectionStatus === "Replay finished" ? (
        <button type="button" className="fm-replay-button" onClick={restart}>
          Replay again
        </button>
      ) : null}
    </p>
  );
}

function Dashboard({ connectionStatus, state, notice }) {
  return (
    <div className="fm-app">
      <Header connectionStatus={connectionStatus} mode={state.mode} />
      {notice}
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

function LiveApp() {
  const { connectionStatus, state } = useDashboardSocket(WS_URL);
  return (
    <Dashboard
      connectionStatus={connectionStatus}
      state={state}
      notice={<LiveNotice connectionStatus={connectionStatus} />}
    />
  );
}

function ReplayApp() {
  const { connectionStatus, state, restart } = useDashboardReplay(REPLAY_URL);
  return (
    <Dashboard
      connectionStatus={connectionStatus}
      state={state}
      notice={<ReplayNotice connectionStatus={connectionStatus} restart={restart} />}
    />
  );
}

export default function App() {
  return REPLAY_URL ? <ReplayApp /> : <LiveApp />;
}
