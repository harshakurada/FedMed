import React from "react";

export default function RoundStatus({ state }) {
  const hospitals = Object.values(state.hospitals);
  const completed = hospitals.filter((h) => h.connectionStatus === "Completed").length;
  const failed = hospitals.filter((h) => h.connectionStatus === "Error").length;
  const total = hospitals.length;
  const progressPct = total > 0 ? Math.round((completed / total) * 100) : 0;

  return (
    <section className="fm-panel" aria-label="Federated training progress">
      <h2>Federated Training</h2>
      <p className="fm-round-summary">
        {state.currentRound != null ? (
          <>
            Round {state.currentRound}
            {state.numRounds ? ` of ${state.numRounds}` : ""} — {completed} / {total || "N/A"} clients completed
            {failed > 0 && <span className="fm-status-error"> ({failed} failed)</span>}
          </>
        ) : (
          "No round in progress yet"
        )}
      </p>
      <div
        className="fm-progress-track"
        role="progressbar"
        aria-valuenow={progressPct}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label="Round completion progress"
      >
        <div className="fm-progress-fill" style={{ width: `${progressPct}%` }} />
      </div>
      <p className="fm-round-status">
        Status: <strong>{state.roundStatus || "N/A"}</strong>
      </p>
    </section>
  );
}
