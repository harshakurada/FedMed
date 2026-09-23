import React from "react";

const STATUS_ICON = {
  Online: "●",
  Training: "◐",
  Completed: "✓",
  Offline: "○",
  Error: "✕",
};

const STATUS_CLASS = {
  Online: "fm-status-ok",
  Training: "fm-status-active",
  Completed: "fm-status-ok",
  Offline: "fm-status-muted",
  Error: "fm-status-error",
};

function formatNumber(value, digits = 3) {
  return value === null || value === undefined ? "N/A" : Number(value).toFixed(digits);
}

export default function HospitalStatus({ hospitals }) {
  // Use Object.entries so the map key is the hospital ID string from the
  // state object key -- always defined, unlike h.hospitalId which would be
  // undefined if the backend ever sends hospital_id: null.
  const entries = Object.entries(hospitals);

  return (
    <section className="fm-panel" aria-label="Hospital status">
      <h2>Hospital Status</h2>
      {entries.length === 0 ? (
        <p className="fm-empty">No data yet</p>
      ) : (
        <div className="fm-hospital-grid">
          {entries.map(([id, h]) => {
            const statusClass = STATUS_CLASS[h.connectionStatus] || "fm-status-muted";
            return (
              <article key={id} className="fm-hospital-card" aria-label={`${id} status`}>
                <header>
                  <strong>{id}</strong>
                  <span className={`fm-status-pill ${statusClass}`}>
                    <span aria-hidden="true">{STATUS_ICON[h.connectionStatus] || "?"}</span>{" "}
                    {h.connectionStatus || "N/A"}
                  </span>
                </header>
                <dl>
                  <div>
                    <dt>Round</dt>
                    <dd>{h.currentRound ?? "N/A"}</dd>
                  </div>
                  <div>
                    <dt>Samples</dt>
                    <dd>{h.numExamples ?? "N/A"}</dd>
                  </div>
                  <div>
                    <dt>Local Dice</dt>
                    <dd>{formatNumber(h.trainDice)}</dd>
                  </div>
                  <div>
                    <dt>Local IoU</dt>
                    <dd>{formatNumber(h.trainIou)}</dd>
                  </div>
                  <div>
                    <dt>Local Loss</dt>
                    <dd>{formatNumber(h.trainLoss)}</dd>
                  </div>
                  <div>
                    <dt>Last update</dt>
                    <dd>{h.lastUpdate ? new Date(h.lastUpdate).toLocaleTimeString() : "N/A"}</dd>
                  </div>
                </dl>
              </article>
            );
          })}
        </div>
      )}
    </section>
  );
}
