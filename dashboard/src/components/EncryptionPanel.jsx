import React from "react";

function Row({ label, value }) {
  return (
    <div className="fm-info-row">
      <dt>{label}</dt>
      <dd>{value === null || value === undefined || value === "" ? "N/A" : String(value)}</dd>
    </div>
  );
}

export default function EncryptionPanel({ state }) {
  return (
    <section className="fm-panel" aria-label="Encryption status">
      <h2>Encryption</h2>
      <dl className="fm-info-list">
        <Row label="CKKS Homomorphic Encryption" value={state.ckksEnabled == null ? null : state.ckksEnabled ? "Enabled" : "Disabled"} />
        <Row label="Encryption Status" value={state.encryptionStatus} />
        <Row label="Aggregation Mode" value={state.ckksEnabled ? "Encrypted (ciphertext)" : state.ckksEnabled == null ? null : "Plaintext"} />
        <Row label="TLS Status" value={state.tlsStatus} />
      </dl>
    </section>
  );
}
