// Mirrors server/dashboard/events.py's EventType exactly -- the backend is the
// authoritative source of truth for this schema (see docs/dashboard.md's event table).
// JS and Python can't literally share one file, so this list is kept in sync by hand.
export const EventType = {
  ROUND_STARTED: "ROUND_STARTED",
  ROUND_COMPLETED: "ROUND_COMPLETED",
  CLIENT_CONNECTED: "CLIENT_CONNECTED",
  CLIENT_DISCONNECTED: "CLIENT_DISCONNECTED",
  CLIENT_TRAINING: "CLIENT_TRAINING",
  CLIENT_TRAINING_COMPLETED: "CLIENT_TRAINING_COMPLETED",
  CLIENT_FAILED: "CLIENT_FAILED",
  GLOBAL_MODEL_UPDATED: "GLOBAL_MODEL_UPDATED",
  METRICS_UPDATED: "METRICS_UPDATED",
  PRIVACY_UPDATED: "PRIVACY_UPDATED",
  ENCRYPTION_UPDATED: "ENCRYPTION_UPDATED",
  SYSTEM_WARNING: "SYSTEM_WARNING",
  SYSTEM_ERROR: "SYSTEM_ERROR",
  SYSTEM_READY: "SYSTEM_READY",
};

// A denylist the frontend also checks, purely as defense in depth -- the authoritative
// safety check is the backend's allowlist (server/dashboard/events.py), which makes it
// structurally impossible for these fields to ever be sent in the first place.
export const FORBIDDEN_PAYLOAD_KEYS = [
  "patient_id",
  "patient_name",
  "MRI",
  "image_data",
  "mask",
  "raw_model_weights",
  "gradient",
  "secret_key",
  "private_key",
  "ckks_secret",
  "dp_seed",
  "raw_ciphertext",
];

export function payloadContainsForbiddenField(payload) {
  if (!payload || typeof payload !== "object") return false;
  return FORBIDDEN_PAYLOAD_KEYS.some((key) => key in payload);
}
