import React from "react";
import { EventType } from "../eventSchema";

const SEVERITY = {
  [EventType.SYSTEM_ERROR]: "ERROR",
  [EventType.CLIENT_FAILED]: "WARNING",
  [EventType.SYSTEM_WARNING]: "WARNING",
};

function severityOf(eventType) {
  return SEVERITY[eventType] || "INFO";
}

function describe(event) {
  const { event_type: type, source, round } = event;
  const roundText = round != null ? ` (round ${round})` : "";
  switch (type) {
    case EventType.SYSTEM_READY:
      return "System ready";
    case EventType.ROUND_STARTED:
      return `Round ${round} started`;
    case EventType.ROUND_COMPLETED:
      return `Round ${round} completed`;
    case EventType.CLIENT_CONNECTED:
      return `${source} connected`;
    case EventType.CLIENT_DISCONNECTED:
      return `${source} disconnected`;
    case EventType.CLIENT_TRAINING:
      return `${source} started local training${roundText}`;
    case EventType.CLIENT_TRAINING_COMPLETED:
      return `${source} completed local training${roundText}`;
    case EventType.CLIENT_FAILED:
      return `${source} failed to complete round${roundText}`;
    case EventType.GLOBAL_MODEL_UPDATED:
      return `Global model updated${roundText}`;
    case EventType.METRICS_UPDATED:
      return `Metrics updated${roundText}`;
    case EventType.PRIVACY_UPDATED:
      return `Privacy budget updated${roundText}`;
    case EventType.ENCRYPTION_UPDATED:
      return "Encryption status updated";
    case EventType.SYSTEM_WARNING:
      return event.payload?.message || "System warning";
    case EventType.SYSTEM_ERROR:
      return event.payload?.message || "System error";
    default:
      return type;
  }
}

export default React.memo(function EventLog({ events }) {
  const ordered = React.useMemo(() => [...events].reverse(), [events]); // newest first

  return (
    <section className="fm-panel" aria-label="System events">
      <h2>System Events</h2>
      {ordered.length === 0 ? (
        <p className="fm-empty">No data yet</p>
      ) : (
        <ul className="fm-event-log">
          {ordered.map((event, index) => {
            const severity = severityOf(event.event_type);
            const severityLower = severity.toLowerCase();
            return (
              <li key={`${event.timestamp}-${index}`} className={`fm-event fm-severity-${severityLower}`}>
                <span className="fm-event-time">{new Date(event.timestamp).toLocaleTimeString()}</span>
                <span className={`fm-event-severity fm-severity-badge-${severityLower}`}>{severity}</span>
                <span className="fm-event-text">{describe(event)}</span>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
});
