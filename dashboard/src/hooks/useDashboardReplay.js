import { useCallback, useEffect, useRef, useState } from "react";
import { EMPTY_STATE, applyEvent } from "./useDashboardSocket";

// Gaps between recorded events are real but uneven (4s of setup, then sub-millisecond
// bursts), so they are clamped to keep the replay watchable without reordering anything.
const MIN_GAP_MS = 350;
const MAX_GAP_MS = 1500;

/**
 * Static-hosting stand-in for useDashboardSocket: plays back the `recent_events` log of a
 * real snapshot recorded from `scripts/run_demo.py` (public/replay/demo-run.json) through
 * the exact same `applyEvent` reducer. No numbers are generated here -- every value shown
 * is one the real backend emitted during that recorded run.
 */
export function useDashboardReplay(url) {
  const [connectionStatus, setConnectionStatus] = useState("Loading replay");
  const [state, setState] = useState(EMPTY_STATE);
  const [runId, setRunId] = useState(0);
  const timerRef = useRef(null);

  const restart = useCallback(() => setRunId((n) => n + 1), []);

  useEffect(() => {
    let cancelled = false;
    setState(EMPTY_STATE);
    setConnectionStatus("Loading replay");

    fetch(url)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((snapshot) => {
        if (cancelled) return;
        const events = snapshot.recent_events || [];
        setConnectionStatus("Replaying");
        let i = 0;
        const step = () => {
          if (cancelled) return;
          if (i >= events.length) {
            setConnectionStatus("Replay finished");
            return;
          }
          const event = events[i];
          setState((prev) => applyEvent(prev, event));
          i += 1;
          const next = events[i];
          const gap = next ? Date.parse(next.timestamp) - Date.parse(event.timestamp) : 0;
          timerRef.current = setTimeout(step, Math.min(Math.max(gap, MIN_GAP_MS), MAX_GAP_MS));
        };
        step();
      })
      .catch((err) => {
        if (cancelled) return;
        console.warn("FedMed dashboard: could not load replay.", err);
        setConnectionStatus("Replay unavailable");
      });

    return () => {
      cancelled = true;
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [url, runId]);

  return { connectionStatus, state, restart };
}
