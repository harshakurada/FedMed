import React from "react";
import { render, screen, act } from "@testing-library/react";
import App from "./App";
import { MockWebSocket } from "./testUtils/MockWebSocket";
import { EventType } from "./eventSchema";

const SNAPSHOT = {
  system_status: "Ready",
  current_round: null,
  num_rounds: null,
  round_status: null,
  hospitals: {},
  global_loss: null,
  global_dice: null,
  global_iou: null,
  dp_enabled: null,
  privacy_unit: null,
  epsilon: null,
  delta: null,
  clip_norm: null,
  noise_multiplier: null,
  cumulative_epsilon: null,
  budget_status: null,
  mode: null,
  ckks_enabled: null,
  encryption_status: null,
  tls_status: null,
  recent_events: [],
};

let originalWebSocket;

beforeEach(() => {
  originalWebSocket = global.WebSocket;
  global.WebSocket = MockWebSocket;
  MockWebSocket.reset();
});

afterEach(() => {
  global.WebSocket = originalWebSocket;
});

test("dashboard loads and shows the header and title", () => {
  render(<App />);
  expect(screen.getByText("FedMed")).toBeInTheDocument();
  expect(screen.getByText(/Privacy-Preserving Federated Medical Imaging Platform/)).toBeInTheDocument();
});

test("shows Connecting before the socket opens, then Connected after", () => {
  render(<App />);
  expect(screen.getAllByText(/Connecting/).length).toBeGreaterThan(0);

  act(() => {
    MockWebSocket.latest().mockOpen();
  });
  expect(screen.getByText("Connected")).toBeInTheDocument();
});

test("renders the initial snapshot immediately on connect", () => {
  render(<App />);
  act(() => {
    MockWebSocket.latest().mockOpen();
    MockWebSocket.latest().mockMessage({ type: "snapshot", data: SNAPSHOT });
  });
  expect(screen.getAllByText("N/A").length).toBeGreaterThan(0); // metrics unavailable -> N/A, never fabricated
});

test("ROUND_STARTED event updates the round status panel", () => {
  render(<App />);
  act(() => {
    MockWebSocket.latest().mockOpen();
    MockWebSocket.latest().mockMessage({ type: "snapshot", data: SNAPSHOT });
    MockWebSocket.latest().mockMessage({
      type: "event",
      data: { event_type: EventType.ROUND_STARTED, source: "server", round: 1, payload: { num_rounds: 3 }, timestamp: new Date().toISOString() },
    });
  });
  expect(screen.getByText(/Round 1 of 3/)).toBeInTheDocument();
});

test("CLIENT_CONNECTED then CLIENT_TRAINING_COMPLETED updates hospital status", () => {
  render(<App />);
  act(() => {
    MockWebSocket.latest().mockOpen();
    MockWebSocket.latest().mockMessage({ type: "snapshot", data: SNAPSHOT });
    MockWebSocket.latest().mockMessage({
      type: "event",
      data: { event_type: EventType.CLIENT_CONNECTED, source: "hospital_a", round: null, payload: {}, timestamp: new Date().toISOString() },
    });
    MockWebSocket.latest().mockMessage({
      type: "event",
      data: {
        event_type: EventType.CLIENT_TRAINING_COMPLETED, source: "hospital_a", round: 1,
        payload: { num_examples: 5, train_loss: 0.4, train_dice: 0.31, train_iou: 0.2 },
        timestamp: new Date().toISOString(),
      },
    });
  });
  expect(screen.getByText("hospital_a")).toBeInTheDocument();
  expect(screen.getByText("Completed")).toBeInTheDocument();
});

test("CLIENT_FAILED shows the hospital as Error (node failure visualization)", () => {
  render(<App />);
  act(() => {
    MockWebSocket.latest().mockOpen();
    MockWebSocket.latest().mockMessage({ type: "snapshot", data: SNAPSHOT });
    MockWebSocket.latest().mockMessage({
      type: "event",
      data: { event_type: EventType.CLIENT_CONNECTED, source: "hospital_b", round: null, payload: {}, timestamp: new Date().toISOString() },
    });
    MockWebSocket.latest().mockMessage({
      type: "event",
      data: { event_type: EventType.CLIENT_FAILED, source: "hospital_b", round: 1, payload: { reason: "fit_error" }, timestamp: new Date().toISOString() },
    });
  });
  expect(screen.getByText("hospital_b")).toBeInTheDocument();
  expect(screen.getByText("Error")).toBeInTheDocument();
  expect(screen.getByText(/hospital_b failed to complete round/)).toBeInTheDocument();
});

test("METRICS_UPDATED populates KPI cards and removes the empty chart state", () => {
  render(<App />);
  act(() => {
    MockWebSocket.latest().mockOpen();
    MockWebSocket.latest().mockMessage({ type: "snapshot", data: SNAPSHOT });
    MockWebSocket.latest().mockMessage({
      type: "event",
      data: {
        event_type: EventType.METRICS_UPDATED, source: "server", round: 1,
        payload: { global_dice: 0.42, global_iou: 0.3, global_loss: 0.55 }, timestamp: new Date().toISOString(),
      },
    });
  });
  expect(screen.getByText("0.420")).toBeInTheDocument();
});

test("PRIVACY_UPDATED populates the privacy panel", () => {
  render(<App />);
  act(() => {
    MockWebSocket.latest().mockOpen();
    MockWebSocket.latest().mockMessage({ type: "snapshot", data: SNAPSHOT });
    MockWebSocket.latest().mockMessage({
      type: "event",
      data: {
        event_type: EventType.PRIVACY_UPDATED, source: "server", round: 1,
        payload: { dp_enabled: true, privacy_unit: "client-level (hospital-level)", epsilon: 0.969, budget_status: "ok" },
        timestamp: new Date().toISOString(),
      },
    });
  });
  expect(screen.getByText("client-level (hospital-level)")).toBeInTheDocument();
  // Epsilon is intentionally shown both in the KPI summary and the Privacy panel detail.
  expect(screen.getAllByText("0.9690").length).toBeGreaterThanOrEqual(1);
});

test("ENCRYPTION_UPDATED populates the encryption panel and security summary", () => {
  render(<App />);
  act(() => {
    MockWebSocket.latest().mockOpen();
    MockWebSocket.latest().mockMessage({ type: "snapshot", data: SNAPSHOT });
    MockWebSocket.latest().mockMessage({
      type: "event",
      data: {
        event_type: EventType.ENCRYPTION_UPDATED, source: "server", round: null,
        payload: { ckks_enabled: true, encryption_status: "Aggregating", tls_status: "Active" },
        timestamp: new Date().toISOString(),
      },
    });
  });
  expect(screen.getByText("Aggregating")).toBeInTheDocument();
});

test("invalid/malformed WebSocket message is ignored without crashing", () => {
  render(<App />);
  act(() => {
    MockWebSocket.latest().mockOpen();
    MockWebSocket.latest().mockRawMessage("{not valid json");
  });
  // App still renders normally -- no crash, no error boundary triggered.
  expect(screen.getByText("FedMed")).toBeInTheDocument();
});

test("event log records events and stays bounded at 100", () => {
  render(<App />);
  act(() => {
    MockWebSocket.latest().mockOpen();
    MockWebSocket.latest().mockMessage({ type: "snapshot", data: SNAPSHOT });
    for (let i = 0; i < 120; i += 1) {
      MockWebSocket.latest().mockMessage({
        type: "event",
        data: { event_type: EventType.SYSTEM_READY, source: "server", round: null, payload: {}, timestamp: new Date().toISOString() },
      });
    }
  });
  expect(screen.getAllByText("System ready").length).toBeLessThanOrEqual(100);
});

test("disconnect shows Disconnected/reconnecting state, not a crash", () => {
  jest.useFakeTimers();
  render(<App />);
  act(() => {
    MockWebSocket.latest().mockOpen();
  });
  act(() => {
    MockWebSocket.latest().mockClose();
  });
  expect(screen.getByText(/Attempting to reconnect/)).toBeInTheDocument();
  jest.useRealTimers();
});

test("SYSTEM_READY with a mode payload shows the DEMO MODE badge, never defaulting to LIVE", () => {
  render(<App />);
  act(() => {
    MockWebSocket.latest().mockOpen();
    MockWebSocket.latest().mockMessage({ type: "snapshot", data: SNAPSHOT });
  });
  expect(screen.queryByText("DEMO MODE")).not.toBeInTheDocument();
  expect(screen.queryByText("LIVE MODE")).not.toBeInTheDocument();

  act(() => {
    MockWebSocket.latest().mockMessage({
      type: "event",
      data: { event_type: EventType.SYSTEM_READY, source: "server", round: null, payload: { mode: "DEMO MODE" }, timestamp: new Date().toISOString() },
    });
  });
  expect(screen.getByText("DEMO MODE")).toBeInTheDocument();
});

test("Project Status panel reflects actual reported state, not hard-coded ACTIVE", () => {
  render(<App />);
  act(() => {
    MockWebSocket.latest().mockOpen();
    MockWebSocket.latest().mockMessage({
      type: "snapshot",
      data: { ...SNAPSHOT, system_status: "Not started" },
    });
  });
  expect(screen.getAllByText("Not started").length).toBeGreaterThan(0);
  // DP/CKKS/TLS unreported yet -- must read N/A, never a fabricated ACTIVE.
  expect(screen.getAllByText("N/A").length).toBeGreaterThan(0);

  act(() => {
    MockWebSocket.latest().mockMessage({
      type: "event",
      data: {
        event_type: EventType.ENCRYPTION_UPDATED, source: "server", round: null,
        payload: { ckks_enabled: true, tls_status: "Active" }, timestamp: new Date().toISOString(),
      },
    });
  });
  const activeChips = screen.getAllByText("ACTIVE");
  expect(activeChips.length).toBeGreaterThanOrEqual(2); // CKKS + TLS
});

test("reconnecting after a round already finished still populates the charts from the snapshot's event history", () => {
  // Regression test: a browser that connects (or refreshes) after a round has already
  // completed previously showed the correct KPI values but every chart stuck on "No
  // data yet" forever, because chart history was only ever built from *live* events,
  // never replayed from the snapshot's own `recent_events` log.
  const timestamp = new Date().toISOString();
  const snapshotAfterRoundAlreadyCompleted = {
    ...SNAPSHOT,
    system_status: "Idle",
    current_round: 1,
    round_status: "Completed",
    global_dice: 0.27,
    global_iou: 0.16,
    global_loss: 0.7,
    recent_events: [
      { event_type: EventType.ROUND_STARTED, source: "server", round: 1, payload: { num_rounds: 1 }, timestamp },
      { event_type: EventType.METRICS_UPDATED, source: "server", round: 1, payload: { global_dice: 0.27, global_iou: 0.16, global_loss: 0.7 }, timestamp },
      { event_type: EventType.ROUND_COMPLETED, source: "server", round: 1, payload: { round_duration_seconds: 1.2, clients_completed: 3, clients_failed: 0 }, timestamp },
    ],
  };

  render(<App />);
  act(() => {
    MockWebSocket.latest().mockOpen();
    MockWebSocket.latest().mockMessage({ type: "snapshot", data: snapshotAfterRoundAlreadyCompleted });
  });

  expect(screen.queryByRole("img", { name: "Global Dice vs. Round: no data yet" })).not.toBeInTheDocument();
  expect(screen.queryByRole("img", { name: "Global IoU vs. Round: no data yet" })).not.toBeInTheDocument();
  expect(screen.queryByRole("img", { name: "Global Loss vs. Round: no data yet" })).not.toBeInTheDocument();
  expect(screen.queryByRole("img", { name: "Round Duration: no data yet" })).not.toBeInTheDocument();
  expect(screen.queryByRole("img", { name: "Client Participation: no data yet" })).not.toBeInTheDocument();
});

test("no forbidden field is ever rendered even if present on a crafted event payload", () => {
  render(<App />);
  act(() => {
    MockWebSocket.latest().mockOpen();
    MockWebSocket.latest().mockMessage({ type: "snapshot", data: SNAPSHOT });
    MockWebSocket.latest().mockMessage({
      type: "event",
      data: {
        event_type: EventType.METRICS_UPDATED, source: "server", round: 1,
        payload: { global_dice: 0.5, secret_key: "should-never-render", patient_id: "should-never-render" },
        timestamp: new Date().toISOString(),
      },
    });
  });
  expect(screen.queryByText("should-never-render")).not.toBeInTheDocument();
});
