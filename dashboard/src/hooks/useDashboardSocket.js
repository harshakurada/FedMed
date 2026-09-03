import { useEffect, useRef, useState, useCallback } from "react";
import { EventType } from "../eventSchema";

const DEFAULT_WS_URL = "ws://127.0.0.1:8765";
const MAX_RECENT_EVENTS = 100;
const MAX_CHART_HISTORY = 200;
const INITIAL_BACKOFF_MS = 1000;
const MAX_BACKOFF_MS = 30000;

const EMPTY_STATE = {
  systemStatus: "No data yet",
  mode: null, // "LIVE MODE" / "DEMO MODE" / "SIMULATION MODE" -- Module 13
  currentRound: null,
  numRounds: null,
  roundStatus: null,
  hospitals: {},
  globalLoss: null,
  globalDice: null,
  globalIou: null,
  dpEnabled: null,
  privacyUnit: null,
  epsilon: null,
  delta: null,
  clipNorm: null,
  noiseMultiplier: null,
  cumulativeEpsilon: null,
  budgetStatus: null,
  ckksEnabled: null,
  encryptionStatus: null,
  tlsStatus: null,
  recentEvents: [],
  metricsHistory: [], // [{round, globalDice, globalIou, globalLoss}]
  roundHistory: [], // [{round, durationSeconds, clientsCompleted, clientsFailed}]
};

// Reconstructs chart history from the snapshot's own `recent_events` log. Without this,
// a browser that connects (or reconnects) *after* a round already finished would get the
// final KPI values (those are plain snapshot fields) but empty charts forever -- the
// history arrays below were previously only ever built incrementally from live events,
// never replayed from `recent_events`, even though the backend already sends up to the
// last 100 events in every snapshot.
function historyFromEvents(events) {
  const metricsHistory = [];
  const roundHistory = [];
  for (const event of events) {
    const { event_type: type, round, payload = {} } = event;
    if (type === EventType.METRICS_UPDATED) {
      metricsHistory.push({
        round,
        globalDice: payload.global_dice ?? null,
        globalIou: payload.global_iou ?? null,
        globalLoss: payload.global_loss ?? null,
      });
    } else if (type === EventType.ROUND_COMPLETED) {
      roundHistory.push({
        round,
        durationSeconds: payload.round_duration_seconds ?? null,
        clientsCompleted: payload.clients_completed ?? null,
        clientsFailed: payload.clients_failed ?? null,
      });
    }
  }
  return {
    metricsHistory: metricsHistory.slice(-MAX_CHART_HISTORY),
    roundHistory: roundHistory.slice(-MAX_CHART_HISTORY),
  };
}

function snapshotToState(data) {
  const hospitals = {};
  for (const [id, h] of Object.entries(data.hospitals || {})) {
    hospitals[id] = {
      hospitalId: h.hospital_id,
      connectionStatus: h.connection_status,
      currentRound: h.current_round,
      lastUpdate: h.last_update,
      numExamples: h.num_examples,
      trainLoss: h.train_loss,
      trainDice: h.train_dice,
      trainIou: h.train_iou,
    };
  }
  const { metricsHistory, roundHistory } = historyFromEvents(data.recent_events || []);
  return {
    ...EMPTY_STATE,
    systemStatus: data.system_status ?? EMPTY_STATE.systemStatus,
    mode: data.mode ?? null,
    currentRound: data.current_round ?? null,
    numRounds: data.num_rounds ?? null,
    roundStatus: data.round_status ?? null,
    hospitals,
    globalLoss: data.global_loss ?? null,
    globalDice: data.global_dice ?? null,
    globalIou: data.global_iou ?? null,
    dpEnabled: data.dp_enabled ?? null,
    privacyUnit: data.privacy_unit ?? null,
    epsilon: data.epsilon ?? null,
    delta: data.delta ?? null,
    clipNorm: data.clip_norm ?? null,
    noiseMultiplier: data.noise_multiplier ?? null,
    cumulativeEpsilon: data.cumulative_epsilon ?? null,
    budgetStatus: data.budget_status ?? null,
    ckksEnabled: data.ckks_enabled ?? null,
    encryptionStatus: data.encryption_status ?? null,
    tlsStatus: data.tls_status ?? null,
    recentEvents: (data.recent_events || []).slice(-MAX_RECENT_EVENTS),
    metricsHistory,
    roundHistory,
  };
}

function applyEvent(prev, event) {
  const { event_type: type, source, round, payload = {} } = event;
  let next = { ...prev };
  
  const modifiesHospitals = [
    EventType.CLIENT_CONNECTED,
    EventType.CLIENT_DISCONNECTED,
    EventType.CLIENT_TRAINING,
    EventType.CLIENT_TRAINING_COMPLETED,
    EventType.CLIENT_FAILED,
  ].includes(type);

  if (modifiesHospitals) {
    next.hospitals = { ...prev.hospitals };
  }

  if (round != null) next.currentRound = round;

  const hospitalFor = (id) => next.hospitals[id] || { hospitalId: id, connectionStatus: "Offline" };

  switch (type) {
    case EventType.SYSTEM_READY:
      next.systemStatus = "Ready";
      if (payload.mode != null) next.mode = payload.mode;
      break;
    case EventType.ROUND_STARTED:
      next.systemStatus = "Training";
      next.roundStatus = "Training";
      if (payload.num_rounds != null) next.numRounds = payload.num_rounds;
      break;
    case EventType.ROUND_COMPLETED:
      next.roundStatus = "Completed";
      next.systemStatus = "Idle";
      next.roundHistory = [
        ...next.roundHistory,
        {
          round,
          durationSeconds: payload.round_duration_seconds ?? null,
          clientsCompleted: payload.clients_completed ?? null,
          clientsFailed: payload.clients_failed ?? null,
        },
      ].slice(-MAX_CHART_HISTORY);
      break;
    case EventType.CLIENT_CONNECTED:
      next.hospitals[source] = { ...hospitalFor(source), connectionStatus: "Online", lastUpdate: event.timestamp };
      break;
    case EventType.CLIENT_DISCONNECTED:
      next.hospitals[source] = { ...hospitalFor(source), connectionStatus: "Offline", lastUpdate: event.timestamp };
      break;
    case EventType.CLIENT_TRAINING:
      next.hospitals[source] = {
        ...hospitalFor(source), connectionStatus: "Training", currentRound: round, lastUpdate: event.timestamp,
      };
      break;
    case EventType.CLIENT_TRAINING_COMPLETED:
      next.hospitals[source] = {
        ...hospitalFor(source),
        connectionStatus: "Completed",
        currentRound: round,
        lastUpdate: event.timestamp,
        numExamples: payload.num_examples ?? hospitalFor(source).numExamples,
        trainLoss: payload.train_loss ?? hospitalFor(source).trainLoss,
        trainDice: payload.train_dice ?? hospitalFor(source).trainDice,
        trainIou: payload.train_iou ?? hospitalFor(source).trainIou,
      };
      break;
    case EventType.CLIENT_FAILED:
      next.hospitals[source] = { ...hospitalFor(source), connectionStatus: "Error", lastUpdate: event.timestamp };
      break;
    case EventType.METRICS_UPDATED:
      next.globalLoss = payload.global_loss ?? next.globalLoss;
      next.globalDice = payload.global_dice ?? next.globalDice;
      next.globalIou = payload.global_iou ?? next.globalIou;
      next.metricsHistory = [
        ...next.metricsHistory,
        { round, globalDice: payload.global_dice ?? null, globalIou: payload.global_iou ?? null, globalLoss: payload.global_loss ?? null },
      ].slice(-MAX_CHART_HISTORY);
      break;
    case EventType.PRIVACY_UPDATED:
      next.dpEnabled = payload.dp_enabled ?? next.dpEnabled;
      next.privacyUnit = payload.privacy_unit ?? next.privacyUnit;
      next.epsilon = payload.epsilon ?? next.epsilon;
      next.delta = payload.delta ?? next.delta;
      next.clipNorm = payload.clip_norm ?? next.clipNorm;
      next.noiseMultiplier = payload.noise_multiplier ?? next.noiseMultiplier;
      next.cumulativeEpsilon = payload.cumulative_epsilon ?? next.cumulativeEpsilon;
      next.budgetStatus = payload.budget_status ?? next.budgetStatus;
      break;
    case EventType.ENCRYPTION_UPDATED:
      next.ckksEnabled = payload.ckks_enabled ?? next.ckksEnabled;
      next.encryptionStatus = payload.encryption_status ?? next.encryptionStatus;
      next.tlsStatus = payload.tls_status ?? next.tlsStatus;
      break;
    default:
      // SYSTEM_WARNING / SYSTEM_ERROR and any other known-but-unhandled type: no state
      // mutation beyond the event log below -- never crash on a well-formed event this
      // hook doesn't specifically react to.
      break;
  }

  next.recentEvents = [...next.recentEvents, event].slice(-MAX_RECENT_EVENTS);
  return next;
}

/**
 * The single WebSocket connection for the whole dashboard. Centralizes connection
 * status + derived state so multiple components can consume it via props (no Redux --
 * one socket's worth of state doesn't need it).
 */
export function useDashboardSocket(url = DEFAULT_WS_URL) {
  const [connectionStatus, setConnectionStatus] = useState("Connecting");
  const [state, setState] = useState(EMPTY_STATE);
  const socketRef = useRef(null);
  const backoffRef = useRef(INITIAL_BACKOFF_MS);
  const reconnectTimerRef = useRef(null);
  const stoppedRef = useRef(false);

  const connect = useCallback(() => {
    if (stoppedRef.current) return;
    setConnectionStatus((prev) => (prev === "Connected" ? prev : "Connecting"));

    let socket;
    try {
      socket = new WebSocket(url);
    } catch (err) {
      scheduleReconnect();
      return;
    }
    socketRef.current = socket;

    socket.onopen = () => {
      backoffRef.current = INITIAL_BACKOFF_MS;
      setConnectionStatus("Connected");
    };

    socket.onmessage = (messageEvent) => {
      let message;
      try {
        message = JSON.parse(messageEvent.data);
      } catch (err) {
        console.warn("FedMed dashboard: received malformed WebSocket message, ignoring.");
        return;
      }
      if (!message || typeof message !== "object" || !message.type) {
        console.warn("FedMed dashboard: received invalid event shape, ignoring.");
        return;
      }
      if (message.type === "snapshot") {
        setState(snapshotToState(message.data || {}));
      } else if (message.type === "event") {
        setState((prev) => applyEvent(prev, message.data));
      } else {
        console.warn(`FedMed dashboard: unrecognised message type "${message.type}", ignoring.`);
      }
    };

    socket.onclose = () => {
      setConnectionStatus("Disconnected");
      scheduleReconnect();
    };

    socket.onerror = (err) => {
      console.warn("FedMed dashboard: WebSocket error — closing socket.", err);
      socket.close();
    };
  }, [url]); // eslint-disable-line react-hooks/exhaustive-deps

  function scheduleReconnect() {
    if (stoppedRef.current) return;
    setConnectionStatus("Reconnecting");
    console.log(`FedMed dashboard: reconnecting in ${backoffRef.current}ms…`);
    reconnectTimerRef.current = setTimeout(() => {
      backoffRef.current = Math.min(backoffRef.current * 2, MAX_BACKOFF_MS);
      connect();
    }, backoffRef.current);
  }

  useEffect(() => {
    stoppedRef.current = false;
    connect();
    return () => {
      stoppedRef.current = true;
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      if (socketRef.current) socketRef.current.close();
    };
  }, [connect]);

  return { connectionStatus, state };
}
