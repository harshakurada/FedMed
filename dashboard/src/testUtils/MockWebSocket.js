// A minimal, controllable WebSocket stand-in for tests -- the dashboard is exercised
// against this, never a real network socket, in the frontend test suite (the real
// transport is proven by server/tests/test_dashboard_websocket.py against an actual
// `websockets` server).
export class MockWebSocket {
  static instances = [];

  constructor(url) {
    this.url = url;
    this.readyState = 0; // CONNECTING
    this.onopen = null;
    this.onmessage = null;
    this.onclose = null;
    this.onerror = null;
    MockWebSocket.instances.push(this);
  }

  send() {}

  close() {
    this.readyState = 3;
    if (this.onclose) this.onclose({});
  }

  mockOpen() {
    this.readyState = 1;
    if (this.onopen) this.onopen({});
  }

  mockMessage(data) {
    if (this.onmessage) this.onmessage({ data: JSON.stringify(data) });
  }

  mockRawMessage(rawString) {
    if (this.onmessage) this.onmessage({ data: rawString });
  }

  mockClose() {
    this.readyState = 3;
    if (this.onclose) this.onclose({});
  }

  static reset() {
    MockWebSocket.instances = [];
  }

  static latest() {
    return MockWebSocket.instances[MockWebSocket.instances.length - 1];
  }
}
