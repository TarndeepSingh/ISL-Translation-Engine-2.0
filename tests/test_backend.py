import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
import backend.main as m


class Dummy:
    labels = ["hello", "idle"]
    def predict(self, clip): return "hello", 0.95, 0.9


def frame(y, x):
    pose = np.zeros((33, 3)); pose[11] = [0.6, 0.4, 0]; pose[12] = [0.4, 0.4, 0]
    pose[15] = [0.6 + x, y, 0]; pose[16] = [0.4, 0.9, 0]
    return {"pose": pose.flatten().tolist(), "left": None, "right": None}


def run(client):
    events = []
    with client.websocket_connect("/ws") as ws:
        hello = ws.receive_json(); assert hello["type"] == "hello"
        seq = [frame(0.95, 0)] * 15 + [frame(0.3, 0.05 * (-1) ** i) for i in range(30)] + [frame(0.95, 0)] * 20
        for f in seq:
            ws.send_json(f); events.append(ws.receive_json())
    return hello, events


def test_health():
    assert TestClient(m.app).get("/health").json()["ok"] is True


def test_v1_readiness_reports_each_model_separately():
    m.app.state.pipeline = None
    m.app.state.translation_pipeline = None
    client = TestClient(m.app)
    result = client.get("/api/v1/readiness").json()
    assert result["status"] == "not_ready"
    assert result["isolated_recognition"]["reason_code"] == "MODEL_NOT_READY"
    assert result["continuous_translation"]["reason_code"] == "MODEL_NOT_READY"
    assert client.get("/api/v1/health").json()["status"] == "ok"


def test_legacy_live_websocket_is_disabled_until_gate():
    m.predictor = Dummy()
    with pytest.raises(WebSocketDisconnect):
        with TestClient(m.app).websocket_connect("/ws") as ws:
            ws.receive_json()


def test_translation_is_unavailable_without_its_model():
    m.app.state.translation_pipeline = None
    response = TestClient(m.app).post("/api/v1/translate")
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "MODEL_NOT_READY"
