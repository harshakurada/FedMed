from __future__ import annotations

from server.federated.grpc_service import fedmed_pb2, fedmed_pb2_grpc


def test_health_check_request_round_trips_through_serialization() -> None:
    request = fedmed_pb2.HealthCheckRequest(hospital_id="hospital_a")
    restored = fedmed_pb2.HealthCheckRequest()
    restored.ParseFromString(request.SerializeToString())
    assert restored.hospital_id == "hospital_a"


def test_health_check_response_round_trips_through_serialization() -> None:
    response = fedmed_pb2.HealthCheckResponse(healthy=True, authenticated_hospital_id="hospital_b", message="ok")
    restored = fedmed_pb2.HealthCheckResponse()
    restored.ParseFromString(response.SerializeToString())
    assert restored.healthy is True
    assert restored.authenticated_hospital_id == "hospital_b"
    assert restored.message == "ok"


def test_generated_grpc_stub_and_servicer_classes_exist() -> None:
    assert hasattr(fedmed_pb2_grpc, "FedMedCoordinationStub")
    assert hasattr(fedmed_pb2_grpc, "FedMedCoordinationServicer")
    assert hasattr(fedmed_pb2_grpc, "add_FedMedCoordinationServicer_to_server")
