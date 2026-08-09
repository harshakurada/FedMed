from __future__ import annotations

from pathlib import Path

import pytest
import torch
from flwr.common import Code, FitIns, ndarrays_to_parameters

from cv_model.brats.config import BraTSRawConfig
from cv_model.brats.discovery import discover_studies
from cv_model.params import get_parameters
from hospital_nodes.client_app import HospitalNodeClient
from hospital_nodes.node import HospitalNode
from hospital_nodes.partition import partition_studies
from hospital_nodes.tests.conftest import tiny_hospital_training_config
from server.federated.client_proxy import InProcessClientProxy


def _make_node(hospital_data_config: BraTSRawConfig, tmp_path: Path, partition_id: int = 0) -> HospitalNode:
    studies = discover_studies(hospital_data_config).valid
    partitions = partition_studies(list(studies), num_partitions=3, seed=hospital_data_config.seed)
    return HospitalNode(
        studies=partitions[partition_id],
        data_config=hospital_data_config,
        training_config=tiny_hospital_training_config(partition_id, tmp_path),
    )


def test_proxy_forwards_directly_to_the_wrapped_client_no_copy(hospital_data_config, tmp_path: Path) -> None:
    node = _make_node(hospital_data_config, tmp_path)
    client = HospitalNodeClient(node).to_client()
    proxy = InProcessClientProxy(cid=node.hospital_id, client=client)

    assert proxy.client is client  # in-process pass-through, no serialization boundary
    assert proxy.cid == node.hospital_id

    ins = FitIns(ndarrays_to_parameters(get_parameters(node.model)), {})
    fit_res = proxy.fit(ins, timeout=None, group_id=None)

    assert fit_res.status.code == Code.OK
    assert fit_res.num_examples == node.num_local_train_studies
    assert "train_dice" in fit_res.metrics


def test_proxy_fit_result_never_carries_raw_data(hospital_data_config, tmp_path: Path) -> None:
    node = _make_node(hospital_data_config, tmp_path)
    client = HospitalNodeClient(node).to_client()
    proxy = InProcessClientProxy(cid=node.hospital_id, client=client)

    ins = FitIns(ndarrays_to_parameters(get_parameters(node.model)), {})
    fit_res = proxy.fit(ins, timeout=None, group_id=None)

    for value in fit_res.metrics.values():
        assert not isinstance(value, torch.Tensor)
        assert not hasattr(value, "modality_paths")


def test_proxy_get_properties_and_reconnect_are_explicitly_unimplemented(hospital_data_config, tmp_path: Path) -> None:
    node = _make_node(hospital_data_config, tmp_path)
    client = HospitalNodeClient(node).to_client()
    proxy = InProcessClientProxy(cid=node.hospital_id, client=client)

    with pytest.raises(NotImplementedError):
        proxy.get_properties(None, timeout=None, group_id=None)  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError):
        proxy.reconnect(None, timeout=None, group_id=None)  # type: ignore[arg-type]
