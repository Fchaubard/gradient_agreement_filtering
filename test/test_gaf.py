import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader, Subset
import pytest
from gradient_agreement_filtering.gaf import _filter_gradients_cosine_sim, _compute_gradients

@pytest.fixture
def setup():
    # Simple model
    model = nn.Linear(10, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    device = torch.device('cpu')
    # Simple dataset
    X = torch.randn(20, 10)
    y = torch.randint(0, 2, (20,))
    dataset = TensorDataset(X, y)
    return model, criterion, optimizer, device, dataset

def test_filter_gradients_cosine_sim(setup):
    model, _, _, _, _ = setup
    # Create dummy gradients
    G1 = [torch.randn(p.shape) for p in model.parameters()]
    G2 = [torch.randn(p.shape) for p in model.parameters()]

    filtered_grad, cos_dist = _filter_gradients_cosine_sim(G1, G2, cos_distance_thresh=2.0)
    # With a large threshold, we expect some averaging to happen
    assert filtered_grad is not None
    assert isinstance(cos_dist, float)

    # With a very small threshold, likely no agreement
    filtered_grad_none, _ = _filter_gradients_cosine_sim(G1, G2, cos_distance_thresh=0.0)
    # Most likely none since random gradients are unlikely to match exactly
    assert filtered_grad_none is None

def test_compute_gradients(setup):
    model, criterion, optimizer, device, dataset = setup
    subset_indices = list(range(4))
    b = Subset(dataset, subset_indices)
    G, loss, labels, outputs = _compute_gradients(b, optimizer, model, criterion, device)
    assert isinstance(G, list)
    assert isinstance(loss, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert isinstance(outputs, torch.Tensor)
    assert len(G) == len(list(model.parameters()))
