import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader, Subset
import unittest
from src.gaf import _filter_gradients_cosine_sim, _compute_gradients

class TestGAF(unittest.TestCase):
    def setUp(self):
        # Simple model
        self.model = nn.Linear(10, 2)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)
        self.device = torch.device('cpu')
        # Simple dataset
        X = torch.randn(20, 10)
        y = torch.randint(0, 2, (20,))
        self.dataset = TensorDataset(X, y)

    def test_filter_gradients_cosine_sim(self):
        # Create dummy gradients
        G1 = [torch.randn(p.shape) for p in self.model.parameters()]
        G2 = [torch.randn(p.shape) for p in self.model.parameters()]

        filtered_grad, cos_dist = _filter_gradients_cosine_sim(G1, G2, cos_distance_thresh=2.0)
        # With a large threshold, we expect some averaging to happen
        self.assertIsNotNone(filtered_grad)
        self.assertTrue(isinstance(cos_dist, float))

        # With a very small threshold, likely no agreement
        filtered_grad_none, _ = _filter_gradients_cosine_sim(G1, G2, cos_distance_thresh=0.0)
        # Most likely none since random gradients are unlikely to match exactly
        self.assertIsNone(filtered_grad_none)

    def test_compute_gradients(self):
        subset_indices = list(range(4))
        b = Subset(self.dataset, subset_indices)
        G, loss, labels, outputs = _compute_gradients(b, self.optimizer, self.model, self.criterion, self.device)
        self.assertTrue(isinstance(G, list))
        self.assertTrue(isinstance(loss, torch.Tensor))
        self.assertTrue(isinstance(labels, torch.Tensor))
        self.assertTrue(isinstance(outputs, torch.Tensor))
        self.assertEqual(len(G), len(list(self.model.parameters())))

if __name__ == '__main__':
    unittest.main()
