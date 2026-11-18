import numpy as np

from lighting_ai.model import Network


def test_network_dimensions():
    model = Network()

    x = np.zeros((2, 78))
    y = model(x)

    assert y.shape == (2, 15)


def test_parameters():
    model = Network()

    parameters = model.parameters()

    assert len(parameters) == 3

    for weights, bias in parameters:
        assert isinstance(weights, np.ndarray)
        assert isinstance(bias, np.ndarray)