import torch

from uncertainty.config import FusionConfig, LSTMConfig
from uncertainty.kalman import VehicleKalmanFilter
from uncertainty.losses import gaussian_nll_loss
from uncertainty.lstm_model import LSTMUncertaintyEstimator
from uncertainty.physics import compute_sigma_phys
from uncertainty.track_a import TrackAEstimator


def test_physics_expected_ordering_and_bounds():
    close = compute_sigma_phys(10.0, 0.0, 0.5, 0.0, True)
    far = compute_sigma_phys(90.0, 0.0, 0.5, 0.0, True)
    fast = compute_sigma_phys(10.0, 0.0, 20.0, 0.0, True)
    lateral = compute_sigma_phys(0.0, 10.0, 0.5, 0.0, True)
    occluded = compute_sigma_phys(10.0, 0.0, 0.5, 0.0, False)
    assert 0.0 <= close <= 1.0
    assert far > close
    assert fast > close
    assert lateral > close
    assert occluded >= 0.8


def test_kalman_decreases_then_increases_under_occlusion():
    filt = VehicleKalmanFilter()
    first = filt.step(10.0, 0.0, 1.0, 0.0, True)
    visible = first
    for index in range(1, 12):
        visible = filt.step(10.0 + index * 0.05, 0.0, 1.0, 0.0, True)
    assert visible < first
    occluded = visible
    for _ in range(8):
        occluded = filt.step(10.0, 0.0, 1.0, 0.0, False)
    assert occluded > visible
    assert filt.P is not None
    assert torch.allclose(torch.tensor(filt.P), torch.tensor(filt.P.T), atol=1e-8)


def test_track_a_fusion_output():
    estimator = TrackAEstimator(fusion_config=FusionConfig(alpha_phys=0.25, alpha_kf=0.75))
    output = estimator.step("veh1", 10.0, 0.0, 1.0, 0.0, True)
    assert set(output) == {"sigma_phys", "sigma_kf", "sigma_a"}
    expected = 0.25 * output["sigma_phys"] + 0.75 * output["sigma_kf"]
    assert abs(output["sigma_a"] - expected) < 1e-8
    assert all(0.0 <= value <= 1.0 for value in output.values())


def test_lstm_gaussian_and_scalar_shapes_backprop():
    x = torch.randn(32, 10, 10)
    target = torch.rand(32)
    gaussian_model = LSTMUncertaintyEstimator(LSTMConfig(output_mode="gaussian"))
    gaussian_output = gaussian_model(x)
    assert gaussian_output["mu"].shape == (32,)
    assert gaussian_output["logvar"].shape == (32,)
    assert torch.all((0.0 <= gaussian_output["mu"]) & (gaussian_output["mu"] <= 1.0))
    loss = gaussian_nll_loss(gaussian_output["mu"], gaussian_output["logvar"], target)
    loss.backward()

    scalar_model = LSTMUncertaintyEstimator(LSTMConfig(output_mode="scalar"))
    scalar_output = scalar_model(x)
    assert scalar_output["sigma"].shape == (32,)
    assert torch.all((0.0 <= scalar_output["sigma"]) & (scalar_output["sigma"] <= 1.0))
