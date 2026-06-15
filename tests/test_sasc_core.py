import pandas as pd

from sasc import SASCConfig, fit_sasc_model, predict_sasc_temperature


def test_sasc_predicts_positive_temperature():
    train = pd.DataFrame({
        "entropy_mean": [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9],
        "source_bank_target_temperature": [1.0, 1.02, 1.04, 1.08, 1.1, 1.15, 1.2, 1.25],
    })
    model = fit_sasc_model(train, source_t=1.0, config=SASCConfig())
    pred = predict_sasc_temperature(model, 1.2)
    assert pred["predicted_T"] > 0
    assert pred["support_region"] in {"inside_support", "lower_fallback", "high_extrapolation"}
