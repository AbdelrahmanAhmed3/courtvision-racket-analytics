import numpy as np
import pytest

from courtvision.analytics.ball_events import (
    FEATURES,
    BallEvent,
    EventModel,
    candidate_features,
    detect_ball_events,
)
from courtvision.detectors.tracknet_adapter import BallPoint
from courtvision.evaluation.court import court_estimate

COURT = court_estimate([[400, 250], [880, 250], [1100, 550], [180, 550]])


def bounce_track() -> list[BallPoint]:
    """A ball moving right that comes down to the floor at frame 10 and back up."""
    return [BallPoint(f, 500 + 10 * f, 400 - 12 * abs(f - 10)) for f in range(21)]


def small_model() -> EventModel:
    """Two classes; only speed_in matters: fast is an impact."""
    n = len(FEATURES)
    coef = np.zeros((2, n))
    coef[0, 0], coef[1, 0] = 1.0, -1.0
    return EventModel(
        classes=("impact", "none"),
        fill=np.full(n, 5.0),
        mean=np.full(n, 5.0),
        scale=np.ones(n),
        coef=coef,
        intercept=np.zeros(2),
    )


def test_the_model_fills_missing_numbers_and_picks_the_highest_score() -> None:
    fast, slow, unknown = np.full((3, len(FEATURES)), 5.0)
    fast[0], slow[0], unknown[0] = 20.0, 1.0, np.nan

    assert small_model().predict(np.array([fast, slow, unknown]))[:2] == [
        "impact",
        "none",
    ]


def test_a_saved_model_loads_back_and_must_match_the_features(tmp_path) -> None:
    path = tmp_path / "model.json"
    small_model().save(path, about="test")

    loaded = EventModel.load(path)
    assert loaded.classes == ("impact", "none")
    assert np.array_equal(loaded.coef, small_model().coef)

    path.write_text(path.read_text().replace('"speed_in"', '"speed"'))
    with pytest.raises(ValueError, match="different features"):
        EventModel.load(path)


def test_the_numpy_model_matches_scikit_learn() -> None:
    sklearn = pytest.importorskip("sklearn")  # the train extra
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, len(FEATURES)))
    classes = (x[:, 0] > 0).astype(int) + (x[:, 1] > 1).astype(int)
    y = np.array(["none", "impact", "bounce"])[classes]
    x[rng.random(x.shape) < 0.1] = np.nan
    pipeline = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression()
    ).fit(x, y)
    imputer, scaler, regression = (step for _, step in pipeline.steps)
    model = EventModel(
        tuple(regression.classes_),
        imputer.statistics_,
        scaler.mean_,
        scaler.scale_,
        regression.coef_,
        regression.intercept_,
    )

    assert sklearn  # imported
    assert model.predict(x) == list(pipeline.predict(x))


def test_a_bounce_turns_the_ball_from_down_to_up() -> None:
    (bounce,) = [
        c for c in candidate_features(bounce_track(), {}, COURT) if c.frame == 10
    ]
    values = dict(zip(FEATURES, bounce.features, strict=True))

    assert values["dy_in"] > 0 > values["dy_out"]  # down the image, then up
    assert values["turn_degrees"] > 90
    assert np.isnan(values["player_distance"])  # no players given
    assert bounce.nearest_track_id is None


def test_the_shipped_model_runs_on_a_track() -> None:
    events = detect_ball_events(bounce_track(), {}, COURT)

    assert all(isinstance(event, BallEvent) for event in events)
    assert {event.kind for event in events} <= {"impact", "bounce", "wall_rebound"}
