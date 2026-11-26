import numpy as np
import pandas as pd
import pytest

from clinmetml.core.feature_selector import mrmr_feature_selection


@pytest.mark.parametrize("columns", [
    ["marker", "noise", "disease"],
    ["noise", "disease", "marker"],
    ["disease", "marker", "noise"],
])
def test_mrmr_target_order(columns):
    rng = np.random.default_rng(42)
    y = np.tile([0, 1], 100)

    df = pd.DataFrame({
        "marker": y,
        "noise": rng.integers(0, 2, len(y)),
        "disease": y,
    })[columns]

    selected = mrmr_feature_selection(
        df, "disease", iteration=1, top_k=1
    ).tolist()

    assert "disease" not in selected
    assert selected == ["marker"]
