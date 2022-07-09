from typing import List

import numpy as np
import pandas as pd

from auto_causality.models.monkey_patches import PropensityScoreWeightingEstimator
from auto_causality.models.wrapper import DoWhyMethods, DoWhyWrapper
from auto_causality.scoring import ate


class OutOfSamplePSWEstimator(PropensityScoreWeightingEstimator):
    """
    A flavor of PSWEstimator that doesn't refit the propensity function
    when doing out-of-sample evaluation
    """

    def __init__(self, *args, recalculate_propensity_score=False, **kwargs):
        # for the case when this is not invoked via Econml wrapper,
        # need to merge init_args in
        init_params = kwargs.pop("init_params", {})
        kwargs = {
            **kwargs,
            **init_params,
            "recalculate_propensity_score": recalculate_propensity_score,
        }
        super().__init__(*args, **kwargs)

        # force fitting for the first time
        self.recalculate_propensity_score = True
        self._estimate_effect()
        self.recalculate_propensity_score = recalculate_propensity_score

    def effect(self, df: pd.DataFrame, **kwargs):

        effect = super().effect(
            df,
            propensity_score_model=self.propensity_score_model,
            **kwargs,
        )

        return effect

    # TODO: delete this once PR #486 is merged in dowhy
    def _estimate_effect(self):
        self._refresh_propensity_score()
        return super()._estimate_effect()


class NewDummy(OutOfSamplePSWEstimator):
    """
    Apply a small random disturbance so the effect values are slightly different
    across
    """

    def effect(self, df: pd.DataFrame, **kwargs):
        effect = super(NewDummy, self).effect(df, **kwargs)
        return effect * (1 + 0.01 * np.random.normal(size=effect.shape))


class DummyModel(DoWhyMethods):
    def __init__(
        self,
        propensity_modifiers: List[str],
        outcome_modifiers: List[str],
        treatment: str,
        outcome: str,
    ):
        self.propensity_modifiers = propensity_modifiers
        self.outcome_modifiers = outcome_modifiers
        self.treatment = treatment
        self.outcome = outcome

    def fit(
        self,
        df: pd.DataFrame,
    ):
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        mean_, _, _ = ate(X[self.treatment], X[self.outcome])
        return np.ones(len(X)) * mean_ * (1 + 0.01 * np.random.normal(size=(len(X),)))


class Dummy(DoWhyWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, inner_class=DummyModel, **kwargs)
