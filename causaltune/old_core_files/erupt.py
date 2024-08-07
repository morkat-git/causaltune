from typing import Callable, List, Optional, Union
import copy

import pandas as pd
import numpy as np
import copy
from scipy import stats
import numpy as np
import pandas as pd
from typing import Callable, List, Optional, Union

# implementation of https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3111957
# we assume treatment takes integer values from 0 to n


class DummyPropensity:
    def __init__(self, p: pd.Series, treatment: pd.Series):
        n_vals = max(treatment) + 1
        out = np.zeros((len(treatment), n_vals))
        for i, pp in enumerate(p.values):
            out[i, treatment.values[i]] = pp
        self.p = out

    def fit(self, *args, **kwargs):
        pass

    def predict_proba(self):
        return self.p


class ERUPT:
    def __init__(
        self,
        treatment_name: str,
        propensity_model,
        X_names: Optional[List[str]] = None,
        clip: float = 0.05,
        remove_tiny: bool = True,
    ):
        self.treatment_name = treatment_name
        self.propensity_model = copy.deepcopy(propensity_model)
        self.X_names = X_names
        self.clip = clip
        self.remove_tiny = remove_tiny

    def fit(self, df: pd.DataFrame):
        if self.X_names is None:
            self.X_names = [c for c in df.columns if c != self.treatment_name]
        self.propensity_model.fit(df[self.X_names], df[self.treatment_name])

    def score(
        self, df: pd.DataFrame, outcome: pd.Series, policy: Callable
    ) -> pd.Series:
        w = self.weights(df, policy)
        return (w * outcome).mean()

    def weights(
        self, df: pd.DataFrame, policy: Union[Callable, np.ndarray, pd.Series]
    ) -> pd.Series:
        W = df[self.treatment_name].astype(int)
        assert all(
            [x >= 0 for x in W.unique()]
        ), "Treatment values must be non-negative integers"

        if callable(policy):
            policy = policy(df).astype(int)
        if isinstance(policy, pd.Series):
            policy = policy.values
        policy = np.array(policy)

        d = pd.Series(index=df.index, data=policy)
        assert all(
            [x >= 0 for x in d.unique()]
        ), "Policy values must be non-negative integers"

        if isinstance(self.propensity_model, DummyPropensity):
            p = self.propensity_model.predict_proba()
        else:
            p = self.propensity_model.predict_proba(df[self.X_names])
        p = np.maximum(p, 1e-4)

        weight = np.zeros(len(df))

        for i in W.unique():
            weight[W == i] = 1 / p[:, i][W == i]

        weight[d != W] = 0.0

        if self.remove_tiny:
            weight[weight > 1 / self.clip] = 0.0
        else:
            weight[weight > 1 / self.clip] = 1 / self.clip

        weight *= len(df) / sum(weight)
        assert not np.isnan(weight.sum()), "NaNs in ERUPT weights"

        return pd.Series(index=df.index, data=weight)

    # def probabilistic_erupt_score(
    #     self, df: pd.DataFrame, 
    #     outcome: pd.Series, 
    #     treatment_effects: pd.Series, 
    #     treatment_std_devs: pd.Series,
    #     sd_threshold: float = 1e-2,
    #     iterations: int = 1000
    # ) -> float:
        
    #     if treatment_std_devs[0] < sd_threshold:
    #         return 0

    #     unique_treatments = df[self.treatment_name].unique()
    #     treatment_scores = {treatment: [] for treatment in unique_treatments}

    #     for _ in range(iterations):
    #         sampled_effects = {
    #             treatment: np.random.normal(treatment_effects.loc[treatment], treatment_std_devs.loc[treatment])
    #             for treatment in unique_treatments
    #         }
    #         chosen_treatment = max(sampled_effects, key=sampled_effects.get)
            
    #         weights = self.weights(df, lambda x: np.array([chosen_treatment] * len(x)))
    #         mean_outcome = (weights * outcome).sum() / weights.sum()
    #         treatment_scores[chosen_treatment].append(mean_outcome)

    #     average_outcomes = np.mean([np.mean(scores) for scores in treatment_scores.values() if scores])

    #     return average_outcomes
 
    def probabilistic_erupt_score(
        self, 
        df: pd.DataFrame, 
        outcome: pd.Series, 
        treatment_effects: pd.Series, 
        treatment_std_devs: pd.Series,
        iterations: int = 1000
    ) -> float:
        
        baseline_outcome = outcome[df[self.treatment_name] == 0].mean()
        
        policy_values = []
        treatment_decisions = []

        for _ in range(iterations):
            sampled_effects = pd.Series(
                np.random.normal(treatment_effects, treatment_std_devs),
                index=treatment_effects.index
            )
            
            policy = (sampled_effects > 0).astype(int)
            
            expected_outcome = (
                baseline_outcome + 
                (policy * sampled_effects).mean()
            )
            
            policy_values.append(expected_outcome)
            treatment_decisions.append(policy.mean())
        
        mean_value = np.mean(policy_values)
        se_value = np.std(policy_values) / np.sqrt(iterations)
        
        # Adjusted score: balance between improvement and conservative treatment
        score = (mean_value - 2*se_value)
        
        improvement = (score - baseline_outcome) / baseline_outcome
        
        return improvement
        
    # def probabilistic_erupt_score(
    #     self,
    #     df: pd.DataFrame,
    #     outcome: pd.Series,
    #     treatment_effects: pd.Series,
    #     treatment_std_devs: pd.Series,
    #     iterations: int = 2000  # Increased iterations
    # ) -> float:
    #     baseline_outcome = np.median(outcome[df[self.treatment_name] == 0])
    #     policy_values = []
    #     treatment_decisions = []

    #     for _ in range(iterations):
    #         sampled_effects = pd.Series(
    #             np.random.normal(treatment_effects, treatment_std_devs),
    #             index=treatment_effects.index
    #         )
    #         # More conservative policy
    #         policy = (sampled_effects > 0.5 * treatment_std_devs).astype(int)
    #         expected_outcome = baseline_outcome + np.median(policy * sampled_effects)
    #         policy_values.append(expected_outcome)
    #         treatment_decisions.append(np.mean(policy))

    #     median_value = np.median(policy_values)
    #     mad_value = np.median(np.abs(policy_values - median_value))
    #     avg_treatment_rate = np.mean(treatment_decisions)

    #     # Reintroduce a small treatment penalty
    #     treatment_penalty = 2 * (avg_treatment_rate - 0.5)**2

    #     # Use median absolute deviation for robustness
    #     score = (median_value - 1.5 * mad_value) * (1 - treatment_penalty)

    #     improvement = (score - baseline_outcome) / baseline_outcome
    #     return improvement
    

    # def probabilistic_erupt_score(
    #     self,
    #     df: pd.DataFrame,
    #     outcome: pd.Series,
    #     treatment_effects: pd.Series,
    #     treatment_std_devs: pd.Series,
    #     iterations: int = 1000
    # ) -> float:
    #     baseline_outcome = np.median(outcome[df[self.treatment_name] == 0])
    #     policy_values = []
    #     treatment_decisions = []

    #     for _ in range(iterations):
    #         sampled_effects = pd.Series(
    #             np.random.normal(treatment_effects, treatment_std_devs),
    #             index=treatment_effects.index
    #         )
    #         # Adaptive policy threshold
    #         threshold = np.clip(0.5 * np.median(treatment_std_devs), 1e-3, 1e-1)
    #         policy = (sampled_effects > threshold).astype(int)
            
    #         expected_outcome = baseline_outcome + np.median(policy * sampled_effects)
    #         policy_values.append(expected_outcome)
    #         treatment_decisions.append(np.mean(policy))

    #     median_value = np.median(policy_values)
    #     mad_value = stats.median_abs_deviation(policy_values)
    #     avg_treatment_rate = np.mean(treatment_decisions)

    #     # Adaptive treatment penalty
    #     treatment_penalty = np.clip(4 * (avg_treatment_rate - 0.5)**2, 0, 0.5)

    #     # Calculate relative improvement
    #     relative_improvement = (median_value - baseline_outcome) / np.abs(baseline_outcome)
        
    #     # Incorporate uncertainty and treatment rate into the score
    #     uncertainty_factor = 1 / (1 + mad_value / np.abs(baseline_outcome))
    #     treatment_rate_factor = 4 * avg_treatment_rate * (1 - avg_treatment_rate)  # Peaks at 0.5
        
    #     # Combine factors into final score
    #     score = relative_improvement * uncertainty_factor * treatment_rate_factor * (1 - treatment_penalty)
        
    #     # Apply sigmoid to bound the score and increase sensitivity around 0
    #     final_score = 2 / (1 + np.exp(-10 * score)) - 1
        
    #     return final_score

    # def probabilistic_erupt_score(
    #     self,
    #     df: pd.DataFrame,
    #     outcome: pd.Series,
    #     treatment_effects: pd.Series,
    #     treatment_std_devs: pd.Series,
    #     iterations: int = 2000
    # ) -> float:
    #     baseline_outcome = np.median(outcome[df[self.treatment_name] == 0])
    #     policy_values = []
    #     treatment_decisions = []

    #     for _ in range(iterations):
    #         sampled_effects = pd.Series(
    #             np.random.normal(treatment_effects, treatment_std_devs),
    #             index=treatment_effects.index
    #         )
    #         # Adaptive policy threshold
    #         threshold = np.clip(0.25 * np.median(treatment_std_devs), 1e-4, 1e-2)
    #         policy = (sampled_effects > threshold).astype(int)
            
    #         expected_outcome = baseline_outcome + np.median(policy * sampled_effects)
    #         policy_values.append(expected_outcome)
    #         treatment_decisions.append(np.mean(policy))

    #     median_value = np.median(policy_values)
    #     mad_value = stats.median_abs_deviation(policy_values)
    #     avg_treatment_rate = np.mean(treatment_decisions)

    #     # Adjusted treatment penalty
    #     treatment_penalty = np.clip(2 * (avg_treatment_rate - 0.5)**2, 0, 0.25)

    #     # Use median absolute deviation for robustness
    #     score = (median_value - mad_value) * (1 - treatment_penalty)

    #     # Relative improvement, shifted to ensure positivity
    #     improvement = (score - baseline_outcome) / np.abs(baseline_outcome)
        
    #     # Exponential transformation to ensure positive scores
    #     final_score = np.exp(improvement) - 1

    #     return final_score