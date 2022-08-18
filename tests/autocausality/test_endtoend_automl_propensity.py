import pytest
import warnings

from auto_causality.datasets import synth_ihdp
from auto_causality.data_utils import preprocess_dataset

warnings.filterwarnings("ignore")  # suppress sklearn deprecation warnings for now..


class TestEndToEndAutoMLPropensity(object):
    """tests autocausality model end-to-end
    1/ import autocausality object
    2/ preprocess data
    3/ init autocausality object
    4/ run autocausality on data
    """

    def test_endtoend(self):
        """tests if model can be instantiated and fit to data"""

        from auto_causality import AutoCausality  # noqa F401
        from auto_causality.shap import shap_values  # noqa F401

        treatment = "treatment"
        targets = ["y_factual"]
        data = synth_ihdp()
        data_df, features_X, features_W = preprocess_dataset(
            data.data,
            data.treatment,
            data.outcomes,
        )

        estimator_list = [
            "NewDummy",
            "SparseLinearDML",
            "ForestDRLearner",
            "TransformedOutcome",
            "CausalForestDML",
            ".LinearDML",
            "DomainAdaptationLearner",
            "SLearner",
            "XLearner",
            "TLearner",
            # "Ortho",
        ]
        outcome = targets[0]
        auto_causality = AutoCausality(
            time_budget=60,
            components_time_budget=10,
            estimator_list=estimator_list,  # "all",  #
            use_ray=False,
            verbose=3,
            components_verbose=2,
            propensity_model="auto",
            resources_per_trial={"cpu": 0.5},
        )

        auto_causality.fit(data_df, treatment, outcome, features_W, features_X)

        # now let's test Shapley values calculation
        for est_name, scores in auto_causality.scores.items():
            # Dummy model doesn't support Shapley values
            # Orthoforest shapley calc is VERY slow
            if "Dummy" not in est_name and "Ortho" not in est_name:

                print("Calculating Shapley values for", est_name)
                shap_values(scores["estimator"], data_df[:10])

        print(f"Best estimator: {auto_causality.best_estimator}")


if __name__ == "__main__":
    pytest.main([__file__])
    # TestEndToEndAutoMLPropensity().test_endtoend()
