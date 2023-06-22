import pytest
import warnings

from econml.inference import BootstrapInference

from causaltune import CausalTune
from causaltune.datasets import linear_multi_dataset
from causaltune.params import SimpleParamService

warnings.filterwarnings("ignore")  # suppress sklearn deprecation warnings for now..


class TestEndToEndInference(object):
    """
    tests confidence interval generation
    """

    def test_endtoend_inference_nobootstrap(self):
        """tests if CATE model can be instantiated and fit to data"""
        data = linear_multi_dataset(1000)
        data.preprocess_dataset()

        cfg = SimpleParamService(
            propensity_model=None,
            outcome_model=None,
            n_jobs=-1,
            include_experimental=False,
            multivalue=False,
        )

        estimator_list = cfg.estimator_names_from_patterns(
            "backdoor", "cheap_inference", len(data.data)
        )

        for e in estimator_list:
            causaltune = CausalTune(
                num_samples=4,
                components_time_budget=10,
                estimator_list=[e],
                use_ray=False,
                verbose=3,
                components_verbose=2,
                resources_per_trial={"cpu": 0.5},
            )

            causaltune.fit(data)
            causaltune.effect_stderr(data.data)

    def test_endtoend_inference_bootstrap(self):
        """tests if CATE model can be instantiated and fit to data"""
        data = linear_multi_dataset(1000)
        data.preprocess_dataset()

        BootstrapInference(n_bootstrap_samples=10, n_jobs=10)
        estimator_list = ["SLearner"]

        for e in estimator_list:
            causaltune = CausalTune(
                num_samples=4,
                components_time_budget=10,
                estimator_list=[e],
                use_ray=False,
                verbose=3,
                components_verbose=2,
                resources_per_trial={"cpu": 0.5},
            )

            causaltune.fit(data)
            causaltune.effect_stderr(data.data)

    def test_endtoend_multivalue_nobootstrap(self):
        data = linear_multi_dataset(1000)
        cfg = SimpleParamService(
            propensity_model=None,
            outcome_model=None,
            n_jobs=-1,
            include_experimental=False,
            multivalue=True,
        )

        estimator_list = cfg.estimator_names_from_patterns(
            "backdoor", "cheap_inference", len(data.data)
        )

        for e in estimator_list:
            causaltune = CausalTune(
                num_samples=4,
                components_time_budget=10,
                estimator_list=[e],
                use_ray=False,
                verbose=3,
                components_verbose=2,
                resources_per_trial={"cpu": 0.5},
            )

            causaltune.fit(data)
            causaltune.effect_stderr(data.data)

        # TODO add an effect() call and an effect_tt call
        print("yay!")

    def test_endtoend_multivalue_bootstrap(self):
        data = linear_multi_dataset(1000)

        estimator_list = ["SLearner"]

        for e in estimator_list:
            causaltune = CausalTune(
                num_samples=4,
                components_time_budget=10,
                estimator_list=[e],
                use_ray=False,
                verbose=3,
                components_verbose=2,
                resources_per_trial={"cpu": 0.5},
            )

            causaltune.fit(data)
            tmp = causaltune.effect_stderr(data.data)  # noqa F841

        # TODO add an effect() call and an effect_tt call
        print("yay!")


if __name__ == "__main__":
    pytest.main([__file__])
    # TestEndToEnd().test_endtoend_iv()
