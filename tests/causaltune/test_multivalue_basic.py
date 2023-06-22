from sklearn.model_selection import train_test_split
from econml.dml import LinearDML
from dowhy import CausalModel

from causaltune.datasets import linear_multi_dataset


class TestMultivalueBasic(object):
    def test_econml(self):
        data = linear_multi_dataset(10000)
        train_data, test_data = train_test_split(data.data, train_size=0.9)
        X_test = test_data[data.effect_modifiers]
        est = LinearDML(discrete_treatment=True)
        est.fit(
            train_data[data.outcomes[0]],
            train_data[data.treatment],
            X=train_data[data.effect_modifiers],
            W=train_data[data.common_causes],
        )
        # Get treatment effect and its confidence interval
        test_data["est_effect"] = est.effect(X_test, T1=1)
        assert abs(test_data["est_effect"].mean() - 2.0) < 0.01

    def test_wrapper(self):
        data = linear_multi_dataset(10000)
        train_data, test_data = train_test_split(data.data, train_size=0.9)
        X_test = test_data[data.effect_modifiers]

        causal_model = CausalModel(
            data=train_data,
            treatment=data.treatment,
            outcome=data.outcomes[0],
            common_causes=data.common_causes,
            effect_modifiers=data.effect_modifiers,
        )
        identified_estimand = causal_model.identify_effect(
            proceed_when_unidentifiable=True
        )

        est_2 = causal_model.estimate_effect(
            identified_estimand,
            method_name="backdoor.econml.dml.LinearDML",
            control_value=0,
            treatment_value=[1, 2],
            target_units="ate",  # condition used for CATE
            confidence_intervals=False,
            method_params={
                "init_params": {"discrete_treatment": True},
                "fit_params": {},
            },
        )

        est_2.estimator.effect(X_test)
