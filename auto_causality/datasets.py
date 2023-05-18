import pandas as pd
import numpy as np
from scipy import special

from typing import Union, Callable

from auto_causality.data_utils import CausalityDataset
from auto_causality.utils import generate_psdmat


def linear_multi_dataset(
    n_points=10000, impact=None, include_propensity=False, include_control=False
) -> CausalityDataset:
    if impact is None:
        impact = {0: 0.0, 1: 2.0, 2: 1.0}
    df = pd.DataFrame(
        {
            "X": np.random.normal(size=n_points),
            "W": np.random.normal(size=n_points),
            "T": np.random.choice(np.array(list(impact.keys())), size=n_points),
        }
    )
    df["Y"] = df["X"] + df["T"].apply(lambda x: impact[x])

    propensity_modifiers = []
    if include_propensity:
        skipped_first = False
        for k in impact.keys():
            if (not include_control) and (not skipped_first):
                skipped_first = True
                continue
            df[f"propensity_{k}"] = 1.0 / len(impact)
            propensity_modifiers.append(f"propensity_{k}")

    return CausalityDataset(
        data=df,
        treatment="T",
        outcomes=["Y"],
        common_causes=["W"],
        effect_modifiers=["X"],
        propensity_modifiers=propensity_modifiers,
    )


def nhefs() -> CausalityDataset:
    """loads the NHEFS dataset
    The dataset describes the impact of quitting smoke on weight gain over a period of 11 years
    The data consists of the treatment (quit smoking yes no), the outcome (change in weight) and
    a series of covariates of which we include a subset of 9 (see below).

    If used for academic purposes, pelase consider citing the authors:
    Hernán MA, Robins JM (2020). Causal Inference: What If. Boca Raton: Chapman & Hall/CRC.

    Returns:
        pd.DataFrame: dataset with cols "treatment", "y_factual" and covariates "x1" to "x9"
    """

    df = pd.read_csv(
        "https://cdn1.sph.harvard.edu/wp-content/uploads/sites/1268/1268/20/nhefs.csv"
    )
    covariates = [
        "active",
        "age",
        "education",
        "exercise",
        "race",
        "sex",
        "smokeintensity",
        "smokeyrs",
        "wt71",
    ]

    has_missing = ["wt82"]
    missing = df[has_missing].isnull().any(axis="columns")
    df = df.loc[~missing]

    df = df[covariates + ["qsmk"] + ["wt82_71"]]
    df.rename(
        columns={c: "x" + str(i + 1) for i, c in enumerate(covariates)}, inplace=True
    )

    return CausalityDataset(df, treatment="qsmk", outcomes=["wt82_71"])


def lalonde_nsw() -> CausalityDataset:
    """loads the Lalonde NSW dataset
    The dataset described the impact of a job training programme on the real earnings
    of individuals several years later.
    The data consists of the treatment indicator (training yes no), covariates (age, race,
    academic background, real earnings 1976, real earnings 1977) and the outcome (real earnings in 1978)
    See also https://rdrr.io/cran/qte/man/lalonde.html#heading-0

    If used for academic purposes, please consider citing the authors:
    Lalonde, Robert: "Evaluating the Econometric Evaluations of Training Programs," American Economic Review,
    Vol. 76, pp. 604-620

    Returns:
        pd.DataFrame: dataset with cols "treatment", "y_factual" and covariates "x1" to "x8"
    """

    df_control = pd.read_csv(
        "https://users.nber.org/~rdehejia/data/nswre74_control.txt", sep=" "
    ).dropna(axis=1)
    df_control.columns = (
        ["treatment"] + ["x" + str(x) for x in range(1, 9)] + ["y_factual"]
    )
    df_treatment = pd.read_csv(
        "https://users.nber.org/~rdehejia/data/nswre74_treated.txt", sep=" "
    ).dropna(axis=1)
    df_treatment.columns = (
        ["treatment"] + ["x" + str(x) for x in range(1, 9)] + ["y_factual"]
    )
    df = (
        pd.concat([df_control, df_treatment], axis=0, ignore_index=True)
        .sample(frac=1)
        .reset_index(drop=True)
    )
    return CausalityDataset(df, "treatment", ["y_factual"])


def amazon_reviews(rating="pos") -> CausalityDataset:
    """loads amazon reviews dataset
    The dataset describes the impact of positive (or negative) reviews for products on Amazon on sales.
    The authors distinguish between items with more than three reviews (treated) and less than three
    reviews (untreated). As the rating given by reviews might impact sales, they divide the dataset
    into products with on average positive (more than 3 starts) or negative (less than three stars)
    reviews.
    The dataset consists of 305 covariates (doc2vec features of the review text), a binary treatment
    variable (more than 3 reviews vs less than three reviews) and a continuous outcome (sales).

    If used for academic purposes, please consider citing the authors:
    @inproceedings{rakesh2018linked,
        title={Linked Causal Variational Autoencoder for Inferring Paired Spillover Effects},
        author={Rakesh, Vineeth and Guo, Ruocheng and Moraffah, Raha and Agarwal, Nitin and Liu, Huan},
        booktitle={Proceedings of the 27th ACM International Conference on Information and Knowledge Management},
        pages={1679--1682},
        year={2018},
        organization={ACM}
    }
    Args:
        rating (str, optional): choose between positive ('pos') and negative ('neg') reviews. Defaults to 'pos'.

    Returns:
        pd.DataFrame: dataset with cols "treatment", "y_factual" and covariates "x1" to "x300"
    """
    try:
        assert rating in ["pos", "neg"]
    except AssertionError:
        print(
            "you need to specify which rating dataset you'd like to load. The options are 'pos' or 'neg'"
        )
        return None

    try:
        import gdown
    except ImportError:
        gdown = None

    if rating == "pos":
        url = "https://drive.google.com/file/d/167CYEnYinePTNtKpVpsg0BVkoTwOwQfK/view?usp=sharing"
    elif rating == "neg":
        url = "https://drive.google.com/file/d/1b-MPNqxCyWSJE5uyn5-VJUwC8056HM8u/view?usp=sharing"

    if gdown:
        try:
            df = pd.read_csv("amazon_" + rating + ".csv")
        except FileNotFoundError:
            gdown.download(url, "amazon_" + rating + ".csv", fuzzy=True)
            df = pd.read_csv("amazon_" + rating + ".csv")
        df.drop(df.columns[[2, 3, 4]], axis=1, inplace=True)
        df.columns = ["treatment", "y_factual"] + ["x" + str(i) for i in range(1, 301)]
        return CausalityDataset(df, "treatment", ["y_factual"])
    else:
        print(
            f"""The Amazon dataset is hosted on google drive. As it's quite large,
            the gdown package is required to download the package automatically.
            The package can be installed via 'pip install gdown'. Alternatively, you can
            download it from the following link and store it in the datasets folder:{url}"""
        )
        return None


def synth_ihdp() -> CausalityDataset:
    """loads IHDP dataset
    The Infant Health and Development Program (IHDP) dataset contains data on the impact of visits by specialists
    on the cognitive development of children. The dataset consists of 25 covariates describing various features
    of these children and their mothers, a binary treatment variable (visit/no visit) and a continuous outcome.

    If used for academic purposes, consider citing the authors:
    @article{hill2011,
        title={Bayesian nonparametric modeling for causal inference.},
        author={Hill, Jennifer},
        journal={Journal of Computational and Graphical Statistics},
        volume={20},
        number={1},
        pages={217--240},
        year={2011}
    }

    Returns:
        pd.DataFrame: dataset for causal inference with cols "treatment", "y_factual" and covariates "x1" to "x25"
    """
    # load raw data
    data = pd.read_csv(
        "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv",
        header=None,
    )
    col = [
        "treatment",
        "y_factual",
        "y_cfactual",
        "mu0",
        "mu1",
    ]
    for i in range(1, 26):
        col.append("x" + str(i))
    data.columns = col
    # drop the columns we don't care about
    ignore_patterns = ["y_cfactual", "mu"]
    ignore_cols = [c for c in data.columns if any([s in c for s in ignore_patterns])]
    data = data.drop(columns=ignore_cols)

    return CausalityDataset(data, "treatment", ["y_factual"])


def synth_acic(condition=1) -> CausalityDataset:
    """loads data from ACIC Causal Inference Challenge 2016
    The dataset consists of 58 covariates, a binary treatment and a continuous response.
    There are 10 simulated pairs of treatment and response, which can be selected
    with the condition argument supplied to this function.

    If used for academic purposes, consider citing the authors:
    @article{dorie2019automated,
        title={Automated versus do-it-yourself methods for causal inference: Lessons learned from a
         data analysis competition},
        author={Dorie, Vincent and Hill, Jennifer and Shalit, Uri and Scott, Marc and Cervone, Dan},
        journal={Statistical Science},
        volume={34},
        number={1},
        pages={43--68},
        year={2019},
        publisher={Institute of Mathematical Statistics}
    }

    Args:
        condition (int): in [1,10], corresponds to 10 simulated treatment/response pairs. Defaults to 1.

    Returns:
        pd.DataFrame: dataset for causal inference with columns "treatment", "y_factual" and covariates "x_1" to "x_58"
    """
    try:
        assert condition in range(1, 11)
    except AssertionError:
        print("'condition' needs to be in [1,10]")
        return None

    covariates = pd.read_csv(
        "https://raw.githubusercontent.com/IBM/causallib/"
        + "master/causallib/datasets/data/acic_challenge_2016/x.csv"
    )
    cols = covariates.columns
    covariates.rename(
        columns={c: c.replace("_", "") for c in cols},
        inplace=True,
    )
    url = (
        "https://raw.githubusercontent.com/IBM/causallib/master/causallib/"
        + f"datasets/data/acic_challenge_2016/zymu_{condition}.csv"
    )
    z_y_mu = pd.read_csv(url)
    z_y_mu["y_factual"] = z_y_mu.apply(
        lambda row: row["y1"] if row["z"] else row["y0"], axis=1
    )
    data = pd.concat([z_y_mu["z"], z_y_mu["y_factual"], covariates], axis=1)
    data.rename(columns={"z": "treatment"}, inplace=True)

    return CausalityDataset(data, "treatment", ["y_factual"])


def iv_dgp_econml(
    n: int = 5000, p: int = 10, true_effect: Union[float, int, Callable] = 10
):
    """Generates synthetic IV data for binary treatment and instruments.
    Source: https://github.com/microsoft/EconML/tree/main/notebooks/
    Eg: OrthoIV and DRIV Examples.ipynb

    Args:
        n (int): number of data instances
        p (int): number of observed features
        true_effect (Union[float, int, Callable]): known effect (function or value) to observe.
            Can be a function of covariates or a constant

    Returns:
        CausalityDataset: Data class containing (1) pd.DataFrame with columns "treatment", "y"
                            and covariates from "x_0" to "x_{p}", and labels for treatment,
                            outcome and instrument variables and

    """
    X = np.random.normal(0, 1, size=(n, p))
    Z = np.random.binomial(1, 0.5, size=(n,))
    nu = np.random.uniform(0, 5, size=(n,))
    coef_Z = 0.8
    C = np.random.binomial(
        1, coef_Z * special.expit(0.4 * X[:, 0] + nu)
    )  # Compliers when recomended
    C0 = np.random.binomial(
        1, 0.006 * np.ones(X.shape[0])
    )  # Non-compliers when not recommended
    T = C * Z + C0 * (1 - Z)
    if not isinstance(true_effect, (float, int)):
        true_effect = true_effect(X)
    y = (
        true_effect * T
        + 2 * nu
        + 5 * (X[:, 3] > 0)
        + 0.1 * np.random.uniform(0, 1, size=(n,))
    )
    cov = [f"x{i}" for i in range(1, X.shape[1] + 1)]
    df = pd.DataFrame(X, columns=cov)

    df["y"] = y
    df["treatment"] = T
    df["Z"] = Z

    return CausalityDataset(df, "treatment", ["y"], instruments=["Z"])


def generate_synthetic_data(
    n_samples: int = 100,
    n_covariates: int = 5,
    covariance: Union[str, np.ndarray] = "isotropic",
    confounding: bool = True,
    linear_confounder: bool = False,
    noisy_outcomes: bool = False,
    effect_size: Union[int, None] = None,
    add_instrument: bool = False,
) -> CausalityDataset:
    """generates synthetic dataset with conditional treatment effect (CATE) and optional instrumental variable.
    Supports RCT (unconfounded) and observational (confounded) data.

    Args:
        n_samples (int, optional): number of independent samples. Defaults to 100.
        n_covariates (int, optional): number of covariates. Defaults to 5.
        covariance (Union[str, np.ndarray], optional): covariance matrix of covariates. can be "isotropic",
         "anisotropic" or user-supplied. Defaults to "isotropic".
        confounding (bool, optional): whether or not values of covariates affect treatment effect. Defaults to True.
        noisy_outcomes (bool, optional): additive noise in the outcomes. Defaults to False.
        add_instrument (bool, optional): include instrumental variable (yes/no). Defaults to False
        effect_size (Union[int, None]): if provided, constant effect size (ATE). if None, generate CATE.
            Defaults to None.


    Returns:
        CausalityDataset: columns for covariates, treatment assignment, outcome and true treatment effect
    """

    if covariance == "isotropic":
        sigma = np.random.randn(1)
        covmat = np.eye(n_covariates) * sigma**2
    elif covariance == "anisotropic":
        covmat = generate_psdmat(n_covariates)

    X = np.random.multivariate_normal(
        mean=[0] * n_covariates, cov=covmat, size=n_samples
    )

    if confounding:
        if linear_confounder:
            p = 1 / (1 + np.exp(X[:, 0] * 2 + X[:, 1] * 4))
        else:
            p = 1 / (1 + np.exp(X[:, 0] * X[:, 1] + X[:, 2] * 3))
        p = np.clip(p, 0.1, 0.9)
        C = p > np.random.rand(n_samples)
        print(min(p), max(p))

    else:
        p = 0.5 * np.ones(n_samples)
        C = np.random.binomial(n=1, p=0.5, size=n_samples)

    if add_instrument:
        Z = np.random.binomial(n=1, p=0.5, size=n_samples)
        C0 = np.random.binomial(n=1, p=0.006, size=n_samples)
        T = C * Z + C0 * (1 - Z)
    else:
        T = C

    # fixed effect size:
    if effect_size is not None:
        tau = [effect_size] * n_samples
    else:
        # heterogeneity in effect size:
        weights = np.random.uniform(low=0.4, high=0.7, size=n_covariates)
        e = np.random.randn(n_samples) * 0.01
        tau = X @ weights.T + e + 0.1

    err = np.random.randn(n_samples) * 0.05 if noisy_outcomes else 0

    # nonlinear dependence of Y on X:
    mu = lambda X: X[:, 0] * X[:, 1] + X[:, 2] + X[:, 3] * X[:, 4]  # noqa E731

    Y_base = mu(X) + err
    Y = tau * T + Y_base

    features = [f"X{i+1}" for i in range(n_covariates)]
    df = pd.DataFrame(
        np.array([*X.T, T, Y, tau, p, Y_base]).T,
        columns=features
        + ["treatment", "outcome", "true_effect", "propensity", "base_outcome"],
    )
    data = CausalityDataset(
        data=df,
        treatment="treatment",
        outcomes=["outcome"],
        effect_modifiers=features,
        propensity_modifiers=["propensity"],
    )
    if add_instrument:
        df["instrument"] = Z
        data.instruments = ["instrument"]
    return data


def generate_synth_data_with_categories(
    n_samples=10000,
    n_x=10,
) -> CausalityDataset:
    """Generates synthetic dataset just with categorical features
        Can be used, e.g. in connection with Wise pizza segmentation analysis

    Args:
        n_samples (int, optional): number of independent samples. Defaults to 10000.
        n_x (int, optional): number of features. Defaults to 10.

    Returns:
        CausalityDataset: data object
    """
    T = np.random.binomial(1, 0.5, size=(n_samples,))
    X = np.random.hypergeometric(5, 5, 8, size=(n_samples, n_x))
    epsilon = np.random.uniform(low=-1, high=1, size=(n_samples,))
    gamma = np.random.uniform(low=0.5, high=1.5, size=(n_x,))

    def rho(v):
        return 0.01 * v

    def feature_transform(v):
        return 0.5 * v

    Y = (
        T.T * rho(X[:, : int(n_x / 2)])
        + feature_transform(np.matmul(gamma.T, X.T))
        + epsilon
    )
    features = [f"X{i+1}" for i in range(n_x)]
    df = pd.DataFrame(np.array([*X.T, T, Y]).T, columns=features + ["variant", "Y"])
    cd = CausalityDataset(
        data=df,
        treatment="variant",
        outcomes=["Y"],
    )
    return cd


def generate_non_random_dataset(num_samples=1000):
    num_samples = num_samples

    x1 = np.random.normal(0, 1, num_samples)
    x2 = np.random.normal(0, 1, num_samples)
    x3 = np.random.normal(0, 1, num_samples)
    x4 = np.random.normal(0, 1, num_samples)
    x5 = np.random.normal(0, 1, num_samples)

    propensity = 1 / (
        1 + np.exp(-(0.5 * x1 + 0.8 * x2 - 0.3 * x3 + 0.2 * x4 - 0.1 * x5))
    )
    treatment = np.random.binomial(1, propensity)
    outcome = 2 * treatment + 0.5 * x1 - 0.2 * x2 + np.random.normal(0, 1, num_samples)

    dataset = {
        "T": treatment,
        "Y": outcome,
        "X1": x1,
        "X2": x2,
        "X3": x3,
        "X4": x4,
        "X5": x5,
        "propensity": propensity,
    }

    df = pd.DataFrame(dataset)
    cd = CausalityDataset(
        data=df,
        treatment="T",
        outcomes=["Y"],
        propensity_modifiers=["propensity"],
    )

    return cd
