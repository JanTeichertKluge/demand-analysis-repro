import numpy as np
import pandas as pd

import statsmodels.api as sm
import doubleml as dml

from scipy.linalg import sqrtm
from scipy.stats import norm, chi2

from linearmodels.iv import IV2SLS, IVGMM


def iv_model(
    df,
    dependent,
    exog,
    endog,
    instruments,
    use_gmm=False,
):
    """
    Create and fit an IV model with endogenous regressors and instruments.
    """
    input_dict = {
        "dependent": df[dependent],
        "exog": df[exog],
        "endog": df[endog],
        "instruments": df[instruments],
    }
    if not use_gmm:
        model = IV2SLS(**input_dict)
    else:
        model = IVGMM(**input_dict)
    results = model.fit(cov_type="clustered", clusters=df["ASIN"])
    print(results.summary)
    return results


def lin_model(
    df,
    treatments,
    controls=None,
    level=0.95,
    outcome=["DELTA_SALES_RANK"],
    cov_type="HC0",
    **kwargs,
):
    alpha = 1 - level
    y = df[outcome]
    if controls is None:
        x = df[treatments]
    else:
        x = df[treatments + controls]
    x = sm.add_constant(x)

    model = sm.OLS(y, x)
    results = model.fit(cov_type=cov_type, **kwargs)
    print(f"Adjusted R^2: {results.rsquared_adj.round(4)}")

    # Extract necessary statistics
    coef = results.params
    std_err = results.bse
    z_values = results.tvalues
    p_values = results.pvalues
    ci = results.conf_int(
        alpha=alpha
    )  # Get the confidence intervals using the adjusted alpha

    # Create a DataFrame for the custom summary
    summary_df = pd.DataFrame(
        {
            "coef": coef,
            "std err": std_err,
            "t": z_values,
            "P>|t|": p_values,
            f"[{alpha/2*100:.1f}%": ci[0],  # Lower bound
            f"{(1-alpha/2)*100:.1f}%]": ci[1],  # Upper bound
        }
    )

    # keep as comparison
    # print(results.summary().tables[1])
    print(summary_df.round(3))

    # save coefficient and confidence interval
    lin_model_ci = results.conf_int(alpha=alpha).T
    lin_model_ci.columns = results.params.index
    lin_model_ci = lin_model_ci.T
    lin_model_ci.columns = ["lower", "upper"]
    # add the coefficient
    lin_model_ci["coef"] = results.params

    # select only the treatments
    lin_model_ci = lin_model_ci.loc[treatments]

    return lin_model_ci


def summarize_lin_model(results, level=0.95):
    alpha = 1 - level
    # Extract necessary statistics
    coef = results.params
    std_err = results.bse
    z_values = results.tvalues
    p_values = results.pvalues
    ci = results.conf_int(
        alpha=alpha
    )  # Get the confidence intervals using the adjusted alpha

    # Create a DataFrame for the custom summary
    summary_df = pd.DataFrame(
        {
            "coef": coef,
            "std err": std_err,
            "t": z_values,
            "P>|t|": p_values,
            f"{alpha/2*100:.1f}%": ci[0],  # Lower bound
            f"{(1-alpha/2)*100:.1f}%": ci[1],  # Upper bound
        }
    )

    # keep as comparison
    # print(results.summary().tables[1])
    print(summary_df.round(3))
    return summary_df


def summarize_dml(dml_obj, level=0.95):
    # Extract relevant statistics from the DoubleML object
    coef = dml_obj.coef
    std_err = dml_obj.se
    t_values = dml_obj.t_stat
    p_values = dml_obj.pval

    # Get confidence intervals using the specified level
    conf_int = dml_obj.confint(level=level)
    ci_lower = conf_int.iloc[:, 0]  # Lower bound
    ci_upper = conf_int.iloc[:, 1]  # Upper bound

    # Create a DataFrame for the summary
    summary_df = pd.DataFrame(
        {
            "coef": coef,
            "std err": std_err,
            "t": t_values,
            "P>|t|": p_values,
            f"{(1 - level)/2 * 100:.1f}%": ci_lower,  # Lower bound for confidence interval
            f"{(1 + level)/2 * 100:.1f}%": ci_upper,  # Upper bound for confidence interval
        }
    )

    # Print the custom summary in the format you're looking for
    print(summary_df.round(3))

    return summary_df


def lin_model_cate(
    df,
    basis,
    treatment,
    controls,
    outcome=["DELTA_SALES_RANK"],
    cov_type="HC0",
    **kwargs,
):
    y = df[outcome].squeeze()
    d = df[treatment].squeeze()
    x = df[controls]
    x = sm.add_constant(x)

    outcome_reg = sm.OLS(y, x).fit(
        cov_type=cov_type, **kwargs
    )  # type not really necessary
    treatment_reg = sm.OLS(d, x).fit(
        cov_type=cov_type, **kwargs
    )  # type not really necessary

    y_tilde = y - outcome_reg.predict(x)
    d_tilde = d - treatment_reg.predict(x)

    interacted_basis = basis * d_tilde.values.reshape(-1, 1)
    model = sm.OLS(y_tilde, interacted_basis).fit(cov_type=cov_type, **kwargs)

    return model


def predict_cate(model, basis, treatment_ids, level=0.95, joint=False, n_rep_boot=1000):
    alpha = 1 - level
    if isinstance(model, sm.regression.linear_model.RegressionResultsWrapper):
        coefficients = model.params.iloc[treatment_ids].to_numpy()
        full_cov_matrix = model.cov_params().to_numpy()
        cov_matrix = full_cov_matrix[treatment_ids, :][:, treatment_ids]
    else:
        assert isinstance(model, dml.DoubleMLBLP)
        coefficients = model.blp_model.params.iloc[treatment_ids].to_numpy()
        full_cov_matrix = model.blp_model.cov_params().to_numpy()
        cov_matrix = full_cov_matrix[treatment_ids, :][:, treatment_ids]

    # basis has to be only for the selected treatments
    np_basis = basis.to_numpy()
    effect = np_basis @ coefficients
    blp_se = np.sqrt((np.dot(np_basis, cov_matrix) * np_basis).sum(axis=1))

    if joint:
        # calculate the maximum t-statistic with bootstrap
        normal_samples = np.random.normal(size=[basis.shape[1], n_rep_boot])
        bootstrap_samples = np.multiply(
            np.dot(np_basis, np.dot(sqrtm(cov_matrix), normal_samples)).T,
            (1.0 / blp_se),
        )

        max_t_stat = np.quantile(np.max(np.abs(bootstrap_samples), axis=0), q=level)

        # Lower simultaneous CI
        ci_lower = effect - max_t_stat * blp_se
        # Upper simultaneous CI
        ci_upper = effect + max_t_stat * blp_se

    else:
        # Lower point-wise CI
        ci_lower = effect + norm.ppf(q=alpha / 2) * blp_se
        # Upper point-wise CI
        ci_upper = effect + norm.ppf(q=1 - alpha / 2) * blp_se
    ci = np.vstack((ci_lower, effect, ci_upper)).T
    df_ci = pd.DataFrame(
        ci,
        columns=[
            "{:.1f}%".format(alpha / 2 * 100),
            "effect",
            "{:.1f}%".format((1 - alpha / 2) * 100),
        ],
        index=basis.index,
    )
    return df_ci


def chi2_test(model, treatment_ids):
    if isinstance(model, sm.regression.linear_model.RegressionResultsWrapper):
        coefficients = model.params.iloc[treatment_ids].to_numpy()
        full_cov_matrix = model.cov_params().to_numpy()
        cov_matrix = full_cov_matrix[treatment_ids, :][:, treatment_ids]
    else:
        assert isinstance(model, dml.DoubleMLBLP)
        coefficients = model.blp_model.params.iloc[treatment_ids].to_numpy()
        full_cov_matrix = model.blp_model.cov_params().to_numpy()
        cov_matrix = full_cov_matrix[treatment_ids, :][:, treatment_ids]

    inv_cov_matrix = np.linalg.inv(cov_matrix)
    chi2_stat = np.dot(np.dot(coefficients.T, inv_cov_matrix), coefficients)

    df = len(treatment_ids) - 1
    p_value = 1 - chi2.cdf(chi2_stat, df)

    return p_value
