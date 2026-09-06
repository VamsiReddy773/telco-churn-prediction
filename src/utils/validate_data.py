from typing import Tuple, List
import pandas as pd


def validate_telco_data(df) -> Tuple[bool, List[str]]:
    """Validate raw Telco churn data before training.

    Rewritten with plain pandas checks (no Great Expectations dependency)
    since ge.dataset.PandasDataset is a legacy API removed in modern
    great_expectations versions. Same checks, same return signature.
    """
    print("Starting data validation...")
    failed = []

    required_cols = [
        "customerID", "gender", "Partner", "Dependents", "PhoneService",
        "InternetService", "Contract", "tenure", "MonthlyCharges", "TotalCharges",
    ]
    print("Validating schema and required columns...")
    for col in required_cols:
        if col not in df.columns:
            failed.append(f"expect_column_to_exist:{col}")

    if "customerID" in df.columns and df["customerID"].isnull().any():
        failed.append("expect_column_values_to_not_be_null:customerID")

    print("Validating business logic constraints...")
    value_sets = {
        "gender": ["Male", "Female"],
        "Partner": ["Yes", "No"],
        "Dependents": ["Yes", "No"],
        "PhoneService": ["Yes", "No"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "InternetService": ["DSL", "Fiber optic", "No"],
    }
    for col, allowed in value_sets.items():
        if col in df.columns and not df[col].isin(allowed).all():
            failed.append(f"expect_column_values_to_be_in_set:{col}")

    print("Validating numeric ranges and business constraints...")
    if "tenure" in df.columns:
        if not df["tenure"].between(0, 120).all():
            failed.append("expect_column_values_to_be_between:tenure")
        if df["tenure"].isnull().any():
            failed.append("expect_column_values_to_not_be_null:tenure")

    if "MonthlyCharges" in df.columns:
        if not df["MonthlyCharges"].between(0, 200).all():
            failed.append("expect_column_values_to_be_between:MonthlyCharges")
        if df["MonthlyCharges"].isnull().any():
            failed.append("expect_column_values_to_not_be_null:MonthlyCharges")

    print("Validating data consistency...")
    if "TotalCharges" in df.columns and "MonthlyCharges" in df.columns:
        total_numeric = pd.to_numeric(df["TotalCharges"], errors="coerce")
        monthly_numeric = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
        comparable = total_numeric.notna() & monthly_numeric.notna()
        if comparable.any():
            ok_ratio = (total_numeric[comparable] >= monthly_numeric[comparable]).mean()
            if ok_ratio < 0.95:
                failed.append("expect_column_pair_values_A_to_be_greater_than_B:TotalCharges_vs_MonthlyCharges")

    total_checks = len(required_cols) + 1 + len(value_sets) + 5 + 1
    passed_checks = total_checks - len(failed)
    is_valid = len(failed) == 0

    if is_valid:
        print(f"Data validation PASSED: {passed_checks}/{total_checks} checks successful")
    else:
        print(f"Data validation FAILED: {len(failed)}/{total_checks} checks failed")
        print(f"Failed expectations: {failed}")

    return is_valid, failed