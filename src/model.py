# src/model.py

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def train_models(
    X_real_train,
    y_real_train,
    X_real_test,
    y_real_test,
    X_syn_raw,
    y_syn,
    X_real_train_scaled,
    X_real_test_scaled,
    X_syn_scaled
):
    """
    Train models on real and synthetic data.
    Returns a dictionary in required format.
    """

    models = {}

    #  Logistic Regression (scaled)
    lr_syn = LogisticRegression(max_iter=1000, random_state=42)
    lr_syn.fit(X_syn_scaled, y_syn)

    lr_real = LogisticRegression(max_iter=1000, random_state=42)
    lr_real = lr_real.fit(X_real_train_scaled, y_real_train)

    models["Logistic Regression"] = {
        "synthetic": (lr_syn, X_real_test_scaled),
        "baseline": (lr_real, X_real_test_scaled)
    }

    #  Decision Tree (raw)
    dt_syn = DecisionTreeClassifier(random_state=42)
    dt_syn.fit(X_syn_raw, y_syn)

    dt_real = DecisionTreeClassifier(random_state=42)
    dt_real.fit(X_real_train, y_real_train)

    models["Decision Tree"] = {
        "synthetic": (dt_syn, X_real_test),
        "baseline": (dt_real, X_real_test)
    }

    #  Random Forest (raw)
    rf_syn = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_syn.fit(X_syn_raw, y_syn)

    rf_real = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_real.fit(X_real_train, y_real_train)

    models["Random Forest"] = {
        "synthetic": (rf_syn, X_real_test),
        "baseline": (rf_real, X_real_test)
    }

    return models