import ast
import textwrap

import pytest

from pariksha.checks import is_test_name, run_file_checks


def rules(code: str) -> list[tuple[str, int]]:
    code = textwrap.dedent(code)
    return sorted((f.rule, f.line) for f in run_file_checks(ast.parse(code), code))


@pytest.mark.parametrize("name, expected", [
    ("X_test", True), ("test_loader", True), ("testset", True), ("yTest", True), ("Xtest", True),
    ("latest", False), ("contest", False), ("X_train", False), ("best_acc", False),
])
def test_test_names(name, expected):
    assert is_test_name(name) is expected


# PX001 ----------------------------------------------------------------------------------------

def test_scaler_before_split():
    assert rules("""
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    """) == [("PX001", 3)]


def test_leak_followed_through_later_assignments():
    found = rules("""
        X = StandardScaler().fit_transform(X)
        X_sel = SelectKBest(k=5).fit_transform(X, y)
        X_res, y_res = SMOTE().fit_resample(X_sel, y)
        parts = train_test_split(X_res, y_res, random_state=0)
    """)
    assert found == [("PX001", 2), ("PX001", 3), ("PX001", 4)]


def test_severity_depends_on_preprocessor():
    code = textwrap.dedent("""
        X = StandardScaler().fit_transform(X)
        X, y = SMOTE().fit_resample(X, y)
        parts = train_test_split(X, y, random_state=0)
    """)
    severities = [f.severity for f in run_file_checks(ast.parse(code), code)]
    assert severities == ["medium", "high"]


def test_fit_after_split_is_fine():
    assert rules("""
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        scaler = StandardScaler().fit(X_train)
        X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
    """) == []


def test_unrelated_fit_before_split_is_fine():
    assert rules("""
        import numpy as np
        vocab = CountVectorizer().fit(other_corpus)
        X_train, X_test = train_test_split(np.asarray(X), random_state=0)
    """) == []


def test_preprocessor_on_self_attribute():
    assert rules("""
        class Prep:
            def __init__(self):
                self.scaler = MinMaxScaler()

            def run(self, X, y):
                X = self.scaler.fit_transform(X)
                return train_test_split(X, y, random_state=1)
    """) == [("PX001", 7)]


# PX006 ----------------------------------------------------------------------------------------

def test_refit_on_test():
    assert rules("X_test = StandardScaler().fit_transform(X_test)") == [("PX006", 1)]


def test_model_trained_on_test():
    assert rules("model.fit(X_test, y_test)") == [("PX006", 1)]


def test_transform_on_test_is_fine():
    assert rules("X_test = scaler.transform(X_test)") == []


# PX002 ----------------------------------------------------------------------------------------

def test_keras_validation_on_test_with_early_stopping():
    code = textwrap.dedent("""
        model.fit(X_train, y_train, validation_data=(X_test, y_test),
                  callbacks=[EarlyStopping(patience=3)])
    """)
    (finding,) = run_file_checks(ast.parse(code), code)
    assert (finding.rule, finding.severity) == ("PX002", "high")


def test_xgboost_eval_set_on_test_without_callbacks_is_medium():
    code = "clf.fit(X_train, y_train, eval_set=[(X_test, y_test)])"
    (finding,) = run_file_checks(ast.parse(code), code)
    assert (finding.rule, finding.severity) == ("PX002", "medium")


def test_validation_set_is_fine():
    assert rules("model.fit(X_tr, y_tr, validation_data=(X_val, y_val))") == []


def test_scheduler_and_checkpoint_on_test():
    assert rules("""
        for epoch in range(10):
            test_loss = run(test_loader)
            sched.step(test_loss)
            if test_loss < best_loss:
                torch.save(model.state_dict(), "best.pt")
    """) == [("PX002", 4), ("PX002", 5)]


# PX003 ----------------------------------------------------------------------------------------

def test_scored_on_fit_data():
    assert rules("""
        model.fit(X, y)
        print(accuracy_score(y, model.predict(X)))
        print(model.score(X, y))
    """) == [("PX003", 3), ("PX003", 4)]


def test_training_score_named_train_is_fine():
    assert rules("""
        model.fit(X_train, y_train)
        print(accuracy_score(y_train, model.predict(X_train)))
    """) == []


def test_clustering_predict_is_fine():
    assert rules("""
        km.fit(X)
        labels = km.predict(X)
    """) == []


# PX004 ----------------------------------------------------------------------------------------

def test_unseeded_splitters():
    assert rules("""
        a = train_test_split(X, y)
        b = KFold(n_splits=5, shuffle=True)
        c = StratifiedShuffleSplit(n_splits=3)
    """) == [("PX004", 2), ("PX004", 3), ("PX004", 4)]


def test_seeded_or_unshuffled_splitters_are_fine():
    assert rules("""
        a = train_test_split(X, y, random_state=0)
        b = train_test_split(X, y, shuffle=False)
        c = KFold(n_splits=5)
        d = train_test_split(X, y, **params)
    """) == []


# PX005 / PX007 --------------------------------------------------------------------------------

def test_eval_function_without_eval_mode():
    assert rules("""
        def evaluate(model, loader):
            with torch.no_grad():
                return [model(x) for x in loader]
    """) == [("PX005", 2)]


def test_decorated_eval_function_with_eval_mode_is_fine():
    assert rules("""
        @torch.no_grad()
        def evaluate(model, loader):
            model.eval()
            return [model(x) for x in loader]
    """) == []


def test_eval_mode_set_elsewhere_in_class_is_fine():
    assert rules("""
        class Predictor:
            def __init__(self, model):
                self.model = model
                self.model.eval()

            @torch.inference_mode()
            def predict(self, x):
                return self.model(x)
    """) == []


def test_train_mode_not_restored():
    assert rules("""
        model.train()
        for epoch in range(5):
            for x, y in loader:
                loss_fn(model(x), y).backward()
            model.eval()
    """) == [("PX007", 3)]


def test_train_mode_restored_in_called_function():
    assert rules("""
        def train_one_epoch(model, loader):
            model.train()
            for x, y in loader:
                loss_fn(model(x), y).backward()

        def check(model):
            model.eval()

        for epoch in range(5):
            train_one_epoch(model, loader)
            check(model)
    """) == []
