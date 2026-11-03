"""AST checks. Each check takes a parsed module and returns findings without a source attached."""
from __future__ import annotations

import ast
import builtins
import re
from collections.abc import Iterator

from .findings import Finding

SCALERS = {
    "StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler", "Normalizer",
    "QuantileTransformer", "PowerTransformer", "KBinsDiscretizer", "SimpleImputer", "KNNImputer",
    "IterativeImputer", "OneHotEncoder", "OrdinalEncoder", "PCA", "TruncatedSVD", "KernelPCA",
    "TfidfVectorizer", "CountVectorizer", "HashingVectorizer", "TfidfTransformer",
}
# These use the labels, so fitting them before the split leaks far more.
LABEL_AWARE = {
    "SelectKBest", "SelectPercentile", "SelectFpr", "SelectFdr", "SelectFwe", "RFE", "RFECV",
    "SelectFromModel", "SequentialFeatureSelector", "TargetEncoder", "LinearDiscriminantAnalysis",
}
RESAMPLERS = {
    "SMOTE", "ADASYN", "BorderlineSMOTE", "SVMSMOTE", "KMeansSMOTE", "SMOTENC", "SMOTEN",
    "RandomOverSampler", "RandomUnderSampler", "SMOTETomek", "SMOTEENN", "NearMiss",
    "TomekLinks", "ClusterCentroids",
}
PREPROCESSORS = SCALERS | LABEL_AWARE | RESAMPLERS
FIT_METHODS = {"fit", "fit_transform", "fit_resample", "partial_fit"}
METRICS = {
    "accuracy_score", "balanced_accuracy_score", "f1_score", "precision_score", "recall_score",
    "roc_auc_score", "average_precision_score", "log_loss", "classification_report",
    "confusion_matrix", "r2_score", "mean_squared_error", "mean_absolute_error",
    "root_mean_squared_error", "mean_absolute_percentage_error", "matthews_corrcoef",
    "cohen_kappa_score", "jaccard_score",
}
PREDICT_METHODS = {"predict", "predict_proba", "decision_function"}
SELECTION_KWARGS = {"validation_data", "eval_set", "valid_sets", "eval_dataset", "valid_dataset",
                    "val_dataloaders"}
SAVE_CALLS = {"save", "state_dict", "deepcopy", "save_model", "save_weights", "save_pretrained",
              "dump", "save_checkpoint"}
SHUFFLE_SPLITTERS = {"ShuffleSplit", "StratifiedShuffleSplit", "GroupShuffleSplit"}
KFOLD_SPLITTERS = {"KFold", "StratifiedKFold", "RepeatedKFold", "RepeatedStratifiedKFold",
                   "StratifiedGroupKFold"}
SEED_CALLS = {"manual_seed", "manual_seed_all", "seed_everything", "set_seed", "set_random_seed"}
EARLY_STOPPING = re.compile(
    r"EarlyStopping|ModelCheckpoint|early_stopping|load_best_model_at_end|ReduceLROnPlateau|"
    r"best_iteration|save_best_only")
EVAL_FUNC_NAME = re.compile(r"eval|valid|test|predict|infer", re.I)


# --------------------------------------------------------------------------- helpers

def _tokens(name: str) -> list[str]:
    return [t.lower() for t in re.split(r"_|(?<=[a-z0-9])(?=[A-Z])", name) if t]


def is_test_name(name: str) -> bool:
    """True for names like X_test, test_loader, testset, yTest; False for latest, contest."""
    return any(t.startswith(("test", "tst")) or t in {"xtest", "ytest"} for t in _tokens(name))


def is_train_name(name: str) -> bool:
    return any(t.startswith(("train", "trn")) or t in {"xtrain", "ytrain"} for t in _tokens(name))


def func_name(call: ast.Call) -> str:
    f = call.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return ""


def dotted(expr: ast.AST) -> str | None:
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        base = dotted(expr.value)
        return f"{base}.{expr.attr}" if base else None
    return None


def names(node: ast.AST | None) -> set[str]:
    if node is None:
        return set()
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def arg_names(call: ast.Call) -> set[str]:
    return set().union(*(names(a) for a in call.args))


_BUILTINS = set(dir(builtins))


def imported_names(tree: ast.Module) -> set[str]:
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            out.update((a.asname or a.name).split(".")[0] for a in node.names)
    return out


def pos(node: ast.AST) -> tuple[int, int]:
    return (getattr(node, "lineno", 0), getattr(node, "col_offset", 0))


def is_method_call(node: ast.AST, method: str, nargs: int | None = None) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == method and (nargs is None or len(node.args) == nargs))


_SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def walk_scope(scope: ast.AST) -> Iterator[ast.AST]:
    """Like ast.walk, but does not enter nested functions or classes."""
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, _SCOPE_NODES):
            stack.extend(ast.iter_child_nodes(node))


def scopes(tree: ast.Module) -> Iterator[ast.AST]:
    yield tree
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def calls_in(scope: ast.AST) -> list[ast.Call]:
    return sorted((n for n in walk_scope(scope) if isinstance(n, ast.Call)), key=pos)


def _assign_targets(node: ast.AST) -> list[ast.AST]:
    if isinstance(node, ast.Assign):
        return node.targets
    if isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        return [node.target]
    return []


def preprocessor_vars(tree: ast.Module) -> dict[str, str]:
    """Map variables such as `scaler` or `self.scaler` to the preprocessor class they hold."""
    found = {}
    for node in ast.walk(tree):
        value = getattr(node, "value", None)
        if not isinstance(value, ast.Call) or func_name(value) not in PREPROCESSORS:
            continue
        for target in _assign_targets(node):
            key = dotted(target)
            if key:
                found[key] = func_name(value)
    return found


def assigned_names(tree: ast.Module) -> dict[int, set[str]]:
    """For every call on the right of an assignment, the names on the left."""
    out: dict[int, set[str]] = {}
    for node in ast.walk(tree):
        targets = _assign_targets(node)
        if not targets or getattr(node, "value", None) is None:
            continue
        left = set().union(*(names(t) for t in targets))
        for sub in ast.walk(node.value):
            if isinstance(sub, ast.Call):
                out.setdefault(id(sub), set()).update(left)
    return out


def preprocessor_of(call: ast.Call, known: dict[str, str]) -> str | None:
    """Class name if `call` is a fit method on a preprocessor, e.g. StandardScaler().fit(...)."""
    if func_name(call) not in FIT_METHODS or not isinstance(call.func, ast.Attribute):
        return None
    receiver = call.func.value
    if isinstance(receiver, ast.Call):
        name = func_name(receiver)
        return name if name in PREPROCESSORS else None
    key = dotted(receiver)
    return known.get(key) if key else None


def _first(items) -> str:
    return sorted(items)[0]


# --------------------------------------------------------------------------- PX001 / PX006

def check_preprocess_before_split(tree: ast.Module) -> list[Finding]:
    known = preprocessor_vars(tree)
    assigned = assigned_names(tree)
    not_data = imported_names(tree) | _BUILTINS
    out, reported = [], set()
    for scope in scopes(tree):
        calls = calls_in(scope)
        splits = [c for c in calls if func_name(c) == "train_test_split"]
        if not splits:
            continue
        assigns = sorted((n for n in walk_scope(scope)
                          if _assign_targets(n) and getattr(n, "value", None) is not None), key=pos)
        for split in splits:
            # Every variable the split's inputs were computed from, following assignments back.
            split_inputs = arg_names(split) - not_data
            lineage = set(split_inputs)
            for node in reversed([a for a in assigns if pos(a) < pos(split)]):
                if set().union(*(names(t) for t in _assign_targets(node))) & lineage:
                    lineage |= names(node.value) - not_data
            for call in calls:
                if pos(call) >= pos(split) or id(call) in reported:
                    continue
                cls = preprocessor_of(call, known)
                data = (arg_names(call) | assigned.get(id(call), set())) - not_data
                if cls is None or not data & lineage:
                    continue
                reported.add(id(call))
                shown = data & split_inputs or data & lineage
                out.append(Finding(
                    "PX001", "medium" if cls in SCALERS else "high", call.lineno,
                    f"{cls} is fitted on `{_first(shown)}` before train_test_split (line "
                    f"{split.lineno}), so it has already seen the rows that end up in the "
                    "test set."))
    return out


def check_fit_on_test(tree: ast.Module) -> list[Finding]:
    known = preprocessor_vars(tree)
    out = []
    for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
        if func_name(call) not in FIT_METHODS or not isinstance(call.func, ast.Attribute):
            continue
        test = sorted(n for n in arg_names(call) if is_test_name(n))
        if not test:
            continue
        cls = preprocessor_of(call, known)
        receiver = dotted(call.func.value) or func_name(call.func.value) + "()"
        if cls:
            message = (f"{cls} is fitted on `{test[0]}`, so the test data is scaled with its own "
                       "statistics instead of the ones learned from training data.")
        else:
            message = (f"`{receiver}.{call.func.attr}` is called on `{test[0]}`, which trains "
                       "the model on the test set.")
        out.append(Finding("PX006", "high", call.lineno, message))
    return out


# --------------------------------------------------------------------------- PX002

def check_test_used_for_selection(tree: ast.Module, text: str) -> list[Finding]:
    selects = bool(EARLY_STOPPING.search(text))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                test = sorted(n for n in names(kw.value) if is_test_name(n))
                if kw.arg in SELECTION_KWARGS and test:
                    out.append(Finding(
                        "PX002", "high" if selects else "medium", kw.value.lineno,
                        f"`{kw.arg}` is given `{test[0]}`. The test set then guides training "
                        "(early stopping, checkpoints, tuning), so its score is no longer an "
                        "unbiased estimate. If this is really a validation set, rename it."))
            if is_method_call(node, "step") and node.args:
                test = sorted(n for n in names(node.args[0]) if is_test_name(n))
                if test:
                    out.append(Finding(
                        "PX002", "high", node.lineno,
                        f"The learning-rate scheduler is stepped with `{test[0]}`, so a "
                        "test-set metric is steering training."))
        elif isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
            compared = names(node.test)
            test = sorted(n for n in compared if is_test_name(n))
            if not test or not any("best" in n.lower() for n in compared):
                continue
            saves = any(isinstance(n, ast.Call) and func_name(n) in SAVE_CALLS
                        for stmt in node.body for n in ast.walk(stmt))
            out.append(Finding(
                "PX002", "high" if saves else "medium", node.lineno,
                f"The best model/epoch is picked by comparing `{test[0]}`. Reporting that same "
                "test score afterwards overstates how well the model generalises."))
    return out


# --------------------------------------------------------------------------- PX003

def check_eval_on_train_data(tree: ast.Module) -> list[Finding]:
    out = []
    for scope in scopes(tree):
        fitted: dict[str, tuple[str, int]] = {}
        for call in calls_in(scope):
            name = func_name(call)
            receiver = dotted(call.func.value) if isinstance(call.func, ast.Attribute) else None
            if name == "fit" and receiver and call.args:
                fitted[receiver] = (ast.dump(call.args[0]), call.lineno)
                continue
            scored: list[ast.Call] = []
            if name == "score" and receiver:
                scored = [call]
            elif name in METRICS:
                scored = [s for s in ast.walk(call) if isinstance(s, ast.Call)
                          and func_name(s) in PREDICT_METHODS and isinstance(s.func, ast.Attribute)]
            for s in scored:
                model = dotted(s.func.value)
                if model not in fitted or not s.args:
                    continue
                data_dump, fit_line = fitted[model]
                if ast.dump(s.args[0]) != data_dump or any(is_train_name(n) for n in names(s.args[0])):
                    continue
                out.append(Finding(
                    "PX003", "high", call.lineno,
                    f"`{model}` is scored on `{ast.unparse(s.args[0])}`, the same data it was "
                    f"fitted on at line {fit_line}. This measures memorisation, not how it "
                    "does on new data."))
                break
    return out


# --------------------------------------------------------------------------- PX004

def _kw(call: ast.Call, name: str) -> ast.AST | None:
    return next((k.value for k in call.keywords if k.arg == name), None)


def check_unseeded_splits(tree: ast.Module) -> list[Finding]:
    out = []
    for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
        name = func_name(call)
        if any(k.arg is None for k in call.keywords) or _kw(call, "random_state") is not None:
            continue
        shuffle = _kw(call, "shuffle")
        shuffled = isinstance(shuffle, ast.Constant) and shuffle.value is True
        if name == "train_test_split" and not (isinstance(shuffle, ast.Constant) and shuffle.value is False):
            what = "train_test_split has no random_state, so every run gets a different split"
        elif name in SHUFFLE_SPLITTERS or (name in KFOLD_SPLITTERS and shuffled):
            what = f"{name} shuffles without a random_state, so the folds change on every run"
        else:
            continue
        out.append(Finding("PX004", "low", call.lineno,
                           f"{what} and the reported numbers cannot be reproduced exactly."))
    return out


def torch_seed_facts(tree: ast.Module) -> tuple[int | None, bool]:
    """(line of the first .backward() call, whether any seeding call exists) for one file."""
    backward = [n.lineno for n in ast.walk(tree) if is_method_call(n, "backward")]
    seeded = any(isinstance(n, ast.Call) and func_name(n) in SEED_CALLS for n in ast.walk(tree))
    return (min(backward) if backward else None), seeded


# --------------------------------------------------------------------------- PX005 / PX007

def _has_no_grad(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    guards = {"no_grad", "inference_mode"}
    for dec in func.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        if (dotted(target) or "").split(".")[-1] in guards:
            return True
    for node in walk_scope(func):
        if isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                expr = item.context_expr
                if isinstance(expr, ast.Call) and func_name(expr) in guards:
                    return True
    return False


def check_eval_without_eval_mode(tree: ast.Module) -> list[Finding]:
    # A method is fine if any method of its class switches to eval mode (often __init__ or load).
    owner = {id(f): cls for cls in ast.walk(tree) if isinstance(cls, ast.ClassDef)
             for f in cls.body if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))}
    out = []
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not EVAL_FUNC_NAME.search(func.name) or not _has_no_grad(func):
            continue
        if any(is_method_call(n, "eval", 0) for n in ast.walk(owner.get(id(func), func))):
            continue
        out.append(Finding(
            "PX005", "medium", func.lineno,
            f"`{func.name}()` runs under torch.no_grad() but never calls model.eval(), so "
            "dropout stays active and batch-norm keeps updating its statistics during "
            "evaluation. Ignore this if every caller already calls model.eval()."))
    return out


def check_train_mode_not_restored(tree: ast.Module) -> list[Finding]:
    functions = {n.name: n for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}

    def reach(loop: ast.AST) -> list[ast.AST]:
        """Nodes in the loop plus nodes of the module's functions it calls (one level)."""
        nodes = list(ast.walk(loop))
        for n in list(nodes):
            if isinstance(n, ast.Call):
                callee = functions.get(func_name(n))
                if callee is not None and callee is not loop:
                    nodes.extend(ast.walk(callee))
        return nodes

    out, flagged = [], []
    loops = sorted((n for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While))), key=pos)
    for loop in loops:
        if any(lo <= loop.lineno <= hi for lo, hi in flagged):
            continue
        nodes = reach(loop)
        backward = [n for n in nodes if is_method_call(n, "backward")]
        evals = [n for n in nodes if is_method_call(n, "eval", 0)]
        if backward and evals and not any(is_method_call(n, "train") for n in nodes):
            flagged.append((loop.lineno, loop.end_lineno or loop.lineno))
            out.append(Finding(
                "PX007", "medium", loop.lineno,
                f"This training loop switches the model to eval mode (line "
                f"{min(e.lineno for e in evals)}) but never calls model.train(), so after the "
                "first evaluation training continues with dropout and batch-norm turned off."))
    return out


def run_file_checks(tree: ast.Module, text: str) -> list[Finding]:
    return [
        *check_preprocess_before_split(tree),
        *check_fit_on_test(tree),
        *check_test_used_for_selection(tree, text),
        *check_eval_on_train_data(tree),
        *check_unseeded_splits(tree),
        *check_eval_without_eval_mode(tree),
        *check_train_mode_not_restored(tree),
    ]
