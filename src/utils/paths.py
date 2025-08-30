import os


def project_root() -> str:
    """
    Returns absolute path to the project root.
    Assumes this file lives at src/utils/paths.py
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def artifacts_root(rq: str = "rq2") -> str:
    return os.path.join(project_root(), "artifacts", rq)


def checkpoints_dir(model_type: str, rq: str = "rq2") -> str:
    """
    model_type: "unified" or "experts"
    """
    return os.path.join(artifacts_root(rq), "checkpoints", model_type)


def results_dir(rq: str = "rq2") -> str:
    return os.path.join(artifacts_root(rq), "results")


def logs_dir(rq: str = "rq2") -> str:
    return os.path.join(artifacts_root(rq), "logs")


def checkpoints_oof_dir(model_type: str, rq: str = "rq2") -> str:
    """
    Out-of-fold checkpoints base directory: artifacts/rq2/checkpoints_oof/{model_type}
    model_type: "unified" or "experts"
    """
    return os.path.join(artifacts_root(rq), "checkpoints_oof", model_type)


def dataset_root_unified() -> str:
    return os.path.join(project_root(), "dataset")


def dataset_root_tissues() -> str:
    return os.path.join(project_root(), "dataset_tissues")


def processed_output_root() -> str:
    return os.path.join(project_root(), "output")


def ensure_dirs_exist() -> None:
    for d in [
        artifacts_root(),
        checkpoints_dir("unified"),
        checkpoints_dir("experts"),
        checkpoints_oof_dir("unified"),
        checkpoints_oof_dir("experts"),
        results_dir(),
        logs_dir(),
        dataset_root_unified(),
        dataset_root_tissues(),
    ]:
        os.makedirs(d, exist_ok=True)


