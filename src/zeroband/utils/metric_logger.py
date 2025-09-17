import pickle
from typing import Any, Protocol
import importlib.util
import os


class MetricLogger(Protocol):
    def __init__(self, project, logger_config): ...

    def log(self, metrics: dict[str, Any]): ...

    def finish(self): ...


class WandbMetricLogger(MetricLogger):
    def __init__(self, project, logger_config, resume: bool):
        if importlib.util.find_spec("wandb") is None:
            raise ImportError("wandb is not installed. Please install it to use WandbMonitor.")

        import wandb

        wandb.init(
            project=project, config=logger_config, name=logger_config["config"]["run_name"], resume="auto" if resume else None
        )  # make wandb reuse the same run id if possible

    def log(self, metrics: dict[str, Any]):
        import wandb

        wandb.log(metrics)

    def finish(self):
        import wandb

        wandb.finish()


class SwanlabMetricLogger(MetricLogger):
    def __init__(self, project, logger_config, resume: bool):
        if importlib.util.find_spec("swanlab") is None:
            raise ImportError("swanlab is not installed. Please install it to use SwanlabMetricLogger.")

        import swanlab
        
        # Swanlab does not support resume directly, we'll handle this by checking the run name
        run_name = logger_config["config"]["run_name"]
        swanlab.init(
            project=project,
            config=logger_config["config"],
            name=run_name
        )

    def log(self, metrics: dict[str, Any]):
        import swanlab
        
        # Extract step from metrics if available
        step = metrics.pop("step", None)
        swanlab.log(metrics, step=step)

    def finish(self):
        import swanlab
        
        swanlab.finish()


class TensorboardMetricLogger(MetricLogger):
    def __init__(self, project, logger_config, log_dir: str):
        if importlib.util.find_spec("tensorboard") is None:
            raise ImportError("tensorboard is not installed. Please install it to use TensorboardMetricLogger.")
        
        from torch.utils.tensorboard import SummaryWriter
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Create SummaryWriter
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # Log config parameters
        if "config" in logger_config:
            for key, value in logger_config["config"].items():
                if isinstance(value, (int, float, str, bool)):
                    self.writer.add_text(f"config/{key}", str(value))

    def log(self, metrics: dict[str, Any]):
        # Extract step from metrics if available
        step = metrics.pop("step", None)
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, global_step=step)
            elif isinstance(value, str):
                self.writer.add_text(key, value, global_step=step)

    def finish(self):
        self.writer.close()


class DummyMetricLogger(MetricLogger):
    def __init__(self, project, logger_config, *args, **kwargs):
        self.project = project
        self.logger_config = logger_config
        open(self.project, "a").close()  # Create an empty file to append to

        self.data = []

    def log(self, metrics: dict[str, Any]):
        self.data.append(metrics)

    def finish(self):
        with open(self.project, "wb") as f:
            pickle.dump(self.data, f)
