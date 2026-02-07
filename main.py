import yaml
from src.utils.logger import setup_logger
from src.pipelines.batch_pipeline import run_batch_pipeline
from src.pipelines.real_time_pipeline import run_realtime_pipeline

if __name__ == "__main__":
    # Load configuration from YAML file
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    setup_logger("config/logging.yaml")
    # Determine which pipeline to run based on configuration
    # if cfg["project"]["mode"] == "batch":
    #     run_batch_pipeline(cfg)
    # elif cfg["project"]["mode"] == "real_time":
    #     run_realtime_pipeline(cfg)
    # else:
    #     raise ValueError(f"Unknown pipeline type: {cfg["project"]["mode"]}")