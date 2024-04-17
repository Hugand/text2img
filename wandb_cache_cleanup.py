import wandb, time

while True:
    c = wandb.wandb_sdk.wandb_artifacts.get_artifacts_cache()
    c.cleanup(int(2e9))

    time.sleep(5)