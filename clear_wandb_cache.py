import os, time

while True:
    os.system("wandb artifact cache cleanup --remove-temp 10GB")
    # os.system("rm -rf ~/.local/share/wandb/artifacts/staging/*")
    os.system("ls ~/.local/share/wandb/artifacts/staging/")
    time.sleep(60 * 60)