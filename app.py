import modal

app = modal.App(name="animate-x-app")

RT = "/animatex"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(
        "requirements-modal.txt"
    )
    .add_local_dir("configs", "configs")
    .add_local_python_source("inference.py", "process_data.py")
)

volume = modal.Volume.from_name("animate-x", create_if_missing=True)

@app.cls(
    image=image,
    gpu="A100-80GB",
    volumes={RT: volume},
    timeout=500 # in seconds
)
class Model:
    @modal.enter()
    def enter():
        import os
        from huggingface_hub import snapshot_download 

        # Download checkpoint if not exists
        ckpt_path = RT + "/checkpoints/" + "v2-1_512-ema-pruned.ckpt"

        if not os.path.exists(ckpt_path):
            print("Downloading checkpoint files.")

            snapshot_download(
                "Shuaishuai0219/Animate-X",
                local_dir=RT + "/checkpoints/"
            )

        # Download the videos
        data_path = RT + "/data/"

        if not os.path.exists(data_path):
            print("Downloading data files.")
            with volume.batch_upload() as batch:
                batch.put_directory("data", data_path)

    @modal.method()
    def run_inference():
        import subprocess

        # Pre-process the video
        command = [
            "python", "process_data.py",
            "--source_video_paths", "animatex/data/videos",
            "--saved_pose_dir", "animatex/data/saved_pkl",
            "--saved_pose", "animatex/data/saved_pose",
            "--saved_frame_dir", "animatex/data/saved_frames"
        ]
        
        result = subprocess.run(command, capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)

        # Run Animate-X
        command_inference = [
            "python", "inference.py",
            "--cfg", "configs/Animate_X_infer.yaml"
        ]
        
        result_inference = subprocess.run(command_inference, capture_output=True, text=True)
        print(result_inference.stdout)
        print(result_inference.stderr)

        return True

@app.local_entrypoint()
def main():
    result = Model.run_inference.remote()
    return