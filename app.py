import modal
from process_video import process_video_app, Process
from run_inference import run_inference_app, Inference

app = modal.App("animate-x")
app.include(process_video_app)
app.include(run_inference_app)

@app.local_entrypoint()
def main():
    run_upload = False
    run_preprocess = False
    run_inference = True

    # TODO: modify this for new structure
    if run_upload:
        # if not os.path.isdir("./data/images/"):
        #     raise AssertionError("images_path is not a directory")
            
        print("Uploading data files.")
        # with data_volume.batch_upload() as batch:
        #     batch.put_directory("./data/", data)
        #     batch.put_directory("/data/videos/", data + "/videos")

    # 2. Preprocess video files
    if run_preprocess:
        print("Preprocessing videos")
        
        Process().process_main.remote(
            video_paths=["/data/data/videos/dance_1.mp4", "/data/data/videos/dance_2.mp4"],
            saved_pose_dir="/data/data/saved_pkl/",
            saved_pose="/data/data/saved_pose/",
            saved_frame_dir="/data/data/saved_frames/"
        )

    ## 3. Run Animate-X Inference
    if run_inference:
        print("Running Animate-X Inference.")

        # 0 : frame_interval
        # 1 : ref_image_key
        # 2 : pose_seq_key
        # 3 : original_driven_video_seq_key
        # 4 : pose_embedding_key
        # 5 : seet

        test_list_path = [
            [2, "/data/data/images/1.jpg", "/data/data/saved_pose/dance_1", "/data/data/saved_frames/dance_1", "/data/data/saved_pkl/dance_1.pkl", 14],
            [2, "/data/data/images/2.png", "/data/data/saved_pose/dance_1", "/data/data/saved_frames/dance_2", "/data/data/saved_pkl/dance_2.pkl", 16],
        ]

        Inference().run_inference.remote(
            cfg_update={
                "max_frames": 32,
                "test_model": "/checkpoints/animate-x_ckpt.pth",
                "test_list_path": test_list_path,
                "partial_keys": [
                    ['image','local_image', 'dwpose','pose_embeddings'], 
                ],
                "batch_size": 1,
                "latent_random_ref": True,
                "scale": 8,
                "video_compositions": ['image', 'local_image', 'dwpose', 'randomref', 'randomref_pose', 'pose_embedding'],
                "resolution": [512, 768],
                "round": 4,
                "ddim_timesteps": 30,
                "seed": 13,
                "log_dir": 'results', # save dir
            }
        )