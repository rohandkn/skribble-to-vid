import torch
from model import Model

model = Model(device = "cuda", dtype = torch.float16)


video_path = '__assets__/canny_videos_mp4/deer_pic.jpeg'
prompt = "Deer walking in the street"
params = {"t0": 44, "t1": 47 , "motion_field_strength_x" : 12, "motion_field_strength_y" : 12, "video_length": 3}

out_path, fps = f"./text2video_{prompt.replace(' ','_')}.mp4", 4
model.process_text2video_with_draw(video_path=video_path, prompt=prompt, fps = fps, path = out_path, **params)

