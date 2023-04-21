import torch
from model import Model

model = Model(device = "cuda", dtype = torch.float16)


prompt = 'oil painting of a deer, a high-quality, detailed, and professional photo'
video_path = '__assets__/canny_videos_mp4/deer_pic.jpeg'
out_path = f'./text2video_edge_guidance_{prompt}.mp4'
model.process_controlnet_canny(video_path, prompt=prompt, save_path=out_path)
