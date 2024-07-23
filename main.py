
# from plot_parser import retrive_movie_plot, get_sub_plots
# from scene_voice_generator import generate
# from extract_video import get_video
from pathlib import Path
from frames import create_screenshots
from frame_ranking import SentenceTransformer, get_image_embeddings, retrieve_frames
import math
import shutil
import librosa
from moviepy.editor import VideoFileClip
from frame_to_clip import get_clip
from merge_clips import join_clips
import itertools
from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer, util, InputExample, losses, models, datasets
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from typing import List
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import CLIPProcessor, CLIPModel

TRAILER_DIR = Path("C:/Users/Avinoam Nukrai/Desktop/ml_trailers_project/trailers")
SCENES_DIR = Path("C:/Users/Avinoam Nukrai/Desktop/ml_trailers_project/NLD_gpt_scenes")  # Replace with the path to your scenes directory



class TextImageExample:
    def __init__(self, texts, image_path, label):
        self.texts = texts
        self.image_path = image_path
        self.label = label

    def get_image(self):
        return Image.open(self.image_path).convert("RGB")

few_shot_examples = [
    TextImageExample(texts=["Neo dodges bullets in slow motion during a rooftop battle."], image_path="C:/Users/Avinoam Nukrai/Desktop/ai_trailer/matrix.jpg", label=0.8),
    TextImageExample(texts=["Jack and Rose stand at the bow of the ship with arms outstretched, feeling the wind."], image_path="C:/Users/Avinoam Nukrai/Desktop/ai_trailer/titanic.jpg", label=0.9),
    TextImageExample(texts=["Batman interrogates the Joker in a dimly lit room at the police station."], image_path="C:/Users/Avinoam Nukrai/Desktop/ai_trailer/batmanJoker.jpg", label=0.8),
    TextImageExample(texts=["Cobb and his team experience a gravity-defying fight in a rotating hallway."], image_path="C:/Users/Avinoam Nukrai/Desktop/ai_trailer/cubb.jpg", label=0.85),
    TextImageExample(texts=["Forrest runs across the country, attracting a crowd of followers."], image_path="C:/Users/Avinoam Nukrai/Desktop/ai_trailer/forrestgamp.jpg", label=0.75),
    TextImageExample(texts=["The T-Rex breaks free from its enclosure during a thunderstorm and attacks the tour vehicles."], image_path="C:/Users/Avinoam Nukrai/Desktop/ai_trailer/trax.jpg", label=0.9),
    TextImageExample(texts=["Andy Dufresne escapes through a sewage pipe during a heavy rainstorm."], image_path="C:/Users/Avinoam Nukrai/Desktop/ai_trailer/andy.jpg", label=0.8),
    TextImageExample(texts=["Vincent and Mia dance the twist contest at Jack Rabbit Slim's."], image_path="C:/Users/Avinoam Nukrai/Desktop/ai_trailer/pulpfic.jpg", label=0.7),
    TextImageExample(texts=["Simba is presented to the animals of the Pride Lands on Pride Rock."], image_path="C:/Users/Avinoam Nukrai/Desktop/ai_trailer/star.jpg", label=0.9),
    TextImageExample(texts=["Luke Skywalker watches the binary sunset on Tatooine."], image_path="C:/Users/Avinoam Nukrai/Desktop/ai_trailer/lionKing.jpg", label=0.85)
]

class FineTunedCLIP(nn.Module):
    def __init__(self, model_name='openai/clip-vit-base-patch32'):
        super(FineTunedCLIP, self).__init__()
        self.clip_processor = CLIPProcessor.from_pretrained(model_name)
        self.clip_model = CLIPModel.from_pretrained(model_name)
        for param in self.clip_model.parameters():
            param.requires_grad = True

    def forward(self, texts, images):
        inputs = self.clip_processor(text=texts, images=images, return_tensors="pt", padding=True)
        outputs = self.clip_model(**inputs)
        return outputs.text_embeds, outputs.image_embeds

class ImageTextDataset:
    def __init__(self, examples: List[TextImageExample]):
        self.examples = examples

    def __getitem__(self, index):
        return self.examples[index]

    def __len__(self):
        return len(self.examples)

def collate_fn(batch):
    texts = [item.texts[0] for item in batch]
    images = [item.get_image() for item in batch]
    labels = torch.tensor([item.label for item in batch], dtype=torch.float)
    return texts, images, labels

# Create a DataLoader
train_dataset = ImageTextDataset(few_shot_examples)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=2, collate_fn=collate_fn)

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, text_embeddings, image_embeddings, labels):
        cosine_sim = self.cos(text_embeddings, image_embeddings)
        loss = nn.MSELoss()(cosine_sim, labels)
        return loss

# Initialize the model, loss function, and optimizer
model = FineTunedCLIP()
loss_fn = CosineSimilarityLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_dataloader:
        texts, images, labels = batch

        # Get embeddings
        text_embeddings, image_embeddings = model(texts, images)

        # Compute loss
        loss = loss_fn(text_embeddings, image_embeddings, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

# # Save the fine-tuned model
# torch.save(model, 'C:/Users/Avinoam Nukrai/Desktop/ml_trailers_project/fine-tuned-model.pth')
#
# print("Fine-tuning complete. Model saved.")



# from scene_voice_generator import generate

import torch


if __name__ == "__main__":
    print("Go!")
    video_folder = Path("C:/Users/Avinoam Nukrai/Desktop/ml_trailers_project")
    video_path = "C:/Users/Avinoam Nukrai/Desktop/ml_trailers_project/NLD.mp4"
    frames_dir = "C:/Users/Avinoam Nukrai/Desktop/ml_trailers_project/frames_dir"
    n_frames = 2000

    create_screenshots(video_path, frames_dir, n_frames)
    print("screenshot DONE")
    # scencs = extract_subs()
    model = SentenceTransformer('clip-ViT-B-32', device='cpu')
    # model = torch.load('C:/Users/Avinoam Nukrai/Desktop/ml_trailers_project/fine-tuned-model.pth')
    # model.eval()
    frames_dir = Path(frames_dir)
    img_filepaths = list(frames_dir.glob("*.jpg"))
    img_emb = get_image_embeddings(model, img_filepaths, 128)
    retrieve_frames(img_filepaths, model, img_emb, 1)

    # print("start generating scenes audio!")
    # generate()
    # print("Audio done!")

    video = VideoFileClip(video_path, audio=True)
    get_clip(video, 20)
    if TRAILER_DIR.exists():
        shutil.rmtree(TRAILER_DIR)
    TRAILER_DIR.mkdir(parents=True, exist_ok=True)
    # Get a sorted list of scene directories
    scene_dirs = sorted(SCENES_DIR.iterdir(), key=lambda p: p.name)
    # Collect audio clips from each scene directory in the sorted order
    audio_clips = [list(scene_dir.glob("clips/*.mp4")) for scene_dir in
                   scene_dirs]
    audio_clips.append(audio_clips[1])
    audio_clips.remove(audio_clips[1])
    print(audio_clips)
    # Flatten the list of lists to get a single list of audio clips in the desired order
    ordered_clips = list(itertools.chain.from_iterable(audio_clips))
    print(ordered_clips)
    # Assuming join_clips is a function that takes a list of audio clips and an output directory
    join_clips([ordered_clips], TRAILER_DIR)