# Deep-Fake-Meme-Generator

This repository explores the creation of AI-generated memes using deep-fake technology. The project combines **audio deep-fake generation** with an attempted **image generation pipeline** to produce meme-style content.

The audio portion works.  
The image portion… did not 😭.

---

## 🔊 Audio Generation (Works)

The audio generation component of this project was created by **Iesu Agapito** and is used here with full credit.  
It enables the generation of synthetic voice audio suitable for meme content and experimentation.

All credit for the **audio deep-fake system** goes to:
- **Iesu Agapito**

---

## 🖼️ Image Generation (Mostly Did Not Work)

I attempted to generate images to pair with the audio using deep-learning image generation techniques.  
While the code and models exist in this repository, the overall image generation pipeline was **not successful** due to training and model limitations.

⚠️ **Exception:**  
The `data_scripts/` folder **does work** and was successfully used for image preprocessing and data preparation.  
However, the actual image generation / modeling stages did **not** produce usable results.

Instead of hiding it — I embraced it.

📂 **Bloopers & Failed Image Generations:**  
👉 **https://drive.google.com/file/d/1-3ozDxd3jaVVX63SQrBT8rN4pF5ChgiX/view?usp=sharing**

---

## 📁 Repository Structure

├── audio_gen/ # Audio generation code (from Iesu Agapito) '=
├── image_gen/ # Image generation attempt (mostly broken) 
├── .gitignore 
├── LICENSE 
└── README.md


---

## 🚀 Usage

### Audio Generation
1. Navigate to the `audio_gen/` directory.
2. Install the required dependencies.
3. Run the audio generation scripts to create synthetic meme audio.

### Image Generation (Experimental)
The `image_gen/` directory contains experimental scripts for image generation.

- `data_scripts/` ✅ works (image preprocessing, dataset handling)
- model training / image generation ❌ not production-ready and largely unsuccessful

See the bloopers link above for examples of the failed outputs.

---

## 🙏 Credits

- **Iesu Agapito** — Audio generation implementation used in this project.
- Open-source ML frameworks and tools that made experimentation possible.
