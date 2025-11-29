# **KinetixVerse: AI-Driven 3D Environment Reconstruction & Simulation System**

This project reconstructs **high-fidelity 3D environments from a single video** using a hybrid pipeline combining advanced computer vision, generative AI, and simulation technologies. The generated environment is exported into **Gazebo** with proper physics, affordances, and SDF world files for robotics training and research.

---

## 🚀 **Overview**

Our system converts any real-world video into a **fully simulated 3D environment**.
It extracts camera poses, reconstructs geometry, clusters meshes, applies semantic understanding, and generates a Gazebo-ready world with realistic physics.

---

## 🧠 **AI Components Used**

* **COLMAP** – Camera pose estimation & sparse/dense reconstruction
* **3D Gaussian Splatting** – Fast photorealistic 3D scene generation
* **DINOv2** – Semantic feature extraction
* **CLIP** – Vision-language embeddings & scene-level understanding
* **SAM / SAM2** – Object & region segmentation
* **GaussianMesh (optional)** – Mesh extraction from Gaussians
* **Affordance Modelling** – Defines how objects can be interacted with
* **Gazebo Physics Engine** – Applies gravity, friction, mass, material behaviour

---

## 🎯 **Key Features**

* Single video → Full 3D world
* Semantic object detection + mesh grouping
* Automatic SDF world creation
* Gazebo environment generation (models, physics, materials)
* Realistic physical interactions using affordances
* Exportable meshes for robotics & simulation platforms

---

## 📥 **Input Requirements**

**A single video**, recorded with:

* Slow, continuous panning
* High frame overlap
* Minimal motion blur
* Stable lighting
* Fixed focus
* Avoiding fast rotations or sudden jumps

This maximizes reconstruction accuracy.

---

## 🛠️ **Pipeline**

1. **Video → Frames**
2. **COLMAP** for poses + depth
3. **Gaussian Splatting** for reconstruction
4. **SAM/SAM2** for segmentation
5. **DINOv2/CLIP** for semantics
6. **Mesh extraction & clustering**
7. **SDF world generation**
8. **Gazebo environment with physics**

---

## 🌌 **Scalability Potential**

The system can create **physics-accurate simulation worlds** for:

* Space rover training (Mars gravity, soil friction, atmosphere)
* Drone navigation
* Disaster-response robot training
* Military & defense simulation
* Smart city and indoor digital twin generation

By adjusting physics parameters, any planet/environment can be simulated.
