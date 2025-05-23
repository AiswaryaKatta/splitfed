# 🤖 SplitFed Learning: Privacy-Preserving and Sustainable AI

This project implements **SplitFed Learning** — a hybrid framework that combines **Federated Learning (FL)** and **Split Learning (SL)** — designed for privacy-conscious and scalable model training. Built using PyTorch and Streamlit, the system supports both **SFLV1** (parallel clients with aggregation) and **SFLV2** (sequential training without aggregation).

It also integrates the **CodeCarbon** library to track energy usage and carbon emissions, promoting transparency in machine learning’s environmental impact.

---

## ✅ Features

- 🔐 **Privacy-Preserving Training**  
  - Raw data stays on client devices  
  - Uses **Pixel Differential Privacy** and **Gradient DP**
  
- ⚙️ **Flexible Training Setup**  
  - Choose number of clients, rounds, epochs  
  - Toggle between SFLV1 and SFLV2  
  - Quick Mode for faster runs (uses smaller dataset)
  
- 📊 **Live Visualizations**  
  - Accuracy vs Rounds  
  - Carbon Emissions and Energy Usage  
  - Real-time dashboard built with Streamlit

- 📁 **Exportable Logs**  
  - Download CSV files with round-wise performance and emissions data

---

## 🖼️ Model Architecture

**Client Model**
