# MobileNetV3 Image Classification Project

## Overview
This project implements a complete Deep Learning pipeline using **MobileNetV3** for efficient image classification. The goal is to optimize for cost and speed, enabling deployment on lightweight infrastructure like AWS Lambda or low-tier EC2 instances.

## Project Structure
- `data/`: Dataset directory (Train/Val structure).
- `src/`: Source code for dataset, model, training, and evaluation.
- `deployment/`: Artifacts for AWS deployment (Lambda function, Dockerfile).
- `main.py`: CLI entry point.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Prepare Data:
   Place your dataset in `data/` with subfolders for each class:
   ```
   data/
       train/
           class_A/
           class_B/
       val/
           class_A/
           class_B/
   ```

## Usage
**Train:**
```bash
python main.py --mode train --data_dir ./data --epochs 10
```

**Evaluate:**
```bash
python main.py --mode evaluate --data_dir ./data --model_path best_model.pth
```
