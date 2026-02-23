# PersonSearch-AssignmentML4CV
End-to-end Person Search system combining pedestrian detection and re-identification on real-world surveillance images, built with PyTorch on the PRW dataset.

# Person Search: Joint Pedestrian Detection and Re-Identification

> Academic project for the Machine Learning for Computer Vision (ML4CV) course
> A.Y. 2025/26 — University of Bologna (CVLAB)

## Overview

This repository implements a **Person Search** system that jointly solves pedestrian detection and person re-identification in a single pipeline. Given a query image of a target individual, the model localizes and matches that person across a gallery of raw, uncropped scene images, a more challenging and realistic setting compared to standard Re-ID, which assumes perfectly cropped inputs.

## Task

Unlike standard Person Re-Identification, Person Search operates on full scene images, requiring the model to:
1. **Detect** all pedestrians in a gallery of surveillance images
2. **Match** the detected individuals against a query image

## Dataset

The project uses the **PRW (Person Re-Identification in the Wild)** dataset:

| Split | # Frames | # IDs | # Pedestrians |
|-------|----------|-------|---------------|
| Train | 5,134    | 482   | 16,243        |
| Val   | 570      | 482   | 1,805         |
| Test  | 6,112    | 450   | 25,062        |
| **Total** | **11,816** | **932** | **43,110** |

## Results

| Metric | Value |
|--------|-------|
| mAP    | TBD   |
| top-1  | TBD   |

## References

- Zheng, L. et al. *Person Re-identification in the Wild*. CVPR 2017. [arXiv:1604.02531](https://arxiv.org/abs/1604.02531)
- [ML4CV Assignment 2025-26 — CVLAB UniBO](https://github.com/CVLAB-Unibo/ML4CV-assignment-2025-26)

## License

This project is for academic purposes only. 
It was developed as part of the Machine Learning for Computer Vision course at the University of Bologna (A.Y. 2025/26).



pip install -r requirements.txt

