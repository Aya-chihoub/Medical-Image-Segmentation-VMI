# Medical Image Segmentation — VMI Lab

A laboratory assignment exploring and comparing classical image segmentation techniques and deep learning on three types of medical images: **brain MRI**, **cardiac MRI**, and **retinal fundus images**.

Completed as part of the **M2 Vision et Machine Intelligente (VMI)** programme at **Université Paris Cité**.

## Lab objectives

The aim is to segment different anatomical structures and evaluate the methods using visual comparisons and the **Dice similarity coefficient**:

- **Brain MRI:** segment the corpus callosum
- **Cardiac MRI:** segment the left-ventricular myocardium
- **DRIVE retinal images:** segment blood vessels

## Methods

| Image type | Methods explored |
| --- | --- |
| Brain MRI | Global manual thresholding, thresholding with a region of interest (ROI), active contours, 2D U-Net |
| Cardiac MRI | Global thresholding, thresholding with ROI, active contours for inner/outer boundaries, 2D U-Net |
| Retinal images (DRIVE) | Green-channel thresholding with contrast enhancement, Frangi vesselness filtering, 2D U-Net |

The U-Net models use **PyTorch** and **MONAI**. The notebook includes image preprocessing, mask construction, model training, visualization, and Dice evaluation.

## Lab results

The following Dice scores are reported in the notebook.

### Brain MRI — corpus callosum

| Method | Dice score |
| --- | ---: |
| Global thresholding | 0.0981 |
| Thresholding + ROI | **0.8463** |
| Active contours | 0.2888 |
| U-Net | 0.0881 |

A manually selected ROI considerably improves thresholding on the example image by excluding unrelated bright structures. The U-Net was trained for 20 epochs on only three images and evaluated on one held-out image.

### Cardiac MRI — myocardium

| Method | Dice score |
| --- | ---: |
| Global thresholding | 0.0230 |
| Thresholding + ROI | 0.0518 |
| Active contours | **0.6961** |
| U-Net | 0.3577 |

The manual annotations are provided as landmarks for the inner and outer ventricular contours, which are converted into a myocardium mask. Active contours performed well on the selected example. The U-Net was trained for 12 epochs on 108 annotated frames and evaluated on 28 later temporal frames.

### Retinal images — DRIVE

| Method | Dice score |
| --- | ---: |
| Manual thresholding | **0.4527** |
| Frangi vesselness filter | 0.2904 |
| U-Net | 0.2672 |

Preprocessing uses the green channel of the fundus image, where vessels have useful contrast. The U-Net was trained for 10 epochs on 16 DRIVE training images and evaluated on four held-out images from the same training partition.

> **Interpretation:** These results are exploratory, not a benchmark ranking. For the cardiac and DRIVE experiments, classical methods were evaluated on a selected image while the U-Net scores are averages over multiple images. The cardiac frames also come from the same MRI volume, so they do not measure generalization to new patients.

## Repository structure

```text
Medical-Image-Segmentation-VMI/
├── TP_VMI_Aya_Chihoub.ipynb   # Experiments, visualizations and observations
└── README.md
```

The image datasets are **not included** in this repository.

## Running the notebook

1. Open [TP_VMI_Aya_Chihoub.ipynb](TP_VMI_Aya_Chihoub.ipynb) in **Google Colab**.
2. Prepare a directory called `dataTPseg` in your Google Drive.
3. Place the course-provided archives `brainMRI.zip`, `MRIheart.zip`, and `DRIVE.zip` inside it.
4. Run the notebook cells in order. The notebook mounts Google Drive and installs `monai` for the deep-learning experiments.
5. If your data is stored elsewhere, update `Working_directory` near the beginning of the notebook:

   ```python
   Working_directory = "/content/drive/MyDrive/dataTPseg/"
   ```

A GPU-enabled Colab runtime is recommended for U-Net training. The notebook uses CUDA when available.

### Main dependencies

- Python, NumPy, Matplotlib, SciPy
- scikit-image, Pillow
- PyTorch, MONAI
- Google Colab (for the original Drive-based workflow)

## Observations and takeaways

- Segmentation quality depends strongly on the anatomical structure and the method's assumptions.
- A carefully selected ROI can make simple intensity thresholding more effective.
- Active contours can be helpful when the desired structure has identifiable boundaries.
- A small U-Net trained for only a few epochs may struggle, especially with limited data and small target structures.
- A more reliable comparison would evaluate every method on the **same held-out cases** and test generalization across patients.

## Author

**Aya Chihoub**  
M2 Vision et Machine Intelligente — Université Paris Cité  
[GitHub profile](https://github.com/Aya-chihoub)
