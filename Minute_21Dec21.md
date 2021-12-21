# Meeting Minutes
## Meeting Information
**Meeting Date/Time:** 21-12-2021, 2:00 pm <br>
**Meeting Purpose:** Discussion <br>
**Meeting Location:** Zoom <br>
**Attendees:** Qiuhong Ke, Shawn Li, Ruochong Shen


## Discussion Items
Index | Item | Notes | Further Details |
---- | ---- | ---- | ---- |
1 | Implement metrics | | |
2 | Implement LI and BHC | | |
3 | Next steps | | |


## 1. Implementation
**Content:** 
  - Load, save and show TIFF in Python and MATLAB
  - Metrics (Python): 
    - mean squared error (mse)
    - normalized root mse (rmse)
    - peak signal noise ratio (psnr)
    - normalized mutual information (nmi)
    - structural similarity (ssim)
    - inception score (iscore)
  - Basic algorithms (MATLAB):
    - linear interpolation
    - beam-harden correction
  - Calculate metrics to evaluate the results of the MAR algorithms (Python)

## 2. Next steps
  1. Do more experiments on MAR and calculate metrics. Evaluate the metrics by comparing with subjective evaluation and try to find the shortcomings of the metrics;
  2. Check the difference between MAR results and ground truth (minus: the difference between the images);
  3. MAR on metal scans (about 533 images);
  4. Read the surveys and papers.

## Additional Notes
NA
