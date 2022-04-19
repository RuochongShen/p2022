# Meeting Minutes
## Meeting Information
**Meeting Date/Time:** 19-04-2022, 2:00 pm + 05-04-2022, 2:00 pm <br>
**Meeting Purpose:** Discussion <br>
**Meeting Location:** Zoom <br>
**Attendees:** Shawn Li, Ruochong Shen


## Discussion Items
Index | Item | Notes | Further Details |
---- | ---- | ---- | ---- |
1 | Summary | | |
2 | Next steps | | |


## 1. Summary
**Content:** 
  Make a summary of most important part of works done these days.
  - CT learning;
  - MAR;
  - Paper and implementation
    - TIFF load, save, show
    - Evaluation metrics: ssim, rmse, ...
    - CT: parallel/fan beam transformations
    - LI, BHC
    - Data generation, including inserting metal artifacts, cutting down into small size and building training set
    - Trained CNNMAR (most basic part of the model)
    - Calculated metrics on some cases and compare with original paper



## 2. Next steps
 Make further adjustments on CNNMAR, including:

    -  Finish other parts;
    -  Generate multiple raw images from each ground truth image;
    -  Optimize data generation by using algorithms (Multi-otsu / otsu, refer "Edge-enhanced Instance Segmentation of Wrist CT via a Semi-Automatic Annotation Database Construction Method")
    -  Check metrics (some are different from the results in paper);

## Additional Notes
NA
