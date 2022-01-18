# Meeting Minutes
## Meeting Information
**Meeting Date/Time:** 18-01-2022, 2:00 pm <br>
**Meeting Purpose:** Discussion <br>
**Meeting Location:** Zoom <br>
**Attendees:** Qiuhong Ke, Shawn Li, Ruochong Shen, Chao Sui


## Discussion Items
Index | Item | Notes | Further Details |
---- | ---- | ---- | ---- |
1 | Reading the survey and setting up DuDoNet | | |
2 | Matlab: internal problem | | |
3 | Next steps | | |


## 1. Reading the survey and setting up DuDoNet
**Content:** 
  - Read a survey: Quantitative Comparison of Commercial and Non-Commercial Metal Artifact Reduction Techniques in Computed Tomography 
    - Experiments on 3 scanners
    - Metrics: average absolute error
    - Not so useful: no description of their MAR methods (Commercial)
  - No access to "Comprehensive Survey on Metal Artifact Reduction Methods in Computed Tomography Images": Solved;
  - DuDoNet: No implementation code from the authors. Will try to implement with help of other sources;
    - InDuDoNet: An Interpretable Dual Domain Network for CT Metal Artifact Reduction (MICCAI2021): Python Code. Setting up the environment.

## 2. Matlab: internal problem
  - MATLAB online reports an internal problem when reading TIFF;
  - Try:
    - Deal with the problem by testing on other .tif files, or searching for other solutions on the Internet;
    - Implement CNN-MAR in Python.

## 3. Next steps
  1. Read: Comprehensive Survey on Metal Artifact Reduction Methods in Computed Tomography Images; search more surveys;
  2. Try to deal with Matlab problem;
  3. Implement in Python:
    - DuDoNet/InDuDoNet;
    - CNN-MAR;
  4. Other methods with experiments.

## Additional Notes
NA
