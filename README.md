Gender Classification using LBP
===============================
This code implements a Gender Classification algorithm using LBP features in C++ using OpenCV 2.4.9. The code was originally developed as a homework for a Computer Vision course in 2014 so it possesses no optimizations whatsoever and focused solely on working as intended.

Notes
-----
- The code is intended for use with the data available here (www.cec.uchile.cl/~crsilva/projects/gender_classifier/preprocessed.tar.gz). This data contains samples of the Yale Face Database B (10 subjects, 1 pose, 64 illumination conditions).
- The LBP algorithm used is LBP(8,1,u2) and is described in the references.

Instructions
------------
After compiling, the program has two ways of executing:
```bash
./gender_classifier
```
```bash
./gender_classifier (int from 1 to 4)
```
```
./gender_classifier (int from 1 to 4) (0 or 1)
```

- The first one uses a default kernel (LINEAR) and no-hair faces for the SVM. 
- The second one allows the use of other kernels (1 for LINEAR, 2 for POLYNOMIAL, 3 for RADIAL BASIS FUNCTION and 4 for SIGMOIDAL). Parameters for each kernel can be modified in src/main.cpp for tuning up purposes.
- The third one lets you choose of the two face folders which to use (0 for images with no-hair, 1 for images with hair)

References
----------
- Ahonen, T.; Hadid, A.; Pietikainen, M., "Face Description with Local Binary Patterns: Application to Face Recognition," Pattern Analysis and Machine Intelligence, IEEE Transactions on , vol.28, no.12, pp.2037,2041, Dec. 2006

Contact
-------
Cristobal Silva

crsilva at ing dot uchile dot cl