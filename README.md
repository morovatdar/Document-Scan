# Identification Document Cropping

## Goals
Reads an image of an ID document,  
Identifies the edges of the document,  
Returns an image with the cropped document as the result.


## Steps
Find image edges (image binarization),  
Find an acceptable number of image contours,  
Select the biggest rectangular contour,  
Approximate the corners of the document,  
Update the corners to include all pixels,  
Zero out the contour surrounding,  
Image warping using homography,  
Show/save the result.


## Find Document Edges
1. Convert to grayscale,
2. Perform a Gaussian filter,
3. Determine thresholds by mean intensity,
4. Canny edge detection,
5. Perform morphology close (dilation and erosion),
6. Find contours,
7. Check if the contours are acceptable,
8. If not acceptable update thresholds and repeat from 4,  

![image](https://user-images.githubusercontent.com/83058686/214619086-933502ac-75ad-4c32-8db2-cdc1fe462250.png)
![image](https://user-images.githubusercontent.com/83058686/214619117-21454697-04b7-4e92-abb3-1039b57d3ee4.png)
![image](https://user-images.githubusercontent.com/83058686/214619139-10b9bdef-b27d-426e-964a-edb546968df1.png)  
![image](https://user-images.githubusercontent.com/83058686/214619162-a1893573-10c1-4b65-8e14-bb96bdca506b.png)
![image](https://user-images.githubusercontent.com/83058686/214619185-83c22672-0ad8-451d-81e2-8c37d6f226f0.png)
![image](https://user-images.githubusercontent.com/83058686/214619217-e694bc39-4f2b-4002-ba8c-fd39b734565b.png)


## Find Document Corners
1. Sort contours based on their area,
2. Select the biggest contour,
3. Approximate a polygon curve for the contour,
4. Check the contour to be rectangular and big enough,
5. If not acceptable select the next contour and repeat from 3,
6. Rearrange corners as line endpoints,
7. Update endpoints to include all contour members,
8. Estimate the lines intersection.  

![image](https://user-images.githubusercontent.com/83058686/214620808-9fb3b7e4-348a-4fce-a685-30ddbc809b02.png)
![image](https://user-images.githubusercontent.com/83058686/214620824-86c84847-e5b7-453f-b603-f5314e629cd9.png)
![image](https://user-images.githubusercontent.com/83058686/214620851-ea4ed39b-b294-4cef-84f1-bf82e071e0ee.png)  
![image](https://user-images.githubusercontent.com/83058686/214620877-48ee1263-14ff-415f-a505-305232a01344.png)
![image](https://user-images.githubusercontent.com/83058686/214620908-7c9f5ee6-5822-41fd-8e4d-3015905c8e68.png)
![image](https://user-images.githubusercontent.com/83058686/214620928-20bfd81c-ccfa-409b-99b4-8101e9d3d8ce.png)


## Crop the ID Document
1. Zero out the contour surrounding, if required,
2. Calculate the destination shape and corners (Considering the shape ratio and orientation),
3. Find perspective transform matrix (homography),
4. Warp the image with the perspective transform to the destination corners.

![image](https://user-images.githubusercontent.com/83058686/214621160-21a0bccd-4f83-4c19-afe5-af6bd89588e5.png)
![image](https://user-images.githubusercontent.com/83058686/214621178-bf0302e4-85dc-4093-b05b-a62d6dc8dd8a.png)
![image](https://user-images.githubusercontent.com/83058686/214621206-a34a6725-428d-4be5-8611-780b231ecf92.png)


# Future Improvements 
Improve fill the line capabilities of the algorithm,  
Detect if the document is upside down, or a vertical ID document,  
Use OCR to read the content,  
Use pretrained models to detect card, face, barcode, etc.  
Train a new model (from scratch or on top of a pretrained model).

