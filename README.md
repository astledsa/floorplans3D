# Floor Plan 2d -> 3d

This is a simple repository for converting 2D floor plans (in PNG/JPG) format to 3D models (in OBJ/GLB) format. The libraries utilized are mentioned in `requirements.txt`.
The flow is simple:
- Utilize one of [roboflow](https://app.roboflow.com) models in order to generate a JSON of the predictions of object segmentation.
- Create a binary wall mask using those predictions
- Utilize the `reconstruct_from_mask_and_predictions` function (defined in `newmain.py`) to reconstruct the 3D model.
