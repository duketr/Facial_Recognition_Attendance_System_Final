# Facial_Recognition_Attendance_System


## Note
- This version is non DB connection. It only has timekeeping window and information window.
- This version run with Intel Realsense Camera Device (pyrealsense2). Please change code if you run with another devices.
## Dependencies
-  see 'env.yml' file
## Usage
- Modify path in './app/test_non_cam.py'
- Modify path of model and database in './config/main.cfg'
- Change model in 'model' folder
- Add more image of employees to './face_db_local'
- To Run app, command: python './app/test_non_cam.py'