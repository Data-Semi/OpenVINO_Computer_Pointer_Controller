# Computer Pointer Controller
This is the 2nd project in Udacity Intel® Edge AI for IoT Developers Nanodegree Program.  
This project uses a gaze detection model to control the mouse pointer of my computer. To achieve this, I have used multiple models in the same machine and coordinate the flow of data between those models.
The gaze estimation model requires three inputs: The head pose, the left eye image, the right eye image.  

To get these inputs, I used below OpenVino models:  
- face-detection-0200    
- head-pose-estimation-adas-0001   
- landmarks-regression-retail-0009  
- gaze-estimation-adas-0002  

The following diagram illustrates the workflow:   
![workflow-diagram](images/pipeline.png)  

## Project Directories
```
├── README.md
├── bin
│   ├── demo.mp4
│   └── demo_result.mp4
├── metrics_stats_result
│   └── 20211106_081627stats.txt
├── requirements.txt
└── src
    ├── app.log
    ├── face_detection.py
    ├── facial_landmarks_detection.py
    ├── gaze_estimation.py
    ├── head_pose_estimation.py
    ├── input_feeder.py
    ├── main.py
    ├── mouse_controller.py
    └── visualize.py
```
## Project Set Up and Installation
This project will use InferenceEngine API from Intel's OpenVino ToolKit to build the project.  
Environment has been used is as below:  
```
OS: Ubuntu18.04 run on WSL2 of Windows 10
Openvino version: openvino_2021.4.689
Python version: Python 3.6.9 (Default installed in Ubuntu18.04)
```
To install Openvino, please refer to:    
https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_linux.html  
1. Clone (or download) this repository  

```
git clone this repository
```

2. Create a virtual environment  

```
python -m venv venv
source venv/bin/activate
```

3. Install application dependencies  

```
pip3 install -r requirements.txt
```

4. Download the required models using the model downloader  
 
```
python /opt/intel/openvino_2021/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name face-detection-0200
python /opt/intel/openvino_2021/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name head-pose-estimation-adas-0001
python /opt/intel/openvino_2021/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name landmarks-regression-retail-0009
python /opt/intel/openvino_2021/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name gaze-estimation-adas-0002

```

## Demo
Navigate to the `src/` directory and execute the `main.py` script with the required arguments.  
See the [documentation](#documentation) for more info.  
```
python main.py <arguments>
```
I have set the arguments default value inside the code, so that we can run with no argument for first try.  
```
python main.py 
```

## Documentation

| Arguments  | Description                                                                                                     | Required |
|------------|-----------------------------------------------------------------------------------------------------------------|----------|
| -h, --help | show this help message and exit                                                                                 | False    |
| -cpu_ext   | MKLDNN (CPU) targeted custom layers. Absolute path to a shared library with the kernels impl.                   | False    |
| -m_fd      | *Path to a IR model for face detection detection                                                                     | False     |
| -m_hpe     | *Path to IR model for head pose estimation estimation                                                               | False     |
| -m_ld      | *Path to IR model for facial landmark detection detection                                                          | False     |
| -m_ge      | *Path to IR model for gaze estimation estimation                                                                    | False     |
| -i         | *Path to image or video file, or 'cam' for web cam live feed                                              | False     |
| -d         | Target device to infer on: CPU, GPU, FPGA or MYRIAD.(CPU by default)                              | False    |
| -pt        | Probability treshold for face detection set a number between 0-1                                                               | False    |
| -v         | Visualization flag - set to no display by default, to set display frames, specify 't' or 'yes' or 'true' or '1' | False    |
| -flags         | Visualization flag - set to display different model outputs of each frame. Please see examples below.  | False    |
*Do not add the model extension (e.g., .xml or .bin)  

- Visualization detail flags examples: -flag fd hpe ge fld (Seperate each flag by space), fd for Face Detection Model, hpe for Head Pose Estimation Model, fld for Facial Landmark Detection Model, ge for Gaze Estimation Model."   

## Benchmarks
I used different model precisions on a machine having Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz and 16 GB RAM.  
The program write the results in a text file in 'folder metrics_stats_result' with every execution.  
With all FP32 precision settings, the result was as below:  
```
Information of model name, model_precision, inference time(s), frame per second, load time(s)
../intel/face-detection-0200/FP32/face-detection-0200
FP32
0.004693289934578589
213
0.10627484321594238

../intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009
FP32
0.0007434335805601992
1345
0.026098012924194336

../intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001
FP32
0.001200776989177122
833
0.03454089164733887

../intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002
FP32
0.0012780407727774926
782
0.041086435317993164  
```
With all FP16 precision settings, the result was as below:  

```
Information of model name, model_precision, inference time(s), frame per second, load time(s)

../intel/face-detection-0200/FP16/face-detection-0200
FP16
0.00482627496881
207
0.1073923110961914

../intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009
FP16
0.0007310923883470438
1368
0.025674104690551758

../intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001
FP16
0.001198606976008011
834
0.040769338607788086

../intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002
FP16
0.001255003072447696
797
0.0440669059753418
```

## Results  
The results show that model with lower precision (FP16) is having smaller inference time than higher precision(FP32).  
This is because lower precision models use less memory and they are less expensive computationally.     
Lower precision models loose some performance, I didn't observe big difference in this project.  

## Stand Out Suggestions
- Display zoom in face image frame together with the original frame in the visualization part.
- Save the result video in the folder ./bin
- Save the metrics to a txt as a record for every excecute.  
- Save logs into src/app.log for every important steps in the program 

### Edge Cases
- If there is no face detected the application does not infer anything and logs an event of not detecting any face.  
- If multiple face has been detected in a frame, the application considers the first detected face for inference.  

## Thouble shootings
### 1. DISPLAY setting error
The message was,
```
xhost:  unable to open display ":0.0"
```
Solution found:  
Set DISPLAY correctly for WSL2 system, need to set display address accordingly instead of :0
The command is as below:  

```
echo "export DISPLAY=\$(grep nameserver /etc/resolv.conf | sed 's/nameserver //'):0" >> ~/.bashrc
source ~/.bashrc
source ../venv/bin/activate
```
Check if the environment variable has been set correctly  
```
echo $DISPLAY
```
Disable access control by bellow command line. 
```
xhost +
```
Download and Install X server "VcXsrv" and set the parameter -ac in "Extra Settings" window when start the software, and set correct firewall settings as below.  
 
```
Go to Control Panel > System and Security > Windows Defender Firewall > Advanced Settings > Inbound Rules > New Rule... > Program > %ProgramFiles%\VcXsrv\vcxsrv.exe > Allow the connection > checked Domain/Private/Public > Named and Confirmed Rule.  
```
You can find here https://github.com/microsoft/WSL/issues/6430 for more informations.

### 2. IR model not readable error
The error message was:  
```
RuntimeError: The support of IR v6 has been removed from the product. Please, convert the original model using the Model Optimizer which comes with this version of the OpenVINO to generate supported IR version.  
```
Solved by re-download the model by below command line from the main directory.
