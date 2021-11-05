# Computer Pointer Controller

Run the code -1h
11/3
brush up the code -1h
Write up the Read me -1hour
Rubics -10min
review 1h

*TODO:* Write a short introduction to your project

## Project Set Up and Installation
1. Clone (or download) this repo
```
git clone 

cd 
```

2. Create a virtual environment
e.g.
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
python3  <path/to/downloader.py> --name <model-name>
```

## Demo
Navigate to the `src/` directory and execute the `main.py` script with the required arguments.
See the [documentation](#documentation) for more info.
```
python3 main.py <arguments>
```
For example
```
python main.py 
```


## Documentation
*TODO:* Include any documentation that users might need to better understand your project code. For instance, this is a good place to explain the command line arguments that your project supports.


## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.
for example: Benchmarking results for models of different precisions
Discussion of the difference in the results among the models with different precisions (for instance, are some models more accurate than others?).

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.

------------------------------------------
Rubic memo point

The project demonstrates an ability to show output of intermediate models for visualization.

The code allows the user to set a flag that can display the outputs of intermediate models.

The output is shown using a visualization of the output model (not just printed).


-----------------------------------------
Where possible, default arguments are used for when the user does not specify the arguments.

A --help option should be present and should display information about the different configurations.

# Others
1. DISPLAY error
The message was xhost:  unable to open display ":0.0"
Solved by:
Set DISPLAY correctly for WSL2 system, add below line to ~/.bashrc
```
    export DISPLAY="$(grep nameserver /etc/resolv.conf | sed 's/nameserver //'):0"
```
Check if it is correctly set
```
echo $DISPLAY
```
Disable access control by bellow command line. 
```
xhost +
```
Download X server "VcXsrv" and set the correct firewall settings as below

I went to Control Panel > System and Security > Windows Defender Firewall > Advanced Settings > Inbound Rules > New Rule... > Program > %ProgramFiles%\VcXsrv\vcxsrv.exe > Allow the connection > checked Domain/Private/Public > Named and Confirmed Rule.
You can find here https://github.com/microsoft/WSL/issues/6430 for more detail

2. IR model not readable error
The error message was:  
RuntimeError: The support of IR v6 has been removed from the product. Please, convert the original model using the Model Optimizer which comes with this version of the OpenVINO to generate supported IR version.  

Solved by re download the model by below command line from the main directory:

```
python /opt/intel/openvino_2021/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name landmarks-regression-retail-0009
```