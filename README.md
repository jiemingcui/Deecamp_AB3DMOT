#Introduction

Based on AB3DMOT(https://github.com/xinshuoweng/AB3DMOT.git), apply the 128 lidar data from Deecamp 2020 and achieve the object tracking and visualization.
#Dependencies

This code has been tested on python 2.7 and 3.6, and also requires the following packages: 1. scikit-learn==0.19.2 2. filterpy==1.4.5 3. numba==0.43.1 4. matplotlib==2.2.3 5. pillow==6.2.2 6. opencv-python==3.4.3.18 7. glob2==0.6 8. llvmlite==0.32.1 (for python 3.6) or llvmlite==0.31.0 (for python 2.7)

One can either use the system python or create a virtual enviroment (virtualenv for python2, venv for python3) specifically for this project (https://www.pythonforbeginners.com/basics/how-to-use-python-virtualenv). To install required dependencies on the system python, please run the following command at the root of this code:
```
$ pip install -r requirements.txt
```
To install required dependencies on the virtual environment of the python (e.g., virtualenv for python2), please run the following command at the root of this code:
```
$ pip install virtualenv
$ virtualenv .
$ source bin/activate
$ pip install -r requirements.txt
```
#Dataset

The lidar data format is could see in the folder predictions
```
Deeacmp_AB3DMOT
├── data
│   ├── KITTI
│   │   │──prediction (your lidar data)
├── alfred
├── evaluation
├── results
```
If you want to visualize your lidar detection result, please change your lidar path
```
def load_bin(seq_name):
    file_name = seq_name + ".bin"
    v_f = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "yourfilepath/to/lidar(.bin)", file_name)
    pcs = load_pc_from_file(v_f)
    return pcs
```
#Visualization
```
$ python main.py prediction
```
#Acknowledgement

Part of the code is borrowed from "SORT"(https://github.com/abewley/sort)
