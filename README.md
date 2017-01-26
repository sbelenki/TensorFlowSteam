# Installing TensorFlow on Zotac NEN Steam Machine

## Motivation

Zotac NEN Steam Machine featuring GeForce GTX 970M with 1280 CUDA cores and 3GB of global memory that is suitable for CUDA cuDNN and TensorFlow installation.

## Downloading CUDA 7.5

At Home>>ComputeWorks>>CUDA ZONE>>Tools & Ecosystem>>CUDA Toolkit>>CUDA 7.5>>Downloads [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

Select OS: Linux
Architecture: x86_64
Distro: SteamOS
Version 1.0-beta
Intaller: runtime (local)

Take a note of the provided **CUDA link**.

Connect to Zotac machine through the ssh, then:

```
cd /home/desktop/Downloads
wget cuda_7.5.18_linux.run from the CUDA link
```

## Installing CUDA 7.5

In the Downloads directory:

```
sudo service lightdm stop
sudo sh cuda_7.5.18_linux.run
```

Select n for Drivers as installer will complain about different method used for previous drivers installation.

```
Do you accept the previously read EULA? (accept/decline/quit): accept
Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 352.39? ((y)es/(n)o/(q)uit): n
Install the CUDA 7.5 Toolkit? ((y)es/(n)o/(q)uit): y
Enter Toolkit Location [ default is /usr/local/cuda-7.5 ]: /home/cuda-7.5
Do you want to install a symbolic link at /usr/local/cuda? ((y)es/(n)o/(q)uit): y
Install the CUDA 7.5 Samples? ((y)es/(n)o/(q)uit): y
Enter CUDA Samples Location [ default is /home/desktop ]: /home/desktop/cuda
Installing the CUDA Toolkit in /home/cuda-7.5 ...
Missing recommended library: libGLU.so
Missing recommended library: libX11.so
Missing recommended library: libXi.so
Missing recommended library: libXmu.so
Missing recommended library: libGL.so

Installing the CUDA Samples in /home/desktop/cuda ...
Copying samples to /home/desktop/cuda/NVIDIA_CUDA-7.5_Samples now...
Finished copying samples.

===========
= Summary =
===========

Driver:   Not Selected
Toolkit:  Installed in /home/cuda-7.5
Samples:  Installed in /home/desktop/cuda, but missing recommended libraries

Please make sure that
 -   PATH includes /home/cuda-7.5/bin
 -   LD_LIBRARY_PATH includes /home/cuda-7.5/lib64, or, add /home/cuda-7.5/lib64 to /etc/ld.so.conf and run ldconfig as root

To uninstall the CUDA Toolkit, run the uninstall script in /home/cuda-7.5/bin
To uninstall the NVIDIA Driver, run nvidia-uninstall

Please see CUDA_Installation_Guide_Linux.pdf in /home/cuda-7.5/doc/pdf for detailed information on setting up CUDA.

***WARNING: Incomplete installation! This installation did not install the CUDA Driver. A driver of version at least 352.00 is required for CUDA 7.5 functionality to work.
To install the driver using this installer, run the following command, replacing <CudaInstaller> with the name of this run file:
    sudo <CudaInstaller>.run -silent -driver

Logfile is /tmp/cuda_install_4037.log
```

## Fixing Dependencies

### linux64bit

Download linux64bit from [https://developer.nvidia.com/linux64bit](https://developer.nvidia.com/linux64bit)

```
wget https://developer.nvidia.com/linux64bit
```

Start installation:

```
sudo service lightdm stop
sudo sh linux64bit

  The NVIDIA driver appears to have been installed previously using a different installer. To prevent potential
  conflicts, it is recommended either to update the existing installation using the same mechanism by which it was
  originally installed, or to uninstall the existing installation before installing this driver.

  Please review the message provided by the maintainer of this alternate installation method and decide how to
  proceed:
  ...
```

### libGLU.so

As per StackOverflow advice install libGLU.so
[Missing recommended library: libGLU.so](http://stackoverflow.com/questions/22360771/missing-recommended-library-libglu-so)

```
apt-get install libglu1-mesa libxi-dev libxmu-dev libglu1-mesa-dev
```

### More Missing libraries

Because not all necessary CUDA drivers can be installed through the ssh go to Debian desktop, select `Activities->Packages`, search and install all `355.00.28` packages.

## Configure Environment

[Installing CUDA Toolkit 7.5 on Ubuntu 14.04 Linux](http://www.r-tutor.com/gpu-computing/cuda-installation/cuda7.5-ubuntu)

```
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64
PATH=${CUDA_HOME}/bin:${PATH}
export PATH export PATH
```

## Compile CUDA Samples

```
sudo chmod -R 777 /home/desktop/cuda
cd /home/desktop/cuda/NVIDIA_CUDA-7.5_Samples/1_Utilities/deviceQuery
make
../../bin/x86_64/linux/release/deviceQuery

 CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)

Device 0: "GeForce GTX 960"
  CUDA Driver Version / Runtime Version          7.5 / 7.5
  CUDA Capability Major/Minor version number:    5.2
  Total amount of global memory:                 3063 MBytes (3212181504 bytes)
  (10) Multiprocessors, (128) CUDA Cores/MP:     1280 CUDA Cores
  GPU Max Clock rate:                            1038 MHz (1.04 GHz)
  Memory Clock rate:                             2505 Mhz
  Memory Bus Width:                              192-bit
  L2 Cache Size:                                 1572864 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(65536), 2D=(65536, 65536), 3D=(4096, 4096, 4096)
  Maximum Layered 1D Texture Size, (num) layers  1D=(16384), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(16384, 16384), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  2048
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 2 copy engine(s)
  Run time limit on kernels:                     Yes
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 7.5, CUDA Runtime Version = 7.5, NumDevs = 1, Device0 = GeForce GTX 960
Result = PASS
```
```
sudo apt-get install nvidia-settings

sudo service lightdm start
```

## Install cuDNN

Download cudnn-7.0-linux-x64-v4.0-prod.tgz from the [http://developer.download.nvidia.com](http://developer.download.nvidia.com) website

```
mv cudnn-7.0-linux-x64-v4.0-prod.tgz /usr/local/cuda/
tar xvzf cudnn-7.0-linux-x64-v4.0-prod.tgz

sudo mv cudnn-7.0-linux-x64-v4.0-prod.tgz /usr/local/
cd /usr/local/
sudo tar xvzf cudnn-7.0-linux-x64-v4.0-prod.tgz
sudo chmod a+r /usr/local/cuda/lib64/libcudnn*
sudo rm cudnn-7.0-linux-x64-v4.0-prod.tgz
```

## Install and Configure Anaconda

```
cd ~/Downloads
wget http://repo.continuum.io/archive/Anaconda3-4.0.0-Linux-x86_64.sh
bash Anaconda3-4.0.0-Linux-x86_64.sh

Select "yes" when asked if prepend Anaconda to .bashrc (the old one will be bakced up to .bashrc-anaconda3.bak)
```

Create TensorFlow Environment in Anaconda

```
conda create -n tensorflow python=3.5
source activate tensorflow
wget https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.8.0-cp34-cp34m-linux_x86_64.whl
mv tensorflow-0.8.0-cp34-cp34m-linux_x86_64.whl tensorflow-0.8.0-cp35-cp35m-linux_x86_64.whl
pip install --ignore-installed --upgrade ./tensorflow-0.8.0-cp35-cp35m-linux_x86_64.whl
source deactivate
```

Install Jupyter

```
conda install -n tensorflow jupyter=1.0.0
```

## Test TensorFlow

To test TensorFlow, run python

```
source activate tensorflow
```
```python
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> print(sess.run(hello))
Hello, TensorFlow!
>>> a = tf.constant(10)
>>> b = tf.constant(32)
>>> print(sess.run(a + b))
42
>>>
```
```
source deactivate
```

## IPython Notebooks via SSH

Thanks to this advice: [Remote Access to IPython Notebooks via SSH](https://coderwall.com/p/ohk6cg/remote-access-to-ipython-notebooks-via-ssh)

On remote machine: `source activate tensorflow`
On remote machine: `jupyter notebook --no-browser --port=8889`
On local machine: `ssh -N -L localhost:8888:localhost:8889 <REMOTE_USER>@<REMOTE_IP>`
On local machine browser: `localhost:8888`

Work with the Jupyter notebook...

After finishing: `source deactivate`


## References

[NVIDIA CUDA 7.5](https://developer.nvidia.com/cuda-downloads)

[NVIDIA cuDNN](https://developer.nvidia.com/cudnn)

[TensorFlow Download and Setup](https://www.tensorflow.org/get_started/os_setup)

