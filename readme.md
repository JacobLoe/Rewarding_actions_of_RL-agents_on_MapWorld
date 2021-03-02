# HRL for text based games
How to create a conda environment for HRL, adopted after [Shiyu Chen](https://www.chenshiyu.top/blog/2019/06/19/Tutorial-Installation-and-Configuration-of-MuJoCo-Gym-Baselines/):



### mujoco 
First download [mjpro150](https://www.roboti.us/index.html) and the
[mjpro150 repo](https://github.com/openai/mujoco-py/releases/tag/1.50.1.0).

Install mujoco:
```
mkdir ~/.mujoco
cp mjpro150_linux.zip ~/.mujoco
cd ~/.mujoco
unzip mjpro150_linux.zip
mv mjpro150_linux mjpro150
```

Copy the license
```
cp mjkey.txt ~/.mujoco
cp mjkey.txt ~/.mujoco/mjpro150/bin
```

Add the environment variables to `~/.bashrc`:
```
gedit ~/.bashrc

export LD_LIBRARY_PATH=/home/csy/.mujoco/mjpro/bin${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export MUJOCO_KEY_PATH=/home/csy/.mujoco${MUJOCO_KEY_PATH}
```

Test MuJoCo:
```
cd ~/.mujoco/mjpor150/bin
./simulate ../model/humanoid.xml
```

### mujoco-py

Create anaconda environment:
```
conda create -n mujoco-gym python=3.6
conda activate mujoco-gym
```

Install mujoco-py from repo:
```
cd mujoco-py-1.50.1.0
pip install -e .
```

Install patchelf:
```
Download: https://nixos.org/releases/patchelf/patchelf-0.9/patchelf-0.9.tar.gz
cd patchelf-0.9
./configure
make
sudo make install
```

Test mujoco-py:
```
python tests/test_mujoco.py
```

### openai gym

Install:
```
pip install gym[all]
```

Add glew to .bashrc:
```
gedit ~/.bashrc
export LD_PRELOAD=/usr/lib/libGLEW.so
```
Test:
```
python tests/test_gym.py
```

### tensorflow

```
conda install tensorflow-gpu==1.15
conda install tensorflow-probability==0.8.0
pip install tf-agents==0.3.0
```

