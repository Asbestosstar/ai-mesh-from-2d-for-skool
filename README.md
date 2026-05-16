# ai-mesh-from-2d-for-skool
for school everyone else ignore
Since my GPUs only use CUDA 6.1 there is dep sphagetti and the deps must be installed in a specific order. I have given to my school a link to download the full python instance which contains all the dep sphagetti in a 10GB zip. Simply installing from requirnments.txt will not work on CUDA 6.1 since I had to install in a certain order and remove version in order to ger a working instance. However if you are on a newer version of CUDA like CUDA 7 you may be able to install the newest versions of the packages without issues or specific versions of packages.


you need to source into this folder with python

cd <path to folder>
source <path to folder>/bin/activate

This was tested on RHEL9 on a TESLA P4, your GPU should have at least CUDA 6.1

paddle_depth_to_mesh.py is to generate an image with default depth anything 2

paddle_depth_to_mesh_trained.py is to use the custom trained version


python paddle_depth_to_mesh_trained.py -i 152440709_0_final.png -o 152440709_0_final.png.obj







for training you need this file

render_train_dataset.py this file generates depth with blender, you will need to edit it to run it through blender


/home/rhel/Descargas/blender-5.0.1-linux-x64/blender   --background   --python /home/rhel/paddle_depth_env/render_train_dataset.py




Once you get the depth images you just run
train_depth_anything_v2_lora_fp16.py

and it will fine tune
