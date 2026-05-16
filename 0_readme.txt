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
