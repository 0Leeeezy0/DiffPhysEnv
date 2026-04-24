#! /bin/bash

echo "============== Setup Options =============="
read -p "1.set python env?(true/false): " env
read -p "2.download&&set miniconda3?(true/false): " miniconda3
read -p "3.download&&install airsim?(true/false): " airsim
read -p "4.download diff-phys-drone-cuda12?(true/false): " diff_phys_drone_cuda12
read -p "5.download px4-sitl?(true/false): " px4_sitl
read -p "6.download genesis?(true/false): " genesis
read -p "7.copy robot files to current dir?(true/false): " copy_robot

if $env; then
    echo "============== Create python3 env [fpv] =============="
    conda create -n fpv python=3.11
    conda activate fpv
    echo "============== Install pytorch2.8.0 =============="
    pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
    echo "============== Install matplotlib =============="
    pip install matplotlib
    echo "============== Install ffmpeg =============="
    sudo apt install ffmpeg -y
fi

if $miniconda3; then
    echo "============== Download miniconda3 =============="
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm ~/miniconda3/miniconda.sh
    echo "============== Set miniconda3 =============="
    source ~/miniconda3/bin/activate
    conda init --all
fi

if $airsim; then
    echo "============== Download airsim =============="
    git clone https://github.com/microsoft/AirSim.git
    cd AirSim/PythonClient
    echo "============== Install airsim =============="
    python setup.py install
fi

if $diff_phys_drone_cuda12; then
    echo "============== Download DiffPhysDrone =============="
    git clone https://github.com/0Leeeezy0/DiffPhysDrone.git
fi

if $px4_sitl; then
    echo "============== Download PX4-Sitl =============="
    git clone https://github.com/PX4/PX4-Autopilot.git
    cd PX4-Autopilot
    bash ./Tools/setup/ubuntu.sh --no-nuttx --no-sim-tools
    cd build
    cmake ..
    cd ..
    make px4_sitl none_iris
fi

if $genesis; then
    echo "============== Download Genesis =============="
    pip install git+https://github.com/Genesis-Embodied-AI/Genesis.git
fi

if $copy_robot; then
    echo "============== Copy robot files =============="
    echo "Available robot types:"
    for dir in robot/*/; do
        basename "$dir"
    done
    read -p "Enter robot type to copy (e.g., drone, rigid, or 'all'): " robot_type
    if [ "$robot_type" = "all" ]; then
        for dir in robot/*/; do
            type_name=$(basename "$dir")
            echo "Copying robot/$type_name to ./robot_$type_name/"
            cp -r "robot/$type_name" "./robot_${type_name}"
        done
    else
        if [ -d "robot/$robot_type" ]; then
            echo "Copying robot/$robot_type to ./robot_${robot_type}/"
            cp -r "robot/$robot_type" "./robot_${robot_type}"
        else
            echo "Error: robot/$robot_type does not exist!"
        fi
    fi
fi
