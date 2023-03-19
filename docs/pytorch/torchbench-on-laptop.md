---
slug: torchbench-laptop
title: Running TorchBench on my Gaming Laptop
authors: ajax
tags: [pytorch]
---

walkalong guide of my experience running TorchBench on my gaming laptop 

<!-- truncate -->

Here are my laptop specs for reference
* Model: ASUS RoG Zephyrus G14 2022
* Proc: AMD Ryzen 6900HS
* Graphics: AMD Radeon 6700S
* RAM: 16GB RAM
* OS: Ubuntu 22.04 LTS

# Setup
I'm following [Xu Zhao's presentation from the PyTorch conference in Nov 2022](https://www.youtube.com/watch?v=FKpqn4_KuPU) for the initial setup. 

1. Clone the repo: `git clone https://github.com/pytorch/benchmark.git; pushd benchmark;`

2. Run `python3 install.py`
    * First, you'll probably need to make sure you are root or use `sudo`. I ran into:
    ```
    Checking out https://ossci-datasets.s3.amazonaws.com/torchbench/models/maml_omniglot/batch.pt to /home/dictator/development/pytorch/benchmark/torchbenchmark/models/maml_omniglot/batch.pt
    OK
    decompressing input tarball: Super_SloMo_inputs.tar.gz...Traceback (most recent call last):
    File "/home/dictator/development/pytorch/benchmark/install.py", line 107, in <module>
        decompress_input()
    File "/home/dictator/development/pytorch/benchmark/install.py", line 59, in decompress_input
        tar.extractall(path=decompress_dir)
    File "/usr/lib/python3.10/tarfile.py", line 2059, in extractall
        self.extract(tarinfo, path, set_attrs=not tarinfo.isdir(),
    File "/usr/lib/python3.10/tarfile.py", line 2100, in extract
        self._extract_member(tarinfo, os.path.join(path, tarinfo.name),
    File "/usr/lib/python3.10/tarfile.py", line 2173, in _extract_member
        self.makefile(tarinfo, targetpath)
    File "/usr/lib/python3.10/tarfile.py", line 2214, in makefile
        with bltn_open(targetpath, "wb") as target:
    PermissionError: [Errno 13] Permission denied: '/home/dictator/development/pytorch/benchmark/torchbenchmark/data/.data/Super_SloMo_inputs/data/create_dataset.py'
    ```
    * Next, detectron2 fails to install with the following error:

    ```
    Error for /home/dictator/development/pytorch/benchmark/torchbenchmark/models/detectron2_fasterrcnn_r_101_c4:
    ---------------------------------------------------------------------------
    WARNING: Did not find branch or tag 'c470ca3', assuming revision or ref.
    error: subprocess-exited-with-error
    
    × python setup.py bdist_wheel did not run successfully.
    │ exit code: 1
    ╰─> [640 lines of output]
        /tmp/pip-req-build-n74inrik/detectron2/layers/csrc/ROIAlignRotated/ROIAlignRotated.h -> /tmp/pip-req-build-n74inrik/detectron2/layers/csrc/ROIAlignRotated/ROIAlignRotated.h [skipped, no changes]
        /tmp/pip-req-build-n74inrik/detectron2/layers/csrc/box_iou_rotated/box_iou_rotated.h -> /tmp/pip-req-build-n74inrik/detectron2/layers/csrc/box_iou_rotated/box_iou_rotated.h [skipped, no changes]
        /tmp/pip-req-build-n74inrik/detectron2/layers/csrc/cocoeval/cocoeval.h -> /tmp/pip-req-build-n74inrik/detectron2/layers/csrc/cocoeval/cocoeval.h [skipped, no changes]
        /tmp/pip-req-build-n74inrik/detectron2/layers/csrc/deformable/deform_conv.h -> /tmp/pip-req-build-n74inrik/detectron2/layers/csrc/deformable/deform_conv.h [skipped, no changes]
        /tmp/pip-req-build-n74inrik/detectron2/layers/csrc/nms_rotated/nms_rotated.h -> /tmp/pip-req-build-n74inrik/detectron2/layers/csrc/nms_rotated/nms_rotated.h [skipped, no changes]
        /tmp/pip-req-build-n74inrik/detectron2/layers/csrc/vision.cpp -> /tmp/pip-req-build-n74inrik/detectron2/layers/csrc/vision.cpp [skipped, no changes]
        /tmp/pip-req-build-n74inrik/detectron2/layers/csrc/box_iou_rotated/box_iou_rotated_utils.h -> /tmp/pip-req-build-n74inrik/detectron2/layers/csrc/box_iou_rotated/box_iou_rotated_utils_hip.h [skipped, already hipified]
        /tmp/pip-req-build-n74inrik/detectron2/layers/csrc/nms_rotated/nms_rotated_cpu.cpp -> /tmp/pip-req-build-n74inrik/detectron2/layers/csrc/nms_rotated/nms_rotated_cpu_hip.cpp [skipped, already hipified]
        /tmp/pip-req-build-n74inrik/detectron2/layers/csrc/cocoeval/cocoeval.cpp -> /tmp/pip-req-build-n74inrik/detectron2/layers/csrc/cocoeval/cocoeval.cpp [skipped, no changes]
        /tmp/pip-req-build-n74inrik/detectron2/layers/csrc/box_iou_rotated/box_iou_rotated_cpu.cpp -> /tmp/pip-req-build-n74inrik/detectron2/layers/csrc/box_iou_rotated/box_iou_rotated_cpu_hip.cpp [skipped, already hipified]
        /tmp/pip-req-build-n74inrik/detectron2/layers/csrc/ROIAlignRotated/ROIAlignRotated_cpu.cpp -> /tmp/pip-req-build-n74inrik/detectron2/layers/csrc/ROIAlignRotated/ROIAlignRotated_cpu.cpp [skipped, no changes]
        /tmp/pip-req-build-n74inrik/detectron2/layers/csrc/nms_rotated/nms_rotated_cuda.cu -> /tmp/pip-req-build-n74inrik/detectron2/layers/csrc/nms_rotated/nms_rotated_hip.hip [skipped, already hipified]
        /tmp/pip-req-build-n74inrik/detectron2/layers/csrc/deformable/deform_conv_cuda_kernel.cu -> /tmp/pip-req-build-n74inrik/detectron2/layers/csrc/deformable/deform_conv_hip_kernel.hip [skipped, already hipified]
        /tmp/pip-req-build-n74inrik/detectron2/layers/csrc/deformable/deform_conv_cuda.cu -> /tmp/pip-req-build-n74inrik/detectron2/layers/csrc/deformable/deform_conv_cuda.cu [skipped, no changes]
    ...
        [end of output]
    
    note: This error originates from a subprocess, and is likely not a problem with pip.
    error: legacy-install-failure

    × Encountered error while trying to install package.
    ╰─> detectron2


    ```

    I'm assuming this might have to do with my ROCm setup not being "officially" supported by the folks over at AMD. Others claim they don't show too much love towards ROCm support on consumer graphics cards, and probably care even less about a mobile card. 


