# SMPLify-X_Modify

Modified SMPLify-X.
1. Add perspective Camera.
2. Support a video as input.
3. Support a 3D joint as initialization.
4. Support IK solver.
5. Add ground plane support.


# Usage

## IK solver for body and hands.

```./run_3D_joints.sh```

## Run a single video for single person.

``` temp.sh ```


## MOVER Motion Reconstruction

1. PARE_initialization_SMPL-X model in Perspective Camera.


    * OpenPose: 
        condor_submit_bid 15 /ps/scratch/multi-ioi/hyi/singularity_openpose/example_run_openpose_hyi.sub
        to get preprocess/TEST_IMAGES/openpose
    * OneEuro Filter:
        ```/is/cluster/hyi/workspace/HCI/footskate_reducer/ground_detector/run_op_filter.sh```

    * Run PARE
        * run pare on video:
            to get  preprocess/TEST_IMAGES/pare_results
        ```
        conda activate pare_new
        /is/cluster/hyi/workspace/HCI/hdsr/projects/pare/run.sh video_path
        ```
        warnings while using PARE:
        ```
        (phosa_pt3d) hyi@ps048:/is/cluster/hyi/workspace/HCI/hdsr/projects/pare$ pip uninstall yacs
        Found existing installation: yacs 0.1.8
        Uninstalling yacs-0.1.8:
        Would remove:
            /home/hyi/anaconda3/envs/phosa_pt3d/lib/python3.7/site-packages/yacs-0.1.8.dist-info/*
            /home/hyi/anaconda3/envs/phosa_pt3d/lib/python3.7/site-packages/yacs/*
        Proceed (y/n)? y
        Successfully uninstalled yacs-0.1.8
        (phosa_pt3d) hyi@ps048:/is/cluster/hyi/workspace/HCI/hdsr/projects/pare$ pip install git+https://github.com/rbgirshick/yacs.git
        Collecting git+https://github.com/rbgirshick/yacs.git
        Cloning https://github.com/rbgirshick/yacs.git to /tmp/pip-req-build-3wfm07jt
        Requirement already satisfied: PyYAML in /home/hyi/anaconda3/envs/phosa_pt3d/lib/python3.7/site-packages (from yacs==0.1.8) (5.3.1)
        Building wheels for collected packages: yacs
        Building wheel for yacs (setup.py) ... done
        Created wheel for yacs: filename=yacs-0.1.8-py3-none-any.whl size=14747 sha256=038f4fef2b2d33ab82c7959312a85977f2e7b8781ef51cfc388170d4676b56f7
        Stored in directory: /tmp/pip-ephem-wheel-cache-fy4c9ni8/wheels/d8/33/cd/5232e413d59153f55a751de1673e9d2824c9ca1681dc16695f
        Successfully built yacs
        Installing collected packages: yacs
        Attempting uninstall: yacs
            Found existing installation: yacs 0.1.6
            Uninstalling yacs-0.1.6:
            Successfully uninstalled yacs-0.1.6
        Successfully installed yacs-0.1.8
        ```

        * merge PARE body as initialization with OPENPOSE hand, face
            to get  preprocess/TEST_IMAGES/mv_smplifyx_input
        ```
        /is/cluster/hyi/workspace/HCI/hdsr/phosa_ori_repo_total3D/hdsr/phosa_ori_repo/preprocess/pre_pare_scripts/run.sh

            # needs modify: 
                    DATA_FOLDER='/ps/scratch/hyi/HCI_dataset/holistic_scene_human/image_high_resolution/'
                    OUTPUT_FOLDER="/ps/scratch/hyi/HCI_dataset/holistic_scene_human/mv_smplify_input_pare3d_opfeetAnkles"
                    pare_result='/ps/scratch/hyi/HCI_dataset/holistic_scene_human/pare_result/pare_output.pkl'
                    cam_dir='/ps/scratch/hyi/total3dunderstand/holitic_human_sample_thr0.7/frame1'
                    CALIBRATION_FOLDER=/ps/scratch/hyi/HCI_dataset/holistic_scene_human/smplifyx_test/xml
        ```
    * SMPL2SMPLX model transfer to get PARE body pose parameters.
        ```
        condor_submit_bid 15 /is/cluster/hyi/cluster_scripts/test_smpl2smplx.sub
        ```
        Notion: body_pose: 21x3, in SMPLX format

        
    * Use PARE to refine pose results:
        * load previous optimized results:
            --pre_load="True" \
        * use pose as initialization
            --pre_load_pare_pose="True" \
        * use pose as prior
            --
    * Run SMPLify-X within user defined camera.
        * get SMPLifx-X input:  preprocess/TEST_IMAGES/mv_smplify_result
        ```
            ```
2. feet contact:
    ```conda activateNoNormalizedLoss_Body2ObjNew_SDF_total3D/${tmp_model_name}/obj_-1/
        ```/is/cluster/hyi/workspace/HCI/hdsr/POSA/interaction_cap/run.sh```