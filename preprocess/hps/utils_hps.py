from import_module import *
from utils_tool import *
project_path='/root/code/mover'
def test_openpose(input_dir, video_name, save_dir):
    shell = '/ps/scratch/multi-ioi/hyi/singularity_openpose/example_run_openpose_hyi.sub'
    # example_run_openpose_hyi.sub ？
    with open(shell, 'r') as fin:
        lines = fin.readlines()
    lines[1] = f'arguments = "exec --nv /ps/project/common/singularity_openpose/openpose.simg python3 \
        /ps/project/common/singularity_openpose/openpose_script.py \
        --input_dir {save_dir}/{video_name}/Color_flip_rename \
        --output_dir {save_dir}/{video_name}/Color_filp_rename_openpose"\n'

    with open(os.path.join(save_dir,video_name, 'process_shell', '3_test_openpose.sub'), 'w') as fout:
        for i in lines:
            fout.write(i)


def test_oneEuro_filter(video_name, save_dir):
# def test_oneEuro_filter(input_dir, video_name, save_dir): modified by lwz
    # shell = '/is/cluster/hyi/workspace/HCI/footskate_reducer/ground_detector/run_op_filter_input.sh' modified by lwz
    shell = f'{project_path}/preprocess/ground_detector/run_op_filter_input.sh'
    # run_op_filter_input.sh modified by lwz
    # need to verify /root/code/mover/preprocess/ground_detector/run_op_filter_input.sh save_dir/video_name
    with open(os.path.join(save_dir, video_name,'process_shell', '4_test_oneEuro.sh'), 'w') as fout:
        fout.write(f'{shell} {save_dir}/{video_name}')

    # save cluster job
    # shell = '/is/cluster/hyi/workspace/HCI/footskate_reducer/ground_detector/run_op_filter_input_cluster.sh'
    shell = f'{project_path}/preprocess/ground_detector/run_op_filter_input_cluster.sh'
    tmp_shell = '/is/cluster/hyi/workspace/HCI/hdsr/phosa_ori_repo_total3D/hdsr/phosa_ori_repo/preprocess/pre_shell/job_shell_template/test_gpu.sub'
    with open(tmp_shell, 'r') as fin:
        lines = fin.readlines()
    lines[0] = f'executable = {shell}\n'
    lines[1] = f'arguments = {save_dir}/{video_name}\n'
    with open(os.path.join(save_dir,video_name, 'process_shell', '4_test_oneEuro.sub'), 'w') as fout:
        for i in lines:
            fout.write(i)

def run_batch_smplifyx(input_dir, video_name, save_dir):
    shell = '/is/cluster/hyi/workspace/HCI/hdsr/phosa_ori_repo_total3D/hdsr/phosa_ori_repo/preprocess/pre_shell/job_shell_template/run_mv_smplifyx_batch_st0.sh'
    # save cluster job
    tmp_shell = '/is/cluster/hyi/workspace/HCI/hdsr/phosa_ori_repo_total3D/hdsr/phosa_ori_repo/preprocess/pre_shell/job_shell_template/test_specific_gpu.sub'
    with open(tmp_shell, 'r') as fin:
        lines = fin.readlines()
    lines[0] = f'executable = {shell}\n'
    lines[8] = 'requirements = TARGET.CUDACapability==10.0\n'
    # import pdb;pdb.set_trace()
    assert os.path.exists(os.path.join(save_dir, video_name, 'Color_flip_rename'))

    video_len = len(glob.glob(os.path.join(save_dir, video_name, "Color_flip_rename", "*.jpg")))
    lines[1] = f'arguments = {save_dir}/{video_name} {video_len}\n'
    # lines[9] = 'requirements = CUDADeviceName== "Tesla V100-SXM2-32GB" \n'
    lines[10] = f'queue 1\n'
    with open(os.path.join(save_dir,video_name, 'process_shell', '8_test_batch_smplify_st0.sub'), 'w') as fout:
        for i in lines:
            fout.write(i)

def run_batch_smplifyx_labelme(input_dir, video_name, save_dir, fps=30):
    if fps != 1:
        shell = '/is/cluster/hyi/workspace/HCI/hdsr/phosa_ori_repo_total3D/hdsr/phosa_ori_repo/preprocess/pre_shell/job_shell_template/run_mv_smplifyx_batch_st0_total3D_labelme.sh'
    else:
        shell = '/is/cluster/hyi/workspace/HCI/hdsr/phosa_ori_repo_total3D/hdsr/phosa_ori_repo/preprocess/pre_shell/job_shell_template/multiple_imgs/run_mv_smplifyx_batch_st0_total3D_labelme.sh'
    # save cluster job
    tmp_shell = '/is/cluster/hyi/workspace/HCI/hdsr/phosa_ori_repo_total3D/hdsr/phosa_ori_repo/preprocess/pre_shell/job_shell_template/test_specific_gpu.sub'
    with open(tmp_shell, 'r') as fin:
        lines = fin.readlines()
    lines[0] = f'executable = {shell}\n'
    lines[8] = 'requirements = TARGET.CUDACapability==10.0\n'
    # import pdb;pdb.set_trace()
    assert os.path.exists(os.path.join(save_dir, video_name, 'Color_flip_rename'))

    video_len = len(glob.glob(os.path.join(save_dir, video_name, "Color_flip_rename", "*.jpg")))
    lines[1] = f'arguments = {save_dir}/{video_name} {video_len}\n'
    # lines[9] = 'requirements = CUDADeviceName== "Tesla V100-SXM2-32GB" \n'
    lines[9] = 'requirements = (CUDADeviceName== "Tesla V100-SXM2-32GB" || CUDADeviceName== "Quadro RTX 6000") && UtsnameNodename =!= "g045" && UtsnameNodename =!= "g047" && UtsnameNodename =!= "g095" && UtsnameNodename =!= "g087"\n'
    lines[10] = f'queue 1\n'
    with open(os.path.join(save_dir,video_name, 'process_shell', '8_test_batch_smplify_st0_labelme.sub'), 'w') as fout:
        for i in lines:
            fout.write(i)
            

def test_posa(input_dir, video_name, save_dir):
    # split all pkl
    shell = '/is/cluster/hyi/workspace/HCI/hdsr/phosa_ori_repo_total3D/hdsr/phosa_ori_repo/preprocess/pre_SMPLX/run.sh'
    with open(os.path.join(save_dir, video_name,'process_shell', '8.01_split_pkl.sh'), 'w') as fout:
        fout.write(f'{shell} {save_dir}/{video_name}/smplifyx_results_st0/results/001_all.pkl')

    tmp_shell = '/is/cluster/hyi/workspace/HCI/hdsr/phosa_ori_repo_total3D/hdsr/phosa_ori_repo/preprocess/pre_shell/job_shell_template/test_nogpu.sub'
    with open(tmp_shell, 'r') as fin:
        lines = fin.readlines()
    lines[0] = f'executable = {shell}\n'
    lines[1] = f'arguments = {save_dir}/{video_name}/smplifyx_results_st0/results/001_all.pkl\n'
    with open(os.path.join(save_dir,video_name, 'process_shell', '8.01_split_pkl.sub'), 'w') as fout:
        for i in lines:
            fout.write(i)

    # run posa
    shell = '/is/cluster/hyi/workspace/HCI/hdsr/POSA/interaction_cap/run_input.sh'
    with open(os.path.join(save_dir, video_name,'process_shell', '8.02_test_posa.sh'), 'w') as fout:
        fout.write(f'{shell} {save_dir}/{video_name}/smplifyx_results_st0/results')

    tmp_shell = '/is/cluster/hyi/workspace/HCI/hdsr/phosa_ori_repo_total3D/hdsr/phosa_ori_repo/preprocess/pre_shell/job_shell_template/test_specific_gpu.sub'
    with open(tmp_shell, 'r') as fin:
        lines = fin.readlines()
    lines[0] = f'executable = {shell}\n'
    lines[8] = 'requirements = TARGET.CUDACapability==10.0\n'
    assert os.path.exists(os.path.join(save_dir, video_name, 'Color_flip_rename'))
    lines[1] = f'arguments = {save_dir}/{video_name}/smplifyx_results_st0/results\n'
    lines[10] = f'queue 1\n'
    with open(os.path.join(save_dir,video_name, 'process_shell', '8.02_test_posa.sub'), 'w') as fout:
        for i in lines:
            fout.write(i)

def test_posa_PROXD(input_dir, video_name, save_dir):
    # run posa
    shell = '/is/cluster/hyi/workspace/HCI/hdsr/POSA/interaction_cap/run_input.sh'
    with open(os.path.join(save_dir, video_name,'process_shell', '8.02_test_posa.sh'), 'w') as fout:
        fout.write(f'{shell} {save_dir}/{video_name}/smplifyx_results_PROXD/results')

    tmp_shell = '/is/cluster/hyi/workspace/HCI/hdsr/phosa_ori_repo_total3D/hdsr/phosa_ori_repo/preprocess/pre_shell/job_shell_template/test_specific_gpu.sub'
    with open(tmp_shell, 'r') as fin:
        lines = fin.readlines()
    lines[0] = f'executable = {shell}\n'
    lines[8] = 'requirements = TARGET.CUDACapability==10.0\n'
    assert os.path.exists(os.path.join(save_dir, video_name, 'Color_flip_rename'))
    lines[1] = f'arguments = {save_dir}/{video_name}/smplifyx_results_PROXD/results\n'
    lines[10] = f'queue 1\n'
    with open(os.path.join(save_dir,video_name, 'process_shell', '8.02_test_posa_PROXD.sub'), 'w') as fout:
        for i in lines:
            fout.write(i)

def test_posa_multiioi(input_dir, video_name, save_dir):
    # run posa
    shell = '/is/cluster/hyi/workspace/HCI/hdsr/POSA/interaction_cap/run_input_multiioi.sh'
    with open(os.path.join(save_dir, video_name,'process_shell', '8.02_test_posa_multiioi.sh'), 'w') as fout:
        fout.write(f'{shell} {save_dir}/{video_name}')

    tmp_shell = '/is/cluster/hyi/workspace/HCI/hdsr/phosa_ori_repo_total3D/hdsr/phosa_ori_repo/preprocess/pre_shell/job_shell_template/test_specific_gpu.sub'
    with open(tmp_shell, 'r') as fin:
        lines = fin.readlines()
    lines[0] = f'executable = {shell}\n'
    lines[8] = 'requirements = TARGET.CUDACapability==10.0\n'
    # assert os.path.exists(os.path.join(save_dir, video_name, 'Color_flip_rename'))
    lines[1] = f'arguments = {save_dir}/{video_name} {input_dir}/{video_name}\n'
    lines[10] = f'queue 1\n'
    with open(os.path.join(save_dir,video_name, 'process_shell', '8.02_test_posa_multiioi.sub'), 'w') as fout:
        for i in lines:
            fout.write(i)


def run_batch_smplifyx_newCamera_GPconstraints(input_dir, video_name, save_dir):
    
    shell = '/is/cluster/hyi/workspace/HCI/hdsr/phosa_ori_repo_total3D/hdsr/phosa_ori_repo/preprocess/pre_shell/job_shell_template/run_mv_smplifyx_batch_st2_gpConstraints.sh'
    # save cluster job
    tmp_shell = '/is/cluster/hyi/workspace/HCI/hdsr/phosa_ori_repo_total3D/hdsr/phosa_ori_repo/preprocess/pre_shell/job_shell_template/test_specific_gpu.sub'
    with open(tmp_shell, 'r') as fin:
        lines = fin.readlines()
    lines[0] = f'executable = {shell}\n'
    lines[8] = 'requirements = TARGET.CUDACapability==10.0\n'
    # import pdb;pdb.set_trace()
    assert os.path.exists(os.path.join(save_dir, video_name, 'Color_flip_rename'))

    video_len = len(glob.glob(os.path.join(save_dir, video_name, "Color_flip_rename", "*.jpg")))
    lines[1] = f'arguments = {save_dir}/{video_name} {video_len}\n'
    lines[9] = 'requirements = (CUDADeviceName== "Tesla V100-SXM2-32GB" || CUDADeviceName== "Quadro RTX 6000") && UtsnameNodename =!= "g045" && UtsnameNodename =!= "g047" && UtsnameNodename =!= "g095" && UtsnameNodename =!= "g087" \n'
    lines[10] = f'queue 1\n'
    with open(os.path.join(save_dir,video_name, 'process_shell', '10_test_batch_smplify_gpConstraint_st02.sub'), 'w') as fout:
        for i in lines:
            fout.write(i)

def run_batch_smplifyx_newCamera_GPconstraints_posaVersion(input_dir, video_name, save_dir, fps=30):
    if fps != 1 :
        shell = '/is/cluster/hyi/workspace/HCI/hdsr/phosa_ori_repo_total3D/hdsr/phosa_ori_repo/preprocess/pre_shell/job_shell_template/run_mv_smplifyx_batch_st2_gpConstraints_posaV.sh'
    else:
        shell = '/is/cluster/hyi/workspace/HCI/hdsr/phosa_ori_repo_total3D/hdsr/phosa_ori_repo/preprocess/pre_shell/job_shell_template/multiple_imgs/run_mv_smplifyx_batch_st2_gpConstraints_posaV.sh'
    # save cluster job
    tmp_shell = '/is/cluster/hyi/workspace/HCI/hdsr/phosa_ori_repo_total3D/hdsr/phosa_ori_repo/preprocess/pre_shell/job_shell_template/test_specific_gpu.sub'
    with open(tmp_shell, 'r') as fin:
        lines = fin.readlines()
    lines[0] = f'executable = {shell}\n'
    lines[8] = 'requirements = TARGET.CUDACapability==10.0\n'
    # import pdb;pdb.set_trace()
    assert os.path.exists(os.path.join(save_dir, video_name, 'Color_flip_rename'))

    video_len = len(glob.glob(os.path.join(save_dir, video_name, "Color_flip_rename", "*.jpg")))
    lines[1] = f'arguments = {save_dir}/{video_name} {video_len}\n'
    lines[9] = 'requirements = CUDADeviceName== "Tesla V100-SXM2-32GB" && UtsnameNodename =!= "g045" && UtsnameNodename =!= "g047" && UtsnameNodename =!= "g095" && UtsnameNodename =!= "g087"  && UtsnameNodename =!= "g093" \n'
    lines[10] = f'queue 1\n'
    with open(os.path.join(save_dir,video_name, 'process_shell', '10_test_batch_smplify_gpConstraint_st02_posaVersion.sub'), 'w') as fout:
        for i in lines:
            fout.write(i)
        
def run_batch_smplifyx_newCamera_GPconstraints_posaVersion_motionSmoothPrior_loadPrevious(input_dir, video_name, save_dir):
    
    shell = '/is/cluster/hyi/workspace/HCI/hdsr/phosa_ori_repo_total3D/hdsr/phosa_ori_repo/preprocess/pre_shell/job_shell_template/run_mv_smplifyx_batch_st2_gpConstraints_posaV_motionSmoothPrior_loadPrevious.sh'
    
    # save cluster job
    tmp_shell = '/is/cluster/hyi/workspace/HCI/hdsr/phosa_ori_repo_total3D/hdsr/phosa_ori_repo/preprocess/pre_shell/job_shell_template/test_specific_gpu.sub'
    with open(tmp_shell, 'r') as fin:
        lines = fin.readlines()
    lines[0] = f'executable = {shell}\n'
    lines[8] = 'requirements = TARGET.CUDACapability==10.0\n'
    # import pdb;pdb.set_trace()
    assert os.path.exists(os.path.join(save_dir, video_name, 'Color_flip_rename'))

    video_len = len(glob.glob(os.path.join(save_dir, video_name, "Color_flip_rename", "*.jpg")))
    lines[1] = f'arguments = {save_dir}/{video_name} {video_len} 1e11\n' # 1e11
    lines[9] = 'requirements = CUDADeviceName== "Tesla V100-SXM2-32GB" && UtsnameNodename =!= "g045" && UtsnameNodename =!= "g047" && UtsnameNodename =!= "g095" && UtsnameNodename =!= "g087" \n'
    lines[10] = f'queue 1\n'
    with open(os.path.join(save_dir,video_name, 'process_shell', '10_test_batch_smplify_gpConstraint_st02_posaVersion_motionSmoothPrior_loadP.sub'), 'w') as fout:
        for i in lines:
            fout.write(i)

# TODO: save to pare_result
def test_pare(input_dir, video_name, save_dir):
    # shell = '/is/cluster/hyi/workspace/HCI/hdsr/projects/pare/run.sh'
    shell = f'{project_path}/preprocess/pare_scripts/pare/run.sh'

    tmp_save_dir = f'{save_dir}/{video_name}/pare_results'
    os.makedirs(tmp_save_dir, exist_ok=True)
    with open(os.path.join(save_dir, video_name,'process_shell', '5_test_pare.sh'), 'w') as fout:
        fout.write(f'{shell} {save_dir}/{video_name}/Color_flip.mp4 {tmp_save_dir}')

    '''
    remove cluster？
    '''
    # save cluster job
    tmp_shell = '/is/cluster/hyi/workspace/HCI/hdsr/phosa_ori_repo_total3D/hdsr/phosa_ori_repo/preprocess/pre_shell/job_shell_template/test_specific_gpu.sub'
    with open(tmp_shell, 'r') as fin:
        lines = fin.readlines()
    lines[0] = f'executable = {shell}\n'
    lines[8] = 'requirements = TARGET.CUDACapability==10.1\n'
    # import pdb;pdb.set_trace()
    lines[1] = f'arguments = {save_dir}/{video_name}/Color_flip.mp4 {tmp_save_dir}\n'
    lines[10] = f'queue 1\n'
    with open(os.path.join(save_dir,video_name, 'process_shell', '5_test_pare.sub'), 'w') as fout:
        for i in lines:
            fout.write(i)


def op2smplifyx_withPARE(input_dir, video_name, save_dir):
    # shell = '/is/cluster/hyi/workspace/HCI/hdsr/phosa_ori_repo_total3D/hdsr/phosa_ori_repo/preprocess/pre_pare_scripts/run_input.sh'
    shell = f'{project_path}/preprocess/pare_scripts/run_input.sh'
    with open(os.path.join(save_dir, video_name,'process_shell', '6_op2smplifyx_input.sh'), 'w') as fout:
        fout.write(f'{shell} {save_dir}/{video_name}/Color_flip_rename \
            {save_dir}/{video_name}/mv_smplifyx_input_OneEuroFilter_PARE \
            {save_dir}/{video_name}/Color_flip_rename_openpose_OneEurofilter/Color_filp_rename_openpose \
            {save_dir}/{video_name}/pare_results/Color_flip_/pare_output.pkl')
    
    # save cluster job
    tmp_shell = '/is/cluster/hyi/workspace/HCI/hdsr/phosa_ori_repo_total3D/hdsr/phosa_ori_repo/preprocess/pre_shell/job_shell_template/test_gpu.sub'
    with open(tmp_shell, 'r') as fin:
        lines = fin.readlines()
    lines[0] = f'executable = {shell}\n'
    lines[1] = f'arguments = {save_dir}/{video_name}/Color_flip_rename \
            {save_dir}/{video_name}/mv_smplifyx_input_OneEuroFilter_PARE \
            {save_dir}/{video_name}/Color_flip_rename_openpose_OneEurofilter/Color_filp_rename_openpose \
            {save_dir}/{video_name}/pare_results/Color_flip_/pare_output.pkl\n'
    with open(os.path.join(save_dir,video_name, 'process_shell', '6_op2smplifyx_input.sub'), 'w') as fout:
        for i in lines:
            fout.write(i)

def smpl2smplx_pare(input_dir, video_name, save_dir):
    shell = '/is/cluster/hyi/cluster_scripts/test_smpl2smplx.sub'
    with open(shell, 'r') as fin:
        lines = fin.readlines()
    lines[1] = f'arguments = $(Process) \
        {save_dir}/{video_name}/mv_smplifyx_input_withPARE_PARE3DJointOneConfidence_OP2DJoints/meshes \
        {save_dir}/{video_name}/mv_smplifyx_input_withPARE_PARE3DJointOneConfidence_OP2DJoints/meshes_smplx\n'

    with open(os.path.join(save_dir, video_name,'process_shell', '7_smpl2smplx.sub'), 'w') as fout:
        for i in lines:
            fout.write(i)


def run_camera_gp_estimation_posaVersion(input_dir, video_name, save_dir):
    shell = '/is/cluster/hyi/workspace/HCI/hdsr/phosa_ori_repo_total3D/hdsr/phosa_ori_repo/preprocess/pre_shell/job_shell_template/run_estimate_camera_gp_st1_posaVersion.sh'
    # save cluster job
    tmp_shell = '/is/cluster/hyi/workspace/HCI/hdsr/phosa_ori_repo_total3D/hdsr/phosa_ori_repo/preprocess/pre_shell/job_shell_template/test_specific_gpu.sub'
    with open(tmp_shell, 'r') as fin:
        lines = fin.readlines()
    lines[0] = f'executable = {shell}\n'
    lines[8] = 'requirements = TARGET.CUDACapability==10.0\n'
    # import pdb;pdb.set_trace()
    assert os.path.exists(os.path.join(save_dir, video_name, 'Color_flip_rename'))

    video_len = len(glob.glob(os.path.join(save_dir, video_name, "Color_flip_rename", "*.jpg")))
    lines[1] = f'arguments = {save_dir}/{video_name} {video_len} 10000\n'
    # lines[1] = f'arguments = {save_dir}/{video_name} {video_len} $(Process) \n'
    lines[9] = 'requirements = (CUDADeviceName== "Tesla V100-SXM2-32GB" || CUDADeviceName== "Quadro RTX 6000") && UtsnameNodename =!= "g045" && UtsnameNodename =!= "g047" && UtsnameNodename =!= "g095" && UtsnameNodename =!= "g087"\n'
    lines[10] = f'queue 1\n'
    with open(os.path.join(save_dir,video_name, 'process_shell', '9_test_camera_gp_st01_posaVersion.sub'), 'w') as fout:
        for i in lines:
            fout.write(i)

def run_batch_smplifyx_newCamera_GPconstraints_withScene(input_dir, video_name, save_dir, fps=30):
    if fps != 1:
        # shell = '/is/cluster/hyi/workspace/HCI/hdsr/phosa_ori_repo_total3D/hdsr/phosa_ori_repo/preprocess/pre_shell/job_shell_template/run_mv_smplifyx_splitVideo_batch_st3_sceneConstraints_motionSmoothPrior.sh'
        shell = '/is/cluster/hyi/workspace/HCI/hdsr/phosa_ori_repo_total3D/hdsr/phosa_ori_repo/preprocess/pre_shell/job_shell_template/run_mv_smplifyx_splitVideo_batch_st3_sceneConstraints.sh'
    else:
        shell = '/is/cluster/hyi/workspace/HCI/hdsr/phosa_ori_repo_total3D/hdsr/phosa_ori_repo/preprocess/pre_shell/job_shell_template/multiple_imgs/run_mv_smplifyx_splitVideo_batch_st3_sceneConstraints_motionSmoothPrior.sh'
    # save cluster job
    tmp_shell = '/is/cluster/hyi/workspace/HCI/hdsr/phosa_ori_repo_total3D/hdsr/phosa_ori_repo/preprocess/pre_shell/job_shell_template/test_specific_gpu.sub'
    with open(tmp_shell, 'r') as fin:
        lines = fin.readlines()
    lines[0] = f'executable = {shell}\n'
    lines[8] = 'requirements = TARGET.CUDACapability==10.0\n'
    # import pdb;pdb.set_trace()
    assert os.path.exists(os.path.join(save_dir, video_name, 'Color_flip_rename'))

    video_len = len(glob.glob(os.path.join(save_dir, video_name, "Color_flip_rename", "*.jpg")))
    # single frame results;
    # lines[1] = f'arguments = {save_dir}/{video_name} {video_len} -1 1 $(Process) noDepth_sdf0_contact1e4\n' # last: depth loss weight.
    # lines[1] = f'arguments = {save_dir}/{video_name} {video_len} -1 1 $(Process) noDepth_sdf1e2_contact1e4\n'
    # lines[1] = f'arguments = {save_dir}/{video_name} {video_len} -1 {video_len} $(Process) gpContact1e2\n' # last: depth loss weight.
    
    lines[1] = f'arguments = {save_dir}/{video_name} {video_len} -1 {video_len} $(Process) noDepth_sdf1e2_contact1e4\n'
    # lines[1] = f'arguments = {save_dir}/{video_name} {video_len} -1 {video_len} $(Process) noDepth_sdf1e3\n' # last: depth loss weight.
    # lines[9] = 'requirements = (CUDADeviceName== "Tesla V100-SXM2-32GB" || CUDADeviceName==  "NVIDIA A100-SXM-80GB") && UtsnameNodename =!= "g045" && UtsnameNodename =!= "g047" && UtsnameNodename =!= "g095" && UtsnameNodename =!= "g087" \n'
    lines[9] = 'requirements = CUDADeviceName== "Tesla V100-SXM2-32GB" && UtsnameNodename =!= "g045" && UtsnameNodename =!= "g047" && UtsnameNodename =!= "g095" && UtsnameNodename =!= "g087" \n'
    # lines[10] = f'queue {video_len}\n'
    lines[10] = f'queue 1\n'
    with open(os.path.join(save_dir,video_name, 'process_shell', '30_test_batch_smplify_sceneConstraint.sub'), 'w') as fout:
        for i in lines:
            fout.write(i)

def run_hps_all(input_dir, video_name, save_dir, fps=30, input_video_flag='PROX'):
    run_dir = os.path.join(save_dir, video_name, 'process_shell')
    os.makedirs(run_dir, exist_ok=True)
    img_len = len(glob.glob(os.path.join(save_dir, video_name, "Color_flip_rename", "*.jpg")))
            
    test_oneEuro_filter(input_dir, video_name, save_dir)
    test_pare(input_dir, video_name, save_dir)
    op2smplifyx_withPARE(input_dir, video_name, save_dir)
    # run_batch_smplifyx(input_dir, video_name, save_dir)
    run_batch_smplifyx_labelme(input_dir, video_name, save_dir, fps=fps)
    test_posa(input_dir, video_name, save_dir)
    run_camera_gp_estimation_posaVersion(input_dir, video_name, save_dir)
    run_batch_smplifyx_newCamera_GPconstraints_posaVersion(input_dir, video_name, save_dir, fps=fps)
    run_batch_smplifyx_newCamera_GPconstraints_posaVersion_motionSmoothPrior_loadPrevious(input_dir, video_name, save_dir)
    
    preprocess_input_flag = False
    smplify_x_flag = False
    run_posa_flag = False
    run_opt_smplifyx_camera_flag = True

    if preprocess_input_flag:
        # logger.info(f'process input for smplify-x {preprocess_input_flag}')
        #### * preprocess input for smplify-x.
        script_name='3_test_openpose.sub'
        run_shell(script_name, run_dir)
        wait_for(os.path.join(save_dir, video_name, "Color_filp_rename_openpose"), img_len*2)
    
        script_name='4_test_oneEuro.sub'
        run_shell(script_name, run_dir)
        wait_for(os.path.join(save_dir, video_name, \
            "Color_flip_rename_openpose_OneEurofilter/Color_filp_rename_openpose"), img_len*2+1)
        
        ori_path = f'{save_dir}/{video_name}/Color_flip_rename'
        ori_tmp_path = f'{save_dir}/{video_name}/pare_results/Color_flip_'
        output_path = f'{save_dir}/{video_name}/pare_results/Color_flip_/tmp_images'
        os.makedirs(f'{save_dir}/{video_name}/pare_results/Color_flip_', exist_ok=True)

        script_name='5_test_pare.sub'
        if 'pare' in script_name and not os.path.exists(output_path):
            os.system(f'ln -s {ori_path} {output_path}')
        run_shell(script_name, run_dir)

        wait_for(os.path.join(save_dir, video_name, \
            "pare_results/Color_flip_"), 4, format='larger') # render OK has 5 files
        
        script_name='6_op2smplifyx_input.sub'
        run_shell(script_name, run_dir)
        wait_for(os.path.join(save_dir, video_name, \
            "mv_smplifyx_input_OneEuroFilter_PARE_PARE3DJointOneConfidence_OP2DJoints"), img_len+2)
    
    if smplify_x_flag:
        #### * smplify-x.
        script_name='8_test_batch_smplify_st0_labelme.sub'
        run_shell(script_name, run_dir)
        # wait_for(os.path.join(save_dir, video_name, \
        #     "smplifyx_results_st0/results"), 2, format='larger') # after split: exists 3 directories.
    
    if run_posa_flag:
        #### * run posa
        # os.system('sleep 1200s')
        script_name = '8.01_split_pkl.sub'
        run_shell(script_name, run_dir)
        wait_for(os.path.join(save_dir, video_name, \
            "smplifyx_results_st0/results/split"), img_len)
        os.system('sleep 600s')
        
        script_name='8.02_test_posa.sub'
        run_shell(script_name, run_dir)
        # wait_for(os.path.join(save_dir, video_name, \
        #     "smplifyx_results_st0/results/posa_contact_npy_newBottom"), img_len)

    if run_opt_smplifyx_camera_flag:
        #### * optimize camera;
        script_name='9_test_camera_gp_st01_posaVersion.sub'
        run_shell(script_name, run_dir)
        # wait_for(os.path.join(save_dir, video_name, \
        #     "smplifyx_results_st0_camera_gp_posaVersion_meanFeetAsInit/model_scene_0_0_lr0.002_end.pth"), img_len)
        
        #### * get better body under camera;
        # script_name='10_test_batch_smplify_gpConstraint_st02_posaVersion.sub'
        # run_shell(script_name, run_dir)
        # wait_for(os.path.join(save_dir, video_name, \
        #     "smplifyx_results_st2_newCamera_gpConstraints_posaVersion/results"), 2, format='larger')
        
        # ! this will be used in final demo.
        # script_name='10_test_batch_smplify_gpConstraint_st02_posaVersion_motionSmoothPrior_loadP.sub'
        # run_shell(script_name, run_dir)


if __name__ == '__main__':
    # input_dir: all dir save place.
    # video_name: sub_dir name.
    # save_dir: save results place.
    # fps: video fps.
    # input_video_flag='PROX'
    run_hps_all(input_dir, video_name, save_dir, fps=fps, input_video_flag=input_video_flag)