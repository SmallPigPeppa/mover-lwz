from import_module import *

def read_txt(input_fn, end=-1):
    with open(input_fn, 'r') as fin:
        all_list = fin.read().splitlines()
    if end != -1:
        all_list = [one[:end] for one in all_list]
    return all_list

def viz_results(input_path, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.isfile(input_path):
        os.system(f'cp {input_path} {save_path}')
    else:
        os.system(f'cp -r {input_path} {save_path}')

def process_scans(input_dir, video_name, save_dir):
    shell = '/home/hyi/anaconda3/envs/hdsr_new_bvh/bin/python /is/cluster/hyi/workspace/HCI/hdsr/phosa_ori_repo_total3D/hdsr/phosa_ori_repo/preprocess/pre_PROX/pre_prox_scan.py'
    with open(os.path.join(save_dir, video_name,'process_shell', '01_pre_prox_scan.sh'), 'w') as fout:
        fout.write(f'{shell} {video_name} {save_dir}/{video_name}/prox_scans_pre')

def get_gp_gt(input_dir, video_name, save_dir):
    shell = '/home/hyi/anaconda3/envs/hdsr_new_bvh/bin/python /is/cluster/hyi/workspace/HCI/hdsr/phosa_ori_repo_total3D/hdsr/phosa_ori_repo/preprocess/pre_Scan/get_scans_gp.py'
    with open(os.path.join(save_dir, video_name,'process_shell', '01_get_gp_gt_from_scan.sh'), 'w') as fout:
        fout.write(f'{shell} {save_dir}/{video_name}/prox_scans_pre {save_dir}/{video_name}/prox_scans_gp')


def get_input_list(input_path):
    import pickle
    with open(input_path, 'rb') as fin:
        perframe_det_dict = pickle.load(fin)
        filter_obj_list = perframe_det_dict['filter_obj_list']
    return filter_obj_list


def run_evaluate_HPS(save_dir, video_name, sub_dir):
    shell = '/is/cluster/hyi/workspace/HCI/hdsr/phosa_ori_repo_total3D/hdsr/phosa_ori_repo/preprocess/pre_shell/job_shell_template/run_eval_prox_qualitative.sh'
    with open(os.path.join(save_dir, video_name,'process_shell', f'eval_{sub_dir}.sh'), 'w') as fout:
        fout.write(f'{shell} {save_dir}/{video_name} {sub_dir} {video_name}')
    # save cluster job
    tmp_shell = '/is/cluster/hyi/workspace/HCI/hdsr/phosa_ori_repo_total3D/hdsr/phosa_ori_repo/preprocess/pre_shell/job_shell_template/test_specific_gpu.sub'
    with open(tmp_shell, 'r') as fin:
        lines = fin.readlines()
    lines[0] = f'executable = {shell}\n'
    lines[8] = 'requirements = TARGET.CUDACapability==10.0\n'
    # import pdb;pdb.set_trace()
    lines[1] = f'arguments = {save_dir}/{video_name} {sub_dir} {video_name}\n'
    lines[10] = f'queue 1\n'
    with open(os.path.join(save_dir,video_name, 'process_shell', f'eval_{sub_dir}.sub'), 'w') as fout:
        for i in lines:
            fout.write(i)


def get_merge_template_obj(obj_dir, save_dir, interval=1):
    import trimesh
    all_cam_list = sorted(glob.glob(os.path.join(obj_dir, '*_cam_CS.obj')))
    split = 100
    total_num = math.ceil((len(all_cam_list) / split * 1.0))
    # import pdb;pdb.set_trace()
    for i in range(total_num):
        cnt = 0
        cam_all_mesh = []
        all_mesh = []
        for idx, one in enumerate(all_cam_list[i*split:(i+1)*split]):
            print(one)
            if cnt % interval == 0:
                mesh = trimesh.load(one, process=False)

                one_world = one[:-11]
                mesh_world = trimesh.load(one_world, process=False)
                all_mesh.append(mesh_world)
                cam_all_mesh.append(mesh)
            cnt += 1
        all_model = merge_mesh(cam_all_mesh)
        all_model_world = merge_mesh(all_mesh)
        os.makedirs(save_dir, exist_ok=True)
        all_model.export(os.path.join(save_dir, f'all_model_cam_{i}.obj'))
        all_model_world.export(os.path.join(save_dir, f'all_model_{i}.obj'))
    
def merge_mesh(mesh_list):
    import trimesh
    import numpy as np
    all_v = []
    all_f = []
    v_num = 0
    for one in mesh_list:
        v, f = one.vertices, one.faces
        all_f.append(f + v_num)
        v_num += v.shape[0]
        all_v.append(v)

    return trimesh.Trimesh(np.concatenate(all_v), np.concatenate(all_f), process=False)        
def check_finish(save_dir, length, format='equal'):
    if not os.path.exists(save_dir):
        return False
    if os.path.isfile(save_dir):
        return True
    save_len = len(os.listdir(save_dir))
    # print(save_len, length)
    if format == 'equal':
        return save_len == length
    elif format == 'larger':
        return save_len >= length
        
def wait_for(save_dir, length, duration=1800, format='equal'):
    import time
    start = time.time()
    end = start
    # print(len)
    # print(check_finish(save_dir, length))
    # import pdb;pdb.set_trace()
    logger.info(f'wait for {save_dir}')
    while(not check_finish(save_dir, length, format) and end < start + duration):
        end = time.time()
        tmp_d = end-start
        if tmp_d % 10 < 3e-3:
            logger.info(f'wait {tmp_d}s')

        # print(f'wait {tmp_d}')
    logger.info(f'work fine for {save_dir}')
    os.system('sleep 1s')

def run_shell(script_name, run_dir):
    logger.info(f'run {script_name}')
    if 'openpose' in script_name:
        bid = 10
    else:
        bid = 100
    script_name = os.path.join(run_dir, script_name)
    os.system(f'condor_submit_bid {bid} {script_name}')