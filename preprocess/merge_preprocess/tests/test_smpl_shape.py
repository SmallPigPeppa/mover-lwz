import torch
import trimesh
from smplx import SMPL
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from pare.core.config import SMPL_MODEL_DIR
from pare.utils.mesh_viewer import MeshViewer


def main():
    smpl = SMPL(
        SMPL_MODEL_DIR,
        batch_size=1,
        create_transl=False
    )

    faces = smpl.faces

    while True:
        # dim = int(input('enter dim: '))
        # val = float(input('enter value: '))

        betas = torch.zeros(1, 10)
        # betas[:,:] = 0.7
        # betas[:,dim] = val

        s = smpl(betas=betas)

        verts = s.vertices.cpu().detach().numpy()[0]

        print(betas, 'height:', verts[:,1].max() - verts[:,1].min())

        m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        m.show()

@torch.no_grad()
def gui():
    mv = MeshViewer(width=1920, height=1920, body_color=(0.8, 0.8, 0.8, 1.0))

    device = 'cpu'
    smpl = SMPL(SMPL_MODEL_DIR, gender='neutral').to(device)
    f = smpl.faces
    v = smpl().vertices.cpu().numpy().squeeze()

    mv.update_mesh(v, f)

    fig, ax = plt.subplots(nrows=10)

    b = torch.zeros(1, 10, device=device)

    b0 = Slider(ax[0], 'beta[0]', -5, 5, valinit=0.0, valstep=0.0001)
    b1 = Slider(ax[1], 'beta[1]', -5, 5, valinit=0.0, valstep=0.0001)
    b2 = Slider(ax[2], 'beta[2]', -5, 5, valinit=0.0, valstep=0.0001)
    b3 = Slider(ax[3], 'beta[3]', -5, 5, valinit=0.0, valstep=0.0001)
    b4 = Slider(ax[4], 'beta[4]', -5, 5, valinit=0.0, valstep=0.0001)
    b5 = Slider(ax[5], 'beta[5]', -5, 5, valinit=0.0, valstep=0.0001)
    b6 = Slider(ax[6], 'beta[6]', -5, 5, valinit=0.0, valstep=0.0001)
    b7 = Slider(ax[7], 'beta[7]', -5, 5, valinit=0.0, valstep=0.0001)
    b8 = Slider(ax[8], 'beta[8]', -5, 5, valinit=0.0, valstep=0.0001)
    b9 = Slider(ax[9], 'beta[9]', -5, 5, valinit=0.0, valstep=0.0001)

    def u0(val): b[:, 0] = val;v=smpl(betas=b).vertices.cpu().numpy()[0];mv.update_mesh(v,f);ax[0].set_title(f'Height: {v[:,1].max()-v[:,1].min():.2f}m.')
    def u1(val): b[:, 1] = val;v=smpl(betas=b).vertices.cpu().numpy()[0];mv.update_mesh(v,f);ax[0].set_title(f'Height: {v[:,1].max()-v[:,1].min():.2f}m.')
    def u2(val): b[:, 2] = val;v=smpl(betas=b).vertices.cpu().numpy()[0];mv.update_mesh(v,f);ax[0].set_title(f'Height: {v[:,1].max()-v[:,1].min():.2f}m.')
    def u3(val): b[:, 3] = val;v=smpl(betas=b).vertices.cpu().numpy()[0];mv.update_mesh(v,f);ax[0].set_title(f'Height: {v[:,1].max()-v[:,1].min():.2f}m.')
    def u4(val): b[:, 4] = val;v=smpl(betas=b).vertices.cpu().numpy()[0];mv.update_mesh(v,f);ax[0].set_title(f'Height: {v[:,1].max()-v[:,1].min():.2f}m.')
    def u5(val): b[:, 5] = val;v=smpl(betas=b).vertices.cpu().numpy()[0];mv.update_mesh(v,f);ax[0].set_title(f'Height: {v[:,1].max()-v[:,1].min():.2f}m.')
    def u6(val): b[:, 6] = val;v=smpl(betas=b).vertices.cpu().numpy()[0];mv.update_mesh(v,f);ax[0].set_title(f'Height: {v[:,1].max()-v[:,1].min():.2f}m.')
    def u7(val): b[:, 7] = val;v=smpl(betas=b).vertices.cpu().numpy()[0];mv.update_mesh(v,f);ax[0].set_title(f'Height: {v[:,1].max()-v[:,1].min():.2f}m.')
    def u8(val): b[:, 8] = val;v=smpl(betas=b).vertices.cpu().numpy()[0];mv.update_mesh(v,f);ax[0].set_title(f'Height: {v[:,1].max()-v[:,1].min():.2f}m.')
    def u9(val): b[:, 9] = val;v=smpl(betas=b).vertices.cpu().numpy()[0];mv.update_mesh(v,f);ax[0].set_title(f'Height: {v[:,1].max()-v[:,1].min():.2f}m.')

    b0.on_changed(u0)
    b1.on_changed(u1)
    b2.on_changed(u2)
    b3.on_changed(u3)
    b4.on_changed(u4)
    b5.on_changed(u5)
    b6.on_changed(u6)
    b7.on_changed(u7)
    b8.on_changed(u8)
    b9.on_changed(u9)

    plt.show()

if __name__ == '__main__':
    gui()
    # main()