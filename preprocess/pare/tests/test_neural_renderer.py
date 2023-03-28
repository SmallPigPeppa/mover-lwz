import neural_renderer as nr
import torch

device = 'cuda'


vertices, faces = nr.load_obj(obj_filename)
vertices = vertices[None, :, :]  # [num_vertices, XYZ] -> [batch_size=1, num_vertices, XYZ]
faces = faces[None, :, :]  # [num_faces, 3] -> [batch_size=1, num_faces, 3]

# create texture [batch_size=1, num_faces, texture_size, texture_size, texture_size, RGB]
textures = torch.ones(1, faces.shape[1], 2, 2, 2, 3, dtype=torch.float32).cuda()

IMG_SIZE = 224
FOCAL_L = 5000.0
R = torch.eye(3).to(device)
s, tx, ty = torch.from_numpy(cam) #prediction from SPIN estimate
T = torch.Tensor([tx, ty, 2* FOCAL_L / ((IMG_SIZE * s) + 1e-9)]).to(device)
R[2,2] *= -1.0
RT = torch.Tensor(3,4).to(device) # camera extrinsic parameter
RT[:,0:3] = R
RT[:,3] = T
K = torch.Tensor([[FOCAL_L, 0, IMG_SIZE/2],
                  [0, FOCAL_L, IMG_SIZE/2],
                  [0, 0, 1]]).to(device) # camera intrinsic parameter
P = torch.matmul(K, RT).unsqueeze(0) # projection matrix batch_size x 3 x 4
print(P)

renderer = nr.Renderer(camera_mode='projection', P=P, orig_size=IMG_SIZE, image_size=IMG_SIZE,
                      camera_direction=[0,0,1], light_direction=[0,1,0])
images, _, _ = renderer(vertices, faces, textures)  # [batch_size, RGB, image_size, image_size]
image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))  # [image_size, image_size, RGB]