import os

import numpy as np
import open3d as o3d
import partio
from scipy.spatial import cKDTree


def write_bgeo_from_numpy(outpath, pos_arr, vel_arr):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    n = pos_arr.shape[0]
    if not (vel_arr.shape[0] == n and pos_arr.shape[1] == 3 and
            vel_arr.shape[1] == 3):
        raise ValueError(
            "invalid shapes for pos_arr {} and/or vel_arr {}".format(
                pos_arr.shape, vel_arr.shape))

    p = partio.create()
    position_attr = p.addAttribute("position", partio.VECTOR, 3)
    velocity_attr = p.addAttribute("velocity", partio.VECTOR, 3)

    for i in range(n):
        idx = p.addParticle()
        p.set(position_attr, idx, pos_arr[i].astype(float))
        p.set(velocity_attr, idx, vel_arr[i].astype(float))

    partio.write(outpath, p)


def crop_particles_box(npz_folder, box_min, box_max, timestep):
    # fetch all npz files
    npz_files = [f for f in os.listdir(npz_folder) if f.endswith('.npz')]
    # stack all npz files
    all_particles = []
    for npz_file in npz_files:
        data = np.load(os.path.join(npz_folder, npz_file))
        all_particles.append(data['pos'])  # [(N,3), (N,3), ...]
    # stack all particles to (B,N,3)
    all_particles = np.stack(all_particles, axis=0)  # (B,N,3)

    # get mask
    mask = np.all(np.logical_and(all_particles >= box_min, all_particles <= box_max), axis=-1)  # (B,N)
    mask = np.all(mask, axis=0)  # (N,)

    # crop particles
    pos = all_particles[:, mask]  # (B,N',3)
    vel = (pos - np.roll(pos, shift=1, axis=0)) / timestep  # (B,N',3)
    vel[0] = 0  # first frame velocity is zero

    # save to npz
    out_folder = os.path.join(npz_folder, 'cropped')
    os.makedirs(out_folder, exist_ok=True)
    for i in range(all_particles.shape[0]):
        # filename: 0000.npz, 0001.npz, ...
        np.savez(os.path.join(out_folder, f'{i:04d}.npz'), pos=pos[i], vel=vel[i])

    #  save to bgeo
    out_folder = os.path.join(npz_folder, 'cropped', 'bgeo')
    os.makedirs(out_folder, exist_ok=True)

    for i in range(all_particles.shape[0]):
        # filename: 0000.bgeo, 0001.bgeo, ...
        write_bgeo_from_numpy(os.path.join(out_folder, f'{i:04d}.bgeo'), pos[i], vel[i])


def fix_overlap(bgeo_path, out_path, min_dist=0.005, jitter_strength=0.001):
    p = partio.read(bgeo_path)
    n = p.numParticles()

    pos_attr = p.attributeInfo("position")
    vel_attr = p.attributeInfo("velocity")

    positions = np.array([p.get(pos_attr, i) for i in range(n)])
    velocities = np.array([p.get(vel_attr, i) for i in range(n)])

    # 用KD树查找近邻
    tree = cKDTree(positions)
    to_keep = np.ones(n, dtype=bool)

    for i in range(n):
        if not to_keep[i]:
            continue
        neighbors = tree.query_ball_point(positions[i], r=min_dist)
        for j in neighbors:
            if j <= i:
                continue
            # 标记冗余粒子不保留
            to_keep[j] = False

    # 过滤粒子
    positions = positions[to_keep]
    velocities = velocities[to_keep]

    write_bgeo_from_numpy(out_path, positions, velocities)


def write_particles(path_without_ext, pos, vel=None, write_ply=None):
    """Writes the particles as point cloud ply.
    Optionally writes particles as bgeo which also supports velocities.
    """
    # pos = rot_x_90(pos)
    arrs = {'pos': pos}
    if not vel is None:
        arrs['vel'] = vel

    # arrs['vel'] = rot_x_90(arrs['vel'])

    np.savez(path_without_ext + '.npz', **arrs)
    if write_ply:
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pos))
        o3d.io.write_point_cloud(path_without_ext + '.ply', pcd)


def numpy_from_bgeo(path):
    import partio
    p = partio.read(path)
    pos = p.attributeInfo('position')
    vel = p.attributeInfo('velocity')
    ida = p.attributeInfo('trackid')  # old format
    if ida is None:
        ida = p.attributeInfo('id')  # new format after splishsplash update
    n = p.numParticles()
    pos_arr = np.empty((n, pos.count))
    for i in range(n):
        pos_arr[i] = p.get(pos, i)

    vel_arr = None
    if not vel is None:
        vel_arr = np.empty((n, vel.count))
        for i in range(n):
            vel_arr[i] = p.get(vel, i)

    if not ida is None:
        id_arr = np.empty((n,), dtype=np.int64)
        for i in range(n):
            id_arr[i] = p.get(ida, i)[0]

        s = np.argsort(id_arr)
        result = [pos_arr[s]]
        if not vel is None:
            result.append(vel_arr[s])
    else:
        result = [pos_arr, vel_arr]

    return tuple(result)


def bgeo_to_npz(bgeo_folder, out_path):
    # fetch all bgeo files
    bgeo_files = [f for f in os.listdir(bgeo_folder) if f.endswith('.bgeo')]
    #  name : "ParticleData_fluid{0}_{1}.bgeo" .format(fluid_id, frame_id)

    for bgeo_file in bgeo_files:
        bgeo_path = os.path.join(bgeo_folder, bgeo_file)
        pos_, vel_ = numpy_from_bgeo(bgeo_path)

        # get fluid_id and frame_id from bgeo_file
        fluid_id, frame_id = bgeo_file.split('.')[0].split('_')[1:]
        fluid_output_path = os.path.join(out_path, 'fluid_{0:04d}'.format(int(frame_id)))
        write_particles(fluid_output_path, pos_, vel_, write_ply=True)

def crop_particles_manual(bgeo_path, out_path):
    p = partio.read(bgeo_path)
    n = p.numParticles()

    pos_attr = p.attributeInfo("position")
    vel_attr = p.attributeInfo("velocity")

    positions = np.array([p.get(pos_attr, i) for i in range(n)])
    velocities = np.array([p.get(vel_attr, i) for i in range(n)])

    manaul_mask =( positions[:, 0]+positions[:,1] <= 0.2)

    # 过滤粒子
    positions = positions[manaul_mask]
    velocities = velocities[manaul_mask]

    write_bgeo_from_numpy(out_path, positions, velocities)


if __name__ == '__main__':
    # npz_folder = '/workspace/gaussfluids/data/model/continuation_watersphere_train/output/npz'
    # box_min = np.array([-1, -1, -1])
    # box_max = np.array([1, 1, 10])
    # timestep = 1 / 60
    # crop_particles_box(npz_folder, box_min, box_max, timestep)

    # bgeo_path = '/workspace/gaussfluids/data/model/continuation_watersphere_train/output/npz/cropped/bgeo/0020.bgeo'
    # out_path = '/workspace/gaussfluids/data/model/continuation_watersphere_train/output/npz/cropped/bgeo/0020_fixed.bgeo'
    # fix_overlap(bgeo_path, out_path, min_dist=0.005, jitter_strength=0.001)

    bgeo_folder = '/workspace/gaussfluids/data/model/fluid_editing/partio/'
    out_path = '/workspace/gaussfluids/data/model/fluid_editing/partio/npz/'
    os.makedirs(out_path, exist_ok=True)
    bgeo_to_npz(bgeo_folder, out_path)

    # bgeo_path = '/workspace/gaussfluids/data/model/continuation_watersphere_train/output/npz/cropped/bgeo/0015.bgeo'
    # out_path = '/workspace/gaussfluids/data/model/continuation_watersphere_train/output/npz/cropped/bgeo/0015_crop.bgeo'
    # crop_particles_manual(bgeo_path, out_path)
