from pathlib import Path
import torch
import glob
import numpy as np
from lib.pc_utils import read_plyfile, save_point_cloud
from concurrent.futures import ProcessPoolExecutor
SCANNET_DATA_RAW_PATH = Path('./scan_dataset/')
SCANNET_LABEL_RAW_PATH = Path('./scan_label/')
SCANNET_OUT_PATH = Path('./dataset/eff_20/')
TRAIN_DEST = 'train'
TEST_DEST = 'test'
SUBSETS = {TRAIN_DEST: 'scans', TEST_DEST: 'scans_test'}
POINTCLOUD_FILE = '_vh_clean_2.ply'
Eff
BUGS = {
    'train/scene0270_00.ply': 50,
    'train/scene0270_02.ply': 50,
    'train/scene0384_00.ply': 149,
}
print('start preprocess')
# Preprocess data.
t = torch.load('points20')
key = sorted(t.keys())

def handle_process(path):
  f = Path(path.split(',')[0])
  phase_out_path = Path(path.split(',')[1])
  pointcloud = read_plyfile(f)
  # Make sure alpha value is meaningless.
  assert np.unique(pointcloud[:, -1]).size == 1
  # Load label file.
  label_f = f.parent.parent.parent.parent / ('scan_label') / (f.parent.parent.stem + '/' + f.parent.stem+ '/' + f.stem + '.labels' + f.suffix)
  if label_f.is_file():
    label = read_plyfile(label_f)
    # Sanity check that the pointcloud and its label has same vertices.
    assert pointcloud.shape[0] == label.shape[0]
    assert np.allclose(pointcloud[:, :3], label[:, :3])
  else:  # Label may not exist in test case.
    label = np.zeros_like(pointcloud)
  out_f = phase_out_path / (f.name[:-len(POINTCLOUD_FILE)] + f.suffix)
  w = np.array([label[:, -1]]).T
  k = np.ones(len(w))*(255)
  label_str = str(f)
  print(label_str[-27:-15])
  if label_str[-27:-15] in key:
    for c in range(len(w)):
      if c in t[label_str[-27:-15]]:
        k[c] = w[c]
      else: 
        k[c] = 255                  
  label[:, -1] = k
  processed = np.hstack((pointcloud[:, :6], np.array([label[:, -1]]).T))
  save_point_cloud(processed, out_f, with_label=True, verbose=False)


path_list = []
for out_path, in_path in SUBSETS.items():
  phase_out_path = SCANNET_OUT_PATH / out_path
  phase_out_path.mkdir(parents=True, exist_ok=True)
  for f in (SCANNET_DATA_RAW_PATH / in_path).glob('*/*' + POINTCLOUD_FILE):
    path_list.append(str(f) + ',' + str(phase_out_path))
pool = ProcessPoolExecutor(max_workers=20)
result = list(pool.map(handle_process, path_list))

# Fix bug in the data.
for files, bug_index in BUGS.items():
  print(files)

  for f in SCANNET_OUT_PATH.glob(files):
    pointcloud = read_plyfile(f)
    bug_mask = pointcloud[:, -1] == bug_index
    print(f'Fixing {f} bugged label {bug_index} x {bug_mask.sum()}')
    pointcloud[bug_mask, -1] = 0
    save_point_cloud(pointcloud, f, with_label=True, verbose=False)
