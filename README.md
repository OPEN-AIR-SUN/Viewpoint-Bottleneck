# Pointly-supervised 3D Scene Parsing with Viewpoint Bottleneck

[**Paper**](https://arxiv.org/abs/2109.08553)

Created by Liyi Luo, Beiwen Tian, Hao Zhao and Guyue Zhou from [Institute for AI Industry Research (AIR), Tsinghua University, China](http://air.tsinghua.edu.cn/EN/).

---



<img src=".\doc\result7.png" style="zoom:70%;" />

<img src=".\doc\result1.png" style="zoom:70%;" />

<img src=".\doc\result2.png" alt="result2" style="zoom:70%;" />

<img src=".\doc\result3.png" alt="result3" style="zoom:70%;" />

<img src=".\doc\result4.png" alt="result4" style="zoom:70%;" />

<img src=".\doc\result5.png" alt="result5" style="zoom:70%;" />

<img src=".\doc\result6.png" alt="result6" style="zoom:70%;" />

## Introduction

Semantic understanding of 3D point clouds is important for various robotics applications. Given that point-wise semantic annotation is expensive, in our paper, we address the challenge of learning models with extremely sparse labels. The core problem is how to leverage numerous unlabeled points. 

In this repository, we propose a self-supervised 3D representation learning framework named viewpoint bottleneck. It optimizes a mutual-information based objective, which is applied on point clouds under different viewpoints. A principled analysis shows that viewpoint bottleneck leads to an elegant surrogate loss function that is suitable for large-scale point cloud data. Compared with former arts based upon contrastive learning, viewpoint bottleneck operates on the feature dimension instead of the sample dimension. This paradigm shift has several advantages: It is easy to implement and tune, does not need negative samples and performs better on our goal down-streaming task. We evaluate our method on the public benchmark ScanNet, under the pointly-supervised setting. We achieve the best quantitative results among comparable solutions. Meanwhile we provide an extensive qualitative inspection on various challenging scenes. They demonstrate that our models can produce fairly good scene parsing results for robotics applications. 

## Citation

If you find our work useful in your research, please consider citing:

```
@misc{luo2021pointlysupervised,
      title={Pointly-supervised 3D Scene Parsing with Viewpoint Bottleneck}, 
      author={Liyi Luo and Beiwen Tian and Hao Zhao and Guyue Zhou},
      year={2021},
      eprint={2109.08553},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Preparation

### Requirements

* Python 3.6 or higher
* CUDA 11.1 

It is strongly recommended to proceed in a **virtual environment** (`venv`, `conda`)

### Installation

Clone the repository and install the rest of the requirements

```shell
git clone https://github.com/OPEN-AIR-SUN/ViewpointBottleneck/
cd ViewpointBottlencek

# Uncomment following commands to create & activate a conda env
# conda create -n env_name python==3.8
# conda activate env_name

pip install -r requirements.txt
```

## Data Preprocess

1. Download [ScanNetV2](https://github.com/ScanNet/ScanNet) dataset and data-efficient setting [HERE](http://kaldir.vc.in.tum.de/scannet_benchmark/data_efficient/documentation) .

2. Extract point clouds and annotations by running 

```shell
# From root of the repo
# Fully-supervised:
python data_preprocess/scannet.py

# Pointly supervised:
python data_preprocess/scannet_eff.py
```

## Pretrain the model

```shell
# From root of the repo
cd pretrain/
chmod +x ./run.sh
./run.sh
```

You can modify some details with environment variables:

```shell
SHOTS=50 FEATURE_DIM=512 \
LOG_DIR=logs \
PRETRAIN_PATH=actual/path/to/pretrain.pth \
DATASET_PATH=actual/directory/of/dataset \
./run.sh

```

## Fine-tune the model with pretrained checkpoint

```shell
# From root of the repo
cd finetune/
chmod +x ./run.sh
./run.sh
```

You can modify some details with environment variables:

```shell
SHOTS=50 \
LOG_DIR=logs \
PRETRAIN_PATH=actual/path/to/pretrain.pth \
DATASET_PATH=actual/directory/of/dataset \
./run.sh

```

## Model Zoo


<table>
    <tr>
        <th align="center" scope="row" width="30%">Pretrained Checkpoints</th>
        <th align="center" scope="row" colspan="2" width="20%"> Feature Dimension</th>
        <td align="center">
            <a href="https://drive.google.com/file/d/1n-0uOe2J8M6-VTKVImpBBlczXDtZywPT/view?usp=sharing">
                256
            </a>
        </td>
        <td align="center">
            <a href="https://drive.google.com/file/d/1oRIHlEu1fS2eKpaIyi1J7BCIkKSn174k/view?usp=sharing">
                512
            </a>
        </td>
        <td align="center">
            <a href="https://drive.google.com/file/d/1rKxLmAXfhwZF-y7hp-yzNAFPqSWWWwjh/view?usp=sharing">
                1024
            </a>
        </td>
    </tr>
    <th align="center" scope="row" rowspan="5">Final checkpoints <br/>  mIOU(%) on val split </th>
    <th align="center" scope="row" rowspan="5">Supervised points</th>
    <tr>
        <th align="center" scope="row">20</td>
        <td align="center">
            <a href="https://drive.google.com/file/d/1WR7VFXe1mmn6Y42ddgQ3iaSCEaxgEt44/view?usp=sharing">
                56.2
            </a>
        </td>
        <td align="center">
            <a href="https://drive.google.com/file/d/10vp9iEUkm5i4NNvClQOxK3jv3y7-Uxbr/view?usp=sharing">
                57.0
            </a>
        </td>
        <td align="center">
            <a href="https://drive.google.com/file/d/1PpJKH7nD2q-yKf1NVFaUtZGz5ulH1fE1/view?usp=sharing">
                56.3
            </a>
        </td>
    </tr>
    <tr>
        <th align="center" scope="row">50</td>
        <td align="center">
            <a href="https://drive.google.com/file/d/10wRRXHuLEx8ktbSYrlpDhWLReUoDLKm2/view?usp=sharing">
                63.3
            </a>
        </td>
        <td align="center">
            <a href="https://drive.google.com/file/d/1SCVZ353m32mUiIHlM2JZqMeXETqHHj0M/view?usp=sharing">
                63.6
            </a>
        </td>
        <td align="center">
            <a href="https://drive.google.com/file/d/1kNNwpckKMt3G6nelCzS2-Y1l0-NDEw7c/view?usp=sharing">
                63.7
            </a>
        </td>
    </tr>
    <tr>
        <th align="center" scope="row">100</td>
        <td align="center">
            <a href="https://drive.google.com/file/d/1hDofcM0nBgU2PnEliPaZlDTcVIK8fLKn/view?usp=sharing">
                66.5
            </a>
        </td>
        <td align="center">
            <a href="https://drive.google.com/file/d/17LsDAaj6S2g7SsduPv1HOVR5IMHjVknk/view?usp=sharing">
                66.8
            </a>
        </td>
        <td align="center">
            <a href="https://drive.google.com/file/d/1QNPITNceb1JNjQgBX2dIFPWG4HXO-oWk/view?usp=sharing">
                66.5
            </a>
        </td>
    </tr>
    <tr>
        <th align="center" scope="row">200</td>
        <td align="center">
            <a href="https://drive.google.com/file/d/1cd0MVEElxCuwBWDuWRD3C_TIUQCIE-Mg/view?usp=sharing">
                68.4
            </a>
        </td>
        <td align="center">
            <a href="https://drive.google.com/file/d/1wZDmDc7feZFbL6_Ape1Upu9rzAOoDP29/view?usp=sharing">
                68.5
            </a>
        </td>
        <td align="center">
            <a href="https://drive.google.com/file/d/1sP4ci-_3d2y8BAUSlUGZWsTM5GX2t9I6/view?usp=sharing">
                68.4
            </a>
        </td>
    </tr>
</table>

## Acknowledgements

We appreciate the work of  [ScanNet](https://github.com/ScanNet/ScanNet) and [SpatioTemporalSegmentation](https://github.com/chrischoy/SpatioTemporalSegmentation).

We are grateful to Anker Innovations for supporting this project.

