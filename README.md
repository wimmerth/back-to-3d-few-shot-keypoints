# Back to 3D: Few-Shot 3D Keypoint Detection with Back-Projected 2D Features

### [Thomas Wimmer](https://wimmerth.github.io)<sup>1,2</sup>, [Peter Wonka](https://peterwonka.net/)<sup>3</sup>, [Maks Ovsjanikov](https://www.lix.polytechnique.fr/~maks/)<sup>1</sup>
<sup>1</sup>École Polytechnique, <sup>2</sup>Technical University of Munich, <sup>3</sup>KAUST

Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (**CVPR**), 2024

[**Project Website**](https://wimmerth.github.io/back-to-3d.html) | [PDF](https://arxiv.org/pdf/2311.18113.pdf)


---

**Abstract:**
With the immense growth of dataset sizes and computing resources in recent years, so-called foundation models have
become popular in NLP and vision tasks. In this work, we propose to explore foundation models for the task of keypoint
detection on 3D shapes. A unique characteristic of keypoint detection is that it requires semantic and geometric
awareness while demanding high localization accuracy. To address this problem, we propose, first, to back-project
features from large pre-trained 2D vision models onto 3D shapes and employ them for this task. We show that we obtain
robust 3D features that contain rich semantic information and analyze multiple candidate features stemming from
different 2D foundation models. Second, we employ a keypoint candidate optimization module which aims to match the
average observed distribution of keypoints on the shape and is guided by the back-projected features. The resulting
approach achieves a new state of the art for few-shot keypoint detection on the KeyPointNet dataset, almost doubling the
performance of the previous best methods.

![Keypoint Detection using BT3D](https://wimmerth.github.io/b2-3d/static/images/qualitative_results_5.png)

---

**Installation:**

```bash
conda create -n bt3d python=3.9
conda activate bt3d
conda install pytorch=1.13 torchvision pytorch-cuda=11.6 xformers -c pytorch -c nvidia -c xformers
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d
pip install scipy scikit-learn pandas potpourri3d
```
Setting up the environment is a bit tricky, as always with Python packages.
We found that the above setup works well for us, but you might need to adjust the versions of the packages to match your
system (CUDA etc.).
The installation notes for [PyTorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) and for
[DINOv2](https://github.com/facebookresearch/dinov2?tab=readme-ov-file#installation) might be helpful for this.
If you'd like to use the [SAM](https://github.com/facebookresearch/segment-anything) or the
[CLIP](https://github.com/openai/CLIP) model as feature extractors, follow the instructions in the respective 
repositories to install the additionally required packages.

**Feature Visualization:**

We provide a [simple Jupyter notebook](feature_visualization.ipynb) that you can use to validate that your setup is
working where we demonstrate how you can easily visualize the extracted features using a PCA.

**Experiment:**

To run the experiments on the KeypointNet dataset, you need to
first [download the dataset](https://github.com/qq456cvb/KeypointNet).
Next, set the environment variable `export KEYPOINTNET_DATASET_PATH="</path/to/KeypointNet/dataset>"`.
You should now be able to run the evaluation using `python3 bt3d.py`.

In addition to our method, this repository also includes manual annotated labels of keypoint classes in the KeypointNet
dataset that might be helpful in settings where we want to detect keypoints from pure textual descriptions.
The labels can be found [here](utils/data/keypoint_labels.py).

---

**Citation:**

```bibtex
@inproceedings{wimmer2024back,
    title = {Back to 3D: Few-Shot 3D Keypoint Detection with Back-Projected 2D Features},
    author = {Wimmer, Thomas and Wonka, Peter and Ovsjanikov, Maks},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year = {2024}
}
```

