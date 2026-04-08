For the **original π³ paper/model**, the point cloud is **not metric by default**. The paper explicitly says it predicts **scale-invariant local point maps**, and the camera poses are only defined up to a **similarity transform with one unknown global scale factor**. During training, they solve for an optimal scalar (s^*) to align predictions to ground truth, which is a strong signal that the raw output does **not** come with absolute metric scale baked in. ([arXiv][1])

So in practice, the output geometry is:

* **correct up to one global scale per scene/sequence**
* **internally consistent** across views
* but **not guaranteed to be in meters**. ([arXiv][1])

That means the **depth values and point cloud coordinates are not metric** unless you recover scale externally. If you need meters, then yes, you should do a scale calibration using some real-world cue, such as one depth reading, a known camera baseline/pose, a known object size, LiDAR, or another sensor. Since the ambiguity is a **single global scale**, one good metric cue is enough to rescale the whole reconstructed scene. ([arXiv][1])

One more detail: in the repo, the released code also describes π³ as producing **scale-invariant local point maps**. The repo’s newer **Pi3X** update says it adds **approximate metric scale reconstruction**, which implies the original π³ should still be treated as non-metric. ([GitHub][2])

So the practical answer is:

* **π³ paper/model**: not metric, needs external scale if you want meters. ([arXiv][1])
* **Pi3X engineering update**: can provide **approximate** metric scale, but that is beyond the original paper model. ([GitHub][2])

If you want, I can show exactly how to compute that single scale factor from one RGB-D frame or one known camera translation and apply it to the whole π³ point cloud.

[1]: https://arxiv.org/pdf/2507.13347 "$\pi^3$: Permutation-Equivariant Visual Geometry Learning"
[2]: https://github.com/yyfz/Pi3 "GitHub - yyfz/Pi3: [ICLR 2026] π^3: Permutation-Equivariant Visual Geometry Learning · GitHub"
