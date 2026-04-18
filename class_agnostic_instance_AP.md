Yes. The cleanest standard choice is:

## Use **class-agnostic mask AP**

That is just **COCO-style instance-segmentation AP**, but with **semantic categories ignored**. In COCOeval / pycocotools this is explicitly supported via `useCats=0`, which means category labels are ignored “as in proposal scoring.” Recent class-agnostic segmentation papers also report **COCO-style AP50 / AP75 / AP** in exactly this spirit. ([GitHub][1])

Formally, treat **all instances as one foreground class** and compute mask AP exactly as usual:

[
\mathrm{AP}^{\text{agn}} = \frac{1}{10}\sum_{\tau \in {0.50,0.55,\dots,0.95}} \mathrm{AP}^{\text{agn}}(\tau)
]

where, at each IoU threshold (\tau), a prediction is a TP if it matches an unmatched GT mask with (\mathrm{IoU}\ge\tau), **ignoring semantic label**, and AP is the area under the precision-recall curve built from the prediction scores. Report `AP`, `AP50`, and `AP75`. ([GitHub][1])

## Why this is usually the right metric

It measures the things you usually care about for instance segmentation quality without letting class prediction contaminate the score:

* missed instances
* duplicate predictions
* false positives
* mask overlap quality
* confidence ranking quality

So if your model finds the right object boundaries but calls a dog a cat, **class-agnostic AP does not care**. ([GitHub][1])

## When this is *not* the best choice

If your method outputs a **single non-overlapping segmentation of the image** and does **not** produce confidence scores, then AP is less natural. In that setting, **PQ** is the more standard metric family:

[
\mathrm{PQ}=\frac{\sum_{(p,g)\in TP}\mathrm{IoU}(p,g)}
{|TP|+\tfrac12|FP|+\tfrac12|FN|}
]

and it decomposes as (\mathrm{PQ}=\mathrm{SQ}\times\mathrm{RQ}), where SQ is average IoU over matched segments and RQ is an F1-like recognition term. ([CVF Open Access][2])

So the practical rule is:

* **scored instance proposals / masks** → use **class-agnostic mask AP**
* **non-overlapping full-image segmentation without scores** → use **class-agnostic PQ**

## Is there a universally named benchmark metric?

Not really. There is **not one universally canonical name** across all papers. People variously call it:

* class-agnostic AP
* mask AP with categories collapsed
* COCO AP with `useCats=0`
* in entity-segmentation papers, variants like **APe**, which are still mAP-style entity metrics in a class-agnostic setting. ([GitHub][1])

## Recommendation

For your stated goal, use:

[
\boxed{\text{class-agnostic COCO mask AP} = \text{AP/AP50/AP75 with all instances treated as one class}}
]

That is the closest thing to a standard, and it is the least arguable choice if you want to judge **instance segmentation quality independent of semantics**. ([GitHub][1])

If you want, I can also show exactly how to implement this in `pycocotools` or Detectron2.

[1]: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py?utm_source=chatgpt.com "cocoapi/PythonAPI/pycocotools/cocoeval.py at master"
[2]: https://openaccess.thecvf.com/content_CVPR_2019/papers/Kirillov_Panoptic_Segmentation_CVPR_2019_paper.pdf "Panoptic Segmentation"
