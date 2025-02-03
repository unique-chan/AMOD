Please follow the below convention to name config files for our **AMOD** dataset:
~~~
{model}_[model setting]_{backbone}_{neck}_{used_look_angles}_{schedule}_[misc]_amod.py
~~~
{xxx} is required field and [yyy] is optional.

* `{model}`: model type like faster_rcnn, mask_rcnn, etc.
* `[model setting]`: specific setting for some model, like **_pretrained on AMOD and then fine-tuned for other datasets_**.
* `{backbone}`: backbone type like r50 (ResNet-50), x101 (ResNeXt-101).
* `{neck}`: neck type like fpn, pafpn, nasfpn, c4.
* `{schedule}`: training schedule, e.g. 30epochs.
* `[misc]`: miscellaneous setting/plugins of model, e.g. le90 (angle version), dconv, gcb, attention, albu, mstrain.
* `{dataset}`: dataset like coco, cityscapes, voc_0712, wider_face.