Please follow the below convention to name config files for our **AMOD** dataset:
~~~
{model}_[model setting]_{backbone}_{neck}_angle{used_look_angles}_{schedule}_[misc]_amod.py
~~~
{xxx} is a required field and [yyy] is optional.

* `{model}`: model type like faster_rcnn, mask_rcnn, etc.
* `[model setting]`: specific setting for some model, like **_pretrained on AMOD and then fine-tuned for other datasets_**.
* `{backbone}`: backbone type like r50 (ResNet-50), x101 (ResNeXt-101).
* `{neck}`: neck type like fpn, pafpn, nasfpn, c4.
* `{used_look_angles}`: used look angles in AMOD dataset (e.g. angle0 or angle0,10 or angle0,10,20,30,40,50)
* `{schedule}`: training schedule, e.g. 30epochs.
* `[misc]`: miscellaneous setting/plugins of model, e.g. le90 (angle version), dconv, gcb, attention, albu, mstrain.
