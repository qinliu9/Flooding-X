# Flooding-X

Code for the ACL 2022 paper [Flooding-X: Improving BERT's Resistance to Adversarial Attacks via Loss-Restricted Fine-Tuning](https://aclanthology.org/2022.acl-long.386.pdf).

## Requirement
Our code is built on [huggingface/transformers](https://github.com/huggingface/transformers). The detailed requirements are listed in `requirements.txt`.

## Training
You can train a flooded language model configuring and running `train/run_flood_*.sh`.

## Attack
As an adversarial defense method, its effectiveness is to be evaluated by adversarial attacks, which is implemented based on [TextAttack](https://github.com/QData/TextAttack).

## Baselines
We re-implemented baseline methods, which could be found in `train`.

## Citation
If you use this code in your research, please cite our paper:
```bibtex
@inproceedings{liu2022flooding,
 title={Flooding-X: Improving BERT’s Resistance to Adversarial Attacks via Loss-Restricted Fine-Tuning},
 author={Liu, Qin and 
  Zheng, Rui and 
  Rong, Bao and 
  Liu, Jingyi and 
  Liu, Zhihua and 
  Cheng, Zhanzhan and 
  Qiao, Liang and 
  Gui, Tao and 
  Zhang, Qi and 
  Huang, Xuan-Jing},
 booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
 pages={5634--5644},
 url={https://aclanthology.org/2022.acl-long.386.pdf},
 year={2022}
}
```
You may also cite the paper "[Do We Need Zero Training Loss After Achieving Zero Training Error?](http://proceedings.mlr.press/v119/ishida20a.html)" on which our work is based:
```bibtex
@inproceedings{ishida2020we,
 author = {Takashi Ishida and 
  Ikko Yamane and 
  Tomoya Sakai and 
  Gang Niu and 
  Masashi Sugiyama},
 bibsource = {dblp computer science bibliography, https://dblp.org},
 biburl = {https://dblp.org/rec/conf/icml/IshidaYS0S20.bib},
 booktitle = {Proceedings of the 37th International Conference on Machine Learning, {ICML} 2020, 13-18 July 2020, Virtual Event},
 pages = {4604--4614},
 publisher = {{PMLR}},
 series = {Proceedings of Machine Learning Research},
 timestamp = {Tue, 15 Dec 2020 00:00:00 +0100},
 title = {Do We Need Zero Training Loss After Achieving Zero Training Error?},
 url = {http://proceedings.mlr.press/v119/ishida20a.html},
 volume = {119},
 year = {2020}
}
```

