# TextGAIL: Generative Adversarial Imitation Learning for Text Generation

This is the repository for the paper: TextGAIL: [Generative Adversarial Imitation Learning for Text Generation](https://arxiv.org/abs/2004.13796)

## Requirements

TorchFly (https://github.com/qywu/TorchFly) is needed for all dependencies. It is included in this repository.

## Datasets

To run the code, You need to download the following datasets, and store them under the folder `data`.

The datasets used in the paper can be found below:
- **COCO**: https://github.com/geek-ai/Texygen/tree/master/data

- **EMNLP2017 NEWS**: https://github.com/geek-ai/Texygen/tree/master/data

- **ROCStories**: https://cs.rochester.edu/nlp/rocstories/

- **CommonGEN**: https://github.com/INK-USC/CommonGen

Then run the pre-processing script to get `train.jsonl`, `valid.jsonl`, `test.jsonl` for each dataset.
The details can be found in the `data` folder.

## Training

### Pre-trained Weights

We use GPT-2 small (117M) and RoBERTa base (117M) as the pre-trained model weights, which are automatically downloaded and loaded.

### Scripts

To obtain the best MLE models, please run scripts in the [`Conditional/MLE`](https://github.com/qywu/TextGAIL/tree/master/Conditional/MLE) folder or [`Conditional/MLE`](https://github.com/qywu/TextGAIL/tree/master/Unconditional/MLE) first.

```bash
cd Conditional/MLE
#  choose the task.name in {CommonGEN, DailyDialog}
python main.py --config config/config.yaml task.name=CommonGEN
```

To train TextGAIL, please specify the MLE model after warm-up training. Then simply run the following command.

```bash
python main.py --config config/config.yaml task.name=CommonGEN task.weights_path="../../../MLE/outputs/CommonGEN/Checkpoints/iter_252_model_state.pth"
```

All hyper-paramters are stored in YAML format and will be automatically loaded by the script. You can adjust them according to your task.

## Evaluation

Please run `Generation.ipynb` to have the geenerated outputs before the evaluation.

Conditional generation tasks and unconditional generation tasks have different evaluation metrics.
Therefore, please refer to the `Evaluation` folder for details.

We have provided generated outputs along with the generated outputs for evaluation.


## Citation

```
@inproceedings{wu2021textgail,
  author    = {Qingyang Wu and
               Lei Li and
               Zhou Yu},
  title     = {TextGAIL: Generative Adversarial Imitation Learning for Text Generation},
  booktitle = {The Thirty-Fifth {AAAI} Conference on Artificial Intelligence, {AAAI} 2021},
  pages     = {online},
  publisher = {{AAAI} Press},
  year      = {2021},
}
```

## Contact

Feel free to contact me (wilwu@ucdavis.edu) if you have any questions.