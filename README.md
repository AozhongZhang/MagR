## MagR [Neurips 2024]

This repository contains the code for the Neurips 2024 paper MagR: Weight Magnitude Reduction for Enhancing Post-Training Quantization. The current release includes the following features:

* `MagR.py`: the functions of MagR. There are some hyper-parameters in these function. $\alpha$: 0.001 (per-channel), 0,0001 (per-group); n_iter: 200 (128 samples). Theoretically, the more iterations, the better. But in order to balance the number of iterations and running time, 200 are chosen.
* `modelutils.py`: model utilities
* `datautils.py`: data utilities
* `quant.py`: quantizer
* `optq.py`: the implementations of OPTQ


## Install

* pip install --upgrade pip 
* pip install -e .

Or
* `torch`: v2.3.0
* `transformers`: v4.36.0
* `datasets`: v2.18.0


## Run MagR

```
# Quantize LlaMa2-7B for 4 bit
python llama.py meta-llama/llama-2-7b-hf wikitext2 --wbits 4 --magr

# Further improve LlaMa2-7B for 4 bit
python llama.py meta-llama/llama-2-7b-hf wikitext2 --wbits 4 --magr --CD_update
```

## Citation

If you found this work useful, please consider citing:
```
@article{zhang2024magr,
  title={MagR: Weight Magnitude Reduction for Enhancing Post-Training Quantization},
  author={Zhang, Aozhong and Wang, Naigang and Deng, Yanxia and Li, Xin and Yang, Zi and Yin, Penghang},
  journal={arXiv preprint arXiv:2406.00800},
  year={2024}
}
```
