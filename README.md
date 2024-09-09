## Files

* `modelutils.py`: model utilities
* `datautils.py`: data utilities
* `quant.py`: quantizer
* `optq.py`: the implementations of OPTQ
* `llama.py`
* `MagR.py`: the functions of MagR. There are some hyper-parameters in these function. $\alpha$: 0.001 (per-channel), 0,0001 (per-group); n_iter: 150 (one sample), 200 (128 samples). Theoretically, the more iterations, the better. But in order to balance the number of iterations and running time, 150 and 200 are chosen.

## Install

* pip install --upgrade pip 
* pip install -e .

Or
* `torch`: v2.3.0
* `transformers`: v4.36.0
* `datasets`: v2.18.0


### Run MagR

```
# Quantize LlaMa2-7B for 4 bit
python llama.py meta-llama/llama-2-7b-hf wikitext2 --wbits 4 --magr
```
