## Files

* `MagR.py`: the implementations functions of MagR
* `modelutils.py`: model utilities
* `datautils.py`: data utilities
* `quant.py`: quantizer
* `optq.py`: the implementations of OPTQ
* `llama.py`

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
