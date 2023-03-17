<h2 align="center"><b><h3>Trained on 100 million words and still in shape:</h3><h3>BERT meets British National Corpus</h3></b></h2><br>


<p align="center">
  <b>David Samuel, Andrey Kutuzov, Lilja Øvrelid and Erik Velldal</b>
</p>

<p align="center">
  <i>
    University of Oslo<br>
    Language Technology Group<br>
  </i>
</p>
<br>

<p align="center">
  <a href=""><b>Paper (TBA)</b></a><br>
  <a href=""><b>HuggingFace models (TBA)</b></a>
</p>

<!--
<p align="center">
  <img src="img/overall.png" alt="Illustration of our model." width="720"/>  
</p>
-->
_______

<br>

This is the official repository for our EACL paper about pre-training language models on a representative 100M-word corpus. We propose a data-efficient LM architecture (LTG-BERT) that outperforms the original BERT model. We believe that this type of modestly-sized, but representative, corpora has great potential as a language modeling benchmark.

_______

<br>

## Content of this repository

- `./modeling_ltgbert.py`: HuggingFace-compatible wrapper for LTG-BERT
- `./preprocessing/`: Scripts for processing the XML format of BNC and for processing the evaluation datasets
- `./training/`: Scripts for training LTG-BERT on processed BNC
- `./evaluation/`: Evaluation scripts for evaluation LTG-BERT on (Super)GLUE, edge probing and BLiMP

_______

<br>

## Please cite the following publication
```bibtex
TBA
```
