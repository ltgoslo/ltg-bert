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
  <a href="https://arxiv.org/abs/2303.09859"><b>Paper</b></a><br>
  <a href="https://huggingface.co/ltg/bnc-bert-span"><b>HuggingFace models</b></a>
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

## Please cite the following publication (just arXiv for now)
```bibtex
@inproceedings{samuel-etal-2023-trained,
    title = "Trained on 100 million words and still in shape: {BERT} meets {B}ritish {N}ational {C}orpus",
    author = "Samuel, David  and
      Kutuzov, Andrey  and
      {\O}vrelid, Lilja  and
      Velldal, Erik",
    booktitle = "Findings of the Association for Computational Linguistics: EACL 2023",
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-eacl.146",
    pages = "1954--1974",
    abstract = "While modern masked language models (LMs) are trained on ever larger corpora, we here explore the effects of down-scaling training to a modestly-sized but representative, well-balanced, and publicly available English text source {--} the British National Corpus. We show that pre-training on this carefully curated corpus can reach better performance than the original BERT model. We argue that this type of corpora has great potential as a language modeling benchmark. To showcase this potential, we present fair, reproducible and data-efficient comparative studies of LMs, in which we evaluate several training objectives and model architectures and replicate previous empirical results in a systematic way. We propose an optimized LM architecture called LTG-BERT.",
}
```
