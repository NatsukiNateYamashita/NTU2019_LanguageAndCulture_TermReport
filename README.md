# Language_and_Culture_Term_Report

This repository is made for term report in Language and Culture class in NTU.
These codes are written only for experiments so they are not clean code.
Some of valiables are needed to modify by hand in each run of programs.
Please do not hesitate to contact to me or ask me, when you have any question.

## How to run


Download MPDD http://nlg.csie.ntu.edu.tw/nlpresource/MPDD/

```
pip install -r requirements.txt
python3 preprocess.py
python3 classify_position.py
python3 classify_relation.py
python3 analize.py
python3 get_attentions.py
```
Note: please modify the code or parameters by yourself when you get errors.

```
cd bertviz
jupyter notebook
```

On jupyter notebook, modify code and set the parameters or a file in which you want to visualize attentions by hand
