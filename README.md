# Implementation of the work of Mattar's [Prioritized memory access explains planning and hippocampal replay](https://www.nature.com/articles/s41593-018-0232-z) (Work in Progress)
## This repo makes use of [osigaud](https://github.com/osigaud) and [araffin](https://github.com/araffin) 's [simple maze MDP simulator](https://github.com/osigaud/SimpleMazeMDP)
## Install requirements

```
pip3 install -r requirements.txt
```


## Reproduce [Sutton & Barto's Introduction to RL figure 8.5 (2nd edition 2014, 2015)](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf) (roughly 5 minutes run time)

```
python3 sutton_Barto_fig_8_5.py
```
The figure is saved in results/Sutton&Barto/

## Run Mattar's default model (dyna Q coupled with smart backup procedure) (roughly 5 minutes run time)

```
python3 main.py --simulations 10 --episodes 30
```

A figure of performances of the agent is saved in results/
