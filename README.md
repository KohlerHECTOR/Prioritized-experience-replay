# Implementation of the work of Mattar's [Prioritized memory access explains planning and hippocampal replay](https://www.nature.com/articles/s41593-018-0232-z) (Work in Progress)

## Install

- install a simple maze mdp simulator in your working environment (all credit to osigaud and araffin): 

```
git clone https://github.com/osigaud/SimpleMazeMDP.git
cd SimpleMazeMDP
python3 setup.py install
cd
```

- clone this repo wherever you want in your working enivornment:

```
git clone https://github.com/KohlerHECTOR/Prioritized_experience_replay.git
```

- go to the project to start experimenting:

```
cd Prioritized_experience_replay
```

## Reproduce [Sutton & Barto's Introduction to RL figure 8.5 (2nd edition 2014, 2015)](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf) (roughly 5 minutes run time)

```
python3 sutton_Barto_fig_8_5.py
```
The figure is saved in results/Sutton&Barto/

## Run Mattar's default model (dyna Q coupled with smart backup procedure) (roughly 5 minutes run time)

```
python3 main_test.py --simulations 2 --episodes 50
```

A figure of performances of the agent is saved in results/Mattar/
