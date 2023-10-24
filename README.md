# Code for [Byzantine-Tolerant Methods for Distributed Variational Inequalities](https://openreview.net/forum?id=ER0bcYXvvo)

## Requirements
```setup
conda env create -f environment.yml
```

## Notebooks
To reproduce the results of the paper, run the train notebooks, then run corresponding plot notebooks.

## Documentation
### Games
Only quadratic game is implemented. If you want to test the optimizers on your own game, create a class that inherits from `Game`. 
You should implement either the `operator` method, and the `sample` method.

### Optimizers
If you want to test your own optimizer on the existing games, create a class that inherits from `Optimizer`.
You should implement the `step` method that update the parameters of the players.

### Distributed
Distributed versions of the algorithms are implemented using `torch.distributed`. Single thread implemetatoin is not tested properly.

## Folder structure
```
- gamesopt:
  - games:
    - quadratic_games.py  # code for the definition of quadratic games.
  - optimizer:
    - distributed.py  
```
