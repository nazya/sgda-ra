# Code for 

## Requirements
- `torch`
- `mlflow`

## Notebooks
To reproduce the results of the paper, run the notebooks.

## Documentation
### Games
Only quadratic game is implementd. If you want to test the different optimizers on your own game, create a class that inherits from `Game`. 
You should implement either the `operator` method, and the `sample` method.

### Optimizers
If you want to test your own optimizer on the existing games, create a class that inherits from `Optimizer`.
You should implement the `step` method that update the parameters of the players.

### Distributed
Distributed versions of the algorithms are implemented using `torch.distributed`. 

## Folder structure
```
- gamesopt:
  - games:
    - quadratic_games.py  # code for the definition of quadratic games.
  - optimizer:
    - distributed.py  
  - train_distributed #to run the distributed experiments.
```
