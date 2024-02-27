# splitter-cells: source code and data file for running the 8-maze navigation task and proceed to the reservoir state analysis 


This repository contains the necessary source code to use run the simulation of the bot executing a navigation task.

### Dependencies

[reservoirpy](https://reservoirpy.readthedocs.io/en/latest/index.html)



### Run the navigation task

Main script to run:

```Bash
python main.py
```

Before running the script, certain configurations are required to be set at the begginning of the : ```main.py``` script:


- ```task``` :   
        1) 'R-L' (alternation task)    
        2) 'RR-LL' (half-alternation task)   

- ```simulation_mode```:   
                1) ```'walls'```: the bot navigates and takes direction automatically using Braitenberg algorithms.    
                            Walls are added to guide the bot in the right direction.
                   Some walls are added so as to force the bot taking the right direction.        
                2) ```'data'```: the bot is data-driven and navigates based on the provided position file.    
                3) ```'esn'```: the bot moves based on ESN predictions, trained using supervised learning.

- ```save_reservoir_states```: set to True if the reservoir states and the bot's positions and orientation need to be recorded.   
- ```save_bot_states```: set to True if the bot's positions and orientation need to be recorded.  
- ```path_to_save```: folder to save.  



### Run the analysis


#### Single-cell analysis

```Bash
python analysis/single_cell_analysis.py
```

#### Population-level analysis

```Bash
python analysis/population_analysis.py
```