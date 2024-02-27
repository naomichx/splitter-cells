# splitter-cells: source code and data file for running the 8-maze navigation task and proceed to the reservoir state analysis 


This repository provides the source code to simulate a bot executing an 8-Maze navigation task using Echo State Network (ESN) predictions. During navigation, it is possible to analyze the internal neurons of the model, particularly focusing on hippocampal cells such as place cells, head-direction cells, and splitter cells.


### Dependencies

-  *numpy*, *scipy*, *sklearn*, *matplotlib*
- *umap*
- *optuna* 
- [reservoirpy](https://reservoirpy.readthedocs.io/en/latest/index.html)


This repository allows to:
- Run the simulation in different configurations (data-driven, esn-driven, braitenberg-driven), and record the reservoir states during the navigation task  (supporting two types of navigation tasks).
- Analyze reservoir activity using two methods: single-cell analysis and population-level analysis.

### Run the navigation task

To run the navigation task, execute the main script:

```Bash
python main.py
```

Before running the script, ensure to set certain configurations at the beginning of ```main.py```:

- ```task``` :   
        1) 'R-L' (alternation task)    
        2) 'RR-LL' (half-alternation task)   

- ```cues```:   set to True if the model includes contextual cues as input.

- ```simulation_mode```:   
                1) ```'walls'```: the bot navigates and takes direction automatically using Braitenberg algorithms.    
                            Walls are added to guide the bot in the right direction.
                   Some walls are added so as to force the bot taking the right direction.        
                2) ```'data'```: the bot is data-driven and navigates based on the provided position file.    
                3) ```'esn'```: the bot moves based on ESN predictions, trained using supervised learning.

- ```save_reservoir_states```: set to True if the reservoir states and the bot's positions and orientation need to be recorded.   
- ```save_bot_states```: set to True if the bot's positions and orientation need to be recorded.  
- ```path_to_save```: folder to save.  



**Comments**: the sensors of the bot vary in size depending on the task. Specifically, for the 'RR-LL' task, the sensors are larger compared to the 'R-L' task. 

### Run the analysis


#### Single-cell analysis
To conduct single-cell analysis, use the following command:

```Bash
python analysis/single_cell_analysis.py
```

#### Population-level analysis
For population-level analysis, run:
```Bash
python analysis/population_analysis.py
```
