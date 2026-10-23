# How to Run the Simulation

## Option 1: On your computer

1. Install Python (version 3.8 or newer).
2. Put all the files in one folder: uav_ga.py, sensitivity.py and requirements.txt.
3. Open a terminal in that folder.
4. Install the libraries:

   pip install -r requirements.txt

5. Run the main simulation:

   python uav_ga.py

   It takes about 30 seconds. It prints the results and saves two images: convergence.png and paths_evolution.png.

6. Run the sensitivity test:

   python sensitivity.py

   It takes a few minutes. It prints the results and saves sensitivity.png.

## Option 2: On Google Colab

1. Go to colab.research.google.com and open a new notebook.
2. Click the folder icon on the left and upload uav_ga.py and sensitivity.py.
3. In a new cell, run:

   !python uav_ga.py

4. In another cell, run:

   !python sensitivity.py

5. To see the images, run this in a new cell:

   from IPython.display import Image, display
   display(Image("convergence.png"))
   display(Image("paths_evolution.png"))
   display(Image("sensitivity.png"))

6. Download the images from the folder icon before you close the tab. Colab deletes the files when the session ends.

## What you should see
The main run prints the final fitness, coverage, detection, energy, time and hazard. The random seed is fixed, so you get the same numbers every time you run it.
