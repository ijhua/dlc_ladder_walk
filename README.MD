The problem:
-	A rat walks across a horizontal ladder with unevenly spaced rungs. So, there are places with rungs and spaces without rungs in a somewhat random order.
-	It can go left to right or right to left. Sometimes, it will stand up.
-	We want to know how many times the rat correctly places its paw (hit) on the rung and how many times it misses it (miss). From that, we also want the total (steps)
-	Only count forward progress. Not if the paw is picked up then put down again. The recovery after a slip is also not a step.

The data:
-	I used Deeplabcut to train ResNet-101 to recognize 16 points on the rat and analyze my videos
-	Output: .h5 files multi-index array with (x,y) coordinates of each point that I tracked and its likelihood 

Main code:
-	functions.py: define all the functions that are used later.
-	mplimbscore.py: Input the .h5 for each video and output counts of hit, miss, and step.
-	multipoint_comparison.py: Inputs .h5 for each video, counts the number of hit, miss, and step and outputs graphs and csv of comparisons. (Differences in count and average differences over time.)

Methods in code
-	for simplicity (and due to poor tracking on the far side of the animal) we are only scoring the two limbs closest to the camera
-	Currently, steps are found by counting the peak x velocities
-	Slips are found by finding the average y position of the two limbs when the absolute value of x and y velocity<0.9 and then counting the peaks of the y position of the limbs when they go “below” that average. (0,0 in the video is actually the top left corner. so a higher y value is lower in the screen)
