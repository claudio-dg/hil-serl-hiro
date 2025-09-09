
# Notes on Training Procedures
### General Guidelines:

•	Initially, the robot must be **continuously guided to success** (since success is binary, “almost correct” attempts do not provide useful learning signals).

•	Over time, interventions should be **gradually reduced**, allowing the robot to explore independently, with human interventions limited to:

 1.	_Preventing the robot from diverging too far from the task._
 2.	_Correcting trajectories to ultimately achieve success._
    
•	For more complex tasks (e.g., screwdriver operation):

* 	First signs of learning typically emerge after **15k–20k learning steps.**
* 	A minimally acceptable level of learning usually requires **35k–40k learning steps**.

PRACTICAL EXPERIENCES IN ROBOT TRAINING / DATA COLLECTION
Importance of Fixed & Consistent Cameras
•	Risk of overfitting to camera angles: it is crucial to maintain consistent camera positioning from data collection through training.
Alternative: fine-tuning with multiple camera angles to promote generalization, although this becomes computationally demanding given the additional variability to be managed
________________________________________
Personal Considerations
•	Initial training sessions are rarely optimal. Progressively, the user themselves learns how to better train the agent.
•	Common patterns of failure or misbehavior become recognizable, leading to:
o	Identification of misinterpretations causing incorrect learning.
o	Detection of bugs or inefficiencies in the classifier.
o	Refinement of parameters such as action ranges, step rate, episode time limits, or additional classifier parameters.
________________________________________
Reward System
•	A robust yet reproducible reward mechanism is essential (e.g., classifier + measurable parameters such as force Y).
•	Example: a screwdriver classifier perfectly detected correct insertions, but was unusable for training due to difficulties in reproducing such precise conditions with joystick lag, cameras, etc.
•	To address this:
o	Fine-tune the classifier after initial training attempts and/or while reviewing recorded demonstrations TO remove false positives misinterpreted by the network.
o	Evaluate strengths and weaknesses of the classifier to improve robustness.
________________________________________
Generalization in Training (especially witnessed with pick_up_box task)
•	Randomized resets (e.g., object position) are valuable ok, but further generalization should be introduced gradually.
•	Recommended approach:
o	Start with constant conditions to establish baseline learning.
o	Once learning has stabilized, progressively introduce variability.
•	Example (box task):
o	Randomized position at each reset accelerates generalization.
o	Keeping the box always oriented in the same direction greatly speeds up early learning, and the agent later generalizes orientation changes autonomously.
o	Introducing this variation during training further improves robustness.
________________________________________
⚠️ Penalty Mechanism ⚠️
•	With gripper tasks, penalties yielded poor results.
•	With screwdriver force tasks, experiments are still in progress.
•	Current observations:
o	Adding penalties clearly extends learning time.
o	Unclear whether they fully prevent undesired behavior or merely delay it.
o	Example: preventing continuous collisions with the table during screwdriver training—at 20k steps, slight improvements observed, but insufficient. Further testing required ASAP.
________________________________________
Action Range
•	Action range managed through box_clip method:
o	Wider ranges lead to longer learning times (due to increased exploration).
o	The clipping mechanism is crucial for safe behaviors, e.g., asymptotically approaching the table without making contact.
________________________________________
Opinions on Simulation → Reality Transfer
•	In my opinion Simulation-to-reality transfer is extremely challenging, may not work.
•	High sensitivity to minor scene variations makes direct transfer unlikely to succeed.
•	Even with near-perfect simulation fidelity (which is very difficult due to both camera setup and contact handling in MuJoCo), achieving robustness in the real world remains questionable.
