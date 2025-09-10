# Notes on RLPD Training Procedures

## ```General Guidelines```
Initially, the robot must be **continuously guided to success** (since success is binary, “almost correct” attempts do not provide useful learning signals).

Over time, interventions should be **gradually reduced**, allowing the robot to explore independently, with human interventions limited to:

1. _Preventing the robot from diverging too far from the task._
2. _Correcting trajectories to ultimately achieve success._

For more complex tasks (e.g., screwdriver operation):

- First signs of learning typically emerge after **15k–20k learning steps**.
- A minimally acceptable level of learning usually requires **35k–40k learning steps**.

## ```Personal Considerations```
Initial training sessions are **rarely optimal**. Progressively, the user themselves learns how to better train the agent.  
Common patterns of failure or misbehavior become recognizable, leading to:

- Identification of misinterpretations causing incorrect learning.  
- Detection of bugs or inefficiencies in the classifier.  
- Refinement of parameters such as action ranges, step rate, episode time limits, or additional classifier parameters.  

---

## ```Practical Experiences in Robot Training / Data Collection```

### 1. 📷 Importance of Fixed & Consistent Cameras 📷
- **Overfitting to camera angles** → it is crucial to maintain _consistent camera positioning from data collection through training_.  
- **Alternative**: fine-tuning with multiple camera angles to promote generalization, although this becomes **computationally demanding** due to the added variability.  

---

### 2. Reward System
A **robust yet reproducible** reward mechanism is essential (e.g., image classifier + measurable parameters such as force along Y-axis).

**Example**:  
A screwdriver classifier perfectly detected correct insertions, but was unusable for training due to difficulties in reproducing such precise conditions with joystick lag, cameras, etc.

**Mitigation strategies:**
- Fine-tune the classifier after initial training attempts and/or while reviewing recorded demonstrations.  
- Remove **false positives** misinterpreted by the network.  
- Evaluate strengths and weaknesses of the classifier to improve robustness.  

#### `Dense Reward Vs Sparse Reward`

In this project, two different reward strategies are explored:
The first one is the **Dense Reward** approach, where the policy receives a continuous feedback signal (from 0 to 1). This value may increase proportionally to the robot’s distance from its goal. Such a method is used in the pick_cube **simulation** tasks, where the robot does not rely on the image classifier, but instead obtains a reward at each simulation step based on its distance from the box to be picked up.
This setup, however, cannot be replicated on the **real robot**. In this case, a **Sparse Reward** strategy is employed: the robot only receives discrete feedback (0 or 1) for each episode, determined by the output of the image classifier (and other measurable parameters, as mentioned earlier). This means that the robot can obtain at most a single reward, at the end of a successful episode.

As one can imagine, _these two reward structures lead to significantly different training dynamics_. **Continuous** rewards enable a form of intermediate learning, where even **partial progress** (e.g., getting closer to success) provides useful feedback to the policy. With **binary** rewards, on the other hand, it is critical that the robot completes the entire task successfully multiple times in order to learn the policy. Consequently, human intervention becomes particularly important during training with sparse rewards, to guide the robot toward success.

---

### 3. Generalization in Training
_(especially observed with the `pick_up_box` task)_

- Randomized resets (e.g., object position) are valuable, but further generalization should be introduced gradually.  

**Recommended approach:**
- Start with **constant conditions** to establish baseline learning.  
- Once learning has stabilized, progressively introduce variability.  

**Example (box task):**
- Randomized position at each reset accelerates generalization.  
- Keeping the box always oriented in the same direction **greatly speeds up** early learning.  
- The agent later generalizes orientation changes autonomously.  
- Introducing this variation during training further improves robustness.  

---

## ⚠️ Penalty Mechanism ⚠️
- With **gripper tasks**, penalties yielded poor results. 🛑  
- With **screwdriver force tasks**, experiments are still in progress. 🔄  

**Current observations:**
- Adding penalties clearly **extends learning time**.  
- Unclear whether they fully **prevent undesired behavior** or merely **delay it**.  
- Example: preventing continuous collisions with the table during screwdriver training — at ~20k steps, slight improvements observed, but insufficient. Further testing required.  

---

## Action Range
- Action range managed through the **`box_clip`** method:  
  - Wider ranges lead to **longer learning times** (due to increased exploration).  
  - The clipping mechanism is crucial for **safe behaviors**, e.g., asymptotically approaching the table without making contact.  

---

## Opinions on Simulation → Reality Transfer
- **Simulation-to-reality transfer is extremely challenging** and may not work as expected.  
- High sensitivity to minor scene variations makes direct transfer unlikely to succeed.  
- Even with near-perfect simulation fidelity (very difficult due to both camera setup and contact handling in MuJoCo), achieving robustness in the real world remains questionable.  
