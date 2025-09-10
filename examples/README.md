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
A **robust yet reproducible** reward mechanism is essential (e.g., classifier + measurable parameters such as force along Y-axis).

**Example**:  
A screwdriver classifier perfectly detected correct insertions, but was unusable for training due to difficulties in reproducing such precise conditions with joystick lag, cameras, etc.

**Mitigation strategies:**
- Fine-tune the classifier after initial training attempts and/or while reviewing recorded demonstrations.  
- Remove **false positives** misinterpreted by the network.  
- Evaluate strengths and weaknesses of the classifier to improve robustness.  

#### `Sparse Reward Vs Dense Reward`
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
