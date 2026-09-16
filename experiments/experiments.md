# Batches
To compare apples to apples, I'm doing model training by batch, in each batch only one set of config changes.<br>
The idea is that if I ever need to change the code/algorithm logic, or do substantial changes in the SDK, I must start
a new batch. 

Each batch has for objective to teach us something. If we can accomplish that that's enough for me, hopefully each bach yields better models on average.

## Batch I : Red

### Goal
This is the first batch, the goal was to test the training/eval loop and explore a bit on how the problem presents itself. <br>
I used this batch to explore a tiny bit with RewardConfig and got some interesting results.
Here are the reward config I used:
* shaped: Honestly the config ChatGPT output at first, good enough to start the experiment, 80% based on outcome then the rest is split in majority between how much HP do we and the enemy have left at the end of the fight, and 5% based on the battle speed
* outcome: 100% based on wether we win or loose
* mixed: 50% based on the outcome, 25% based on own HP and 25% on enemy HP
* hp: 50% on HP and 50% enemy HP
* skewed-own: 50% outcome and 50% own HP
* skewed-enemy: 50% outcome and 50% enemy HP

The interesting ones are hp and the skewed ones, reasonning:
* hp: This one is funny because it gives 0 credit wether we win or loose actually, if we loose or win by one HP the model will think "meh, good enough" while for a normal person, what matters is only the victory. While it's true, the HP one gives soooo much information compared to the boolean win vs lose, that I thought it would be a great starting point.
* Skewed-enemy: I thought that, we probably have more to learn in defeats than in victories, so that's why I thought maybe we should not treat our HP and the enemy HP as something symetric by nature.
* Skewed-own: For science, by symetry with skewed-enemy 


Now, how does own/enemy HP works ? For our own team I attribute a score between 0 and 1 propoertionnaly on how much HP we have left at the end of the fight. It only matters if we win, because if we loose we end up with 0 HP everywhere. For the enemy team it is the opposite, the reward only matters if we loose, and tell us by how much did we loose. 

I suspected it would rewards based on the HP would end up being good because it tells us so much more than a boolean if we win or loose and it is a very good proxy for how good did we do during the battle.

### Results
The results come to confirm my hypothesis, but I was surprized by how:
| model | hp | outcome | mixed | shaped | skewed-enemy | skewed-own |
| --- | --- | --- | --- | --- | --- | --- |
| wr vs Random | 73% | 61% | 58% | 55% | 59% | 56% |
| Rank | 1 | 2 | 3 | 5 | 3 | 5 |

Given that I can have about 3% variation in the Win Rate over a thousand battles,  I gave generous ranking to these models. 

Now, the good new is that the model hp has, without any doubt, learned something and made progress during training.<br>
The bas news is that, it's 73% win rate but against a bot that plays randomly, altogether those numbers aren't that impressive, especally for the other models, there is still a lot to improve. 

As comparison, poke-env has a bot that plays with a few simple heuristics (play super effective moves, don't inflict status on a pokemon already inflicted, etc). Against it, our HP based model has only a 13% win rate.<br>
To give an idea, the simple heuristics model is about as good as what I was in middle school, back when I knew my type chart by heart. 

### Observation
I run each models for 500 iterations, that would amount to aout 6h of training per model.
But most of them reach their peak performance against our random bot by iteration 100, the latest being our HP model learning until iteration 150.

This means performances degrade after a while and we can only guess why.

### Next batch
I currently have a thousand ideas on how to improve our model, but before anything else, we need to measure how stable are our runs. i.e, if I run the same experiment once, am I sure I will get the same results ? 

I'm used Supervised Learning, where a training loop, if not deterministic, at least gives us models with about the same performances. But in RL, the models choices affect the learning, so each model takes its own path to victory. <br>
One can't help be wonder, are these paths close to each other, or can one model find the right strategy that will lead it to victory where another with the same configuration can fail to see ? 

To answer this, we will run the same training with variation in the randomness seed and nothing more. I will take our best model (HP) up to 300 iterations with random seeds and see if we can see variations their training.
