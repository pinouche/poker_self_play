- make sure this is implemented with generation running async in multiprocess while training runs:

"Second, generation is embarrassingly parallel (independent trajectories, no communication) while synchronous SGD across 720 GPUs would be communication-bound. So the natural     
  shape                                                                                                                                                                              
    is many actors → replay buffer → one learner, the standard Ape-X/AlphaZero pattern. The buffer decouples them: actors don't wait for the learner, the learner doesn't wait for   
                                                                                                                                                                                     
    data, and actors just periodically pull fresh weights — so the coupling is real but asynchronous and loose."

also check how long it would take.

- implement: D Domain Knowledge Leveraged in our Poker AI Agent