- encoder-limited — the board is fed to the network as a bare 52-way indicator with no structure (rank, suitedness, connectivity), so it cannot generalize to boards it  
hasn't seen no matter how many it gets. This is the research risk I flagged for stage B.
In ReBel's use card embedding for the board cards similar to [7] and then apply MLP.

- Proper network with GeLu etc (see notes) also for Paradigm A

- Can this make use or mix with paradigm A?

- Train an exploiter like in Paradigm A too see if B is exploitable