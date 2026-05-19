"""Action-selection helpers for source and relay coding decisions."""

import random

import numpy as np


def act_noise(q, action_size, epsilon):
    """Select an action with epsilon-greedy exploration."""
    rd = np.random.rand()
    if rd <= epsilon:
        a = random.randrange(action_size)
        return int(a)
    return int(np.argmax(q))


def act(q):
    """Select the greedy action from a Q-value vector."""
    return int(np.argmax(q))


def p_xor(p1, p2, K=None):
    """XOR two binary packets.

    K is optional for backward compatibility with the old global-K version.
    When omitted, the length of p1 is used.
    """
    packet_length = len(p1) if K is None else K
    p_new = np.zeros(packet_length)
    for i in range(packet_length):
        if p1[i] == p2[i]:
            p_new[i] = 0
        else:
            p_new[i] = 1
    return p_new
