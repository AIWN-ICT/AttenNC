"""Topology helpers for neighbor lookup and TDMA slot expansion."""


def getneighborNum(nodelist, nb_M, node_id):
    """Return next-hop neighbor ids for ``node_id`` according to the neighbor matrix."""
    num = []
    for i in range(node_id + 1, len(nb_M)):
        if nb_M[node_id][i] == 1:
            num.append(i)
    return num


def get_tdma_slot_schedule(frame_slot, node_num):
    """Return per-slot TDMA schedule as a list of ``(slot, node_id)`` pairs.

    ``FRAME_SLOT`` follows the FQF style: ``node_id -> one slot or a list of slots``.
    Slot 0 means the node has no active send slot.

    Important behavior:
    - A node with multiple active slots appears multiple times in one TDMA frame.
    - If no TDMA config is given, keep the old node-id order fallback by mapping
      to pseudo-slots ``1..node_num``.
    """
    if not frame_slot:
        return [(idx + 1, idx) for idx in range(node_num)]

    slot_node_pairs = []
    for node_id in range(node_num):
        slots = frame_slot.get(node_id, [])
        if not isinstance(slots, (list, tuple, set)):
            slots = [slots]

        active_slots = sorted({int(slot) for slot in slots if int(slot) > 0})
        for slot in active_slots:
            slot_node_pairs.append((slot, node_id))

    if not slot_node_pairs:
        return [(idx + 1, idx) for idx in range(node_num)]

    slot_node_pairs.sort(key=lambda x: (x[0], x[1]))
    return slot_node_pairs
