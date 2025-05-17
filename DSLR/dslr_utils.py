




def unique_combinations(data: list) -> list:
    combinations = []
    remaining = data[1:]
    for item in data:
        for r in remaining:
            combinations.append((item, r))
        if len(remaining) > 1:
            remaining.pop(0)
        else:
            break
    return combinations

def next_slot(slot: list, nrows: int, ncols: int) -> None:
    if slot[1] < ncols - 1:
        slot[1] += 1
    elif slot[0] < nrows - 1:
        slot[0] += 1
        slot[1] = 0
