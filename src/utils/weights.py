import torch

def get_weight_by_counts(counts, min_value=1e-8):
    '''
    Args:
        counts:
            counts of symbols in each level

            shape:
                decode_level * number of symbols
    '''
    def normalize(array, min_value):
        array = torch.tensor(array) + 1
        array = array/array.sum()
        array = 1/array
        array = array/array.sum()
        for i in range(array.size(0)):
            array[i] = max(array[i], min_value)
        return array

    out = []
    for count in counts:
        out.append(normalize(count, min_value))

    return out