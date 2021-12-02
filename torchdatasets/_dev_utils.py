################################################################################
#
#                                HELPERS
#
################################################################################


def apply_mapping(sample, mappings, start, end):
    """Helper applying maps in defined threshold."""
    for mapping in mappings[start:end]:
        sample = mapping(sample)
    return sample


def reversed_enumerate(iterable):
    return zip(range(len(iterable) - 1, -1, -1), reversed(iterable))
