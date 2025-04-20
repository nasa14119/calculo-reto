from tqdm import tqdm


def loader(iterable, desc="Loading", width=80):
    return tqdm(
        iterable,
        desc=desc,
        ncols=width,
        bar_format="{bar:20} {n_fmt}/{total_fmt} | {desc}",
    )
