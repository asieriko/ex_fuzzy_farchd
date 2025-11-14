import numpy as np
import ex_fuzzy.fuzzy_sets as fs

def create_fuzzy_set(name:str, domain:list[int], params:list[list], fsnames=["Low","Medium","High"]) -> fs.FS:
    """
    Create a fuzzy set with the given name, domain, parameters and fuzzy set names.

    Args:
        name (str): Name of the fuzzy set.
        domain (list[int]): Domain of the fuzzy set.
        params (list[list]): Parameters for the fuzzy sets.
        fsnames (list[str]): Names of the fuzzy sets.

    Returns:
        fs.FS: The created fuzzy set.
    """
    # cold = fs.FS('Cold', [0, 0, 5, 15], [0, 40])
    # warm = fs.FS('Warm', [15, 20, 25, 30], [0, 40])
    # hot = fs.FS('Hot', [25, 30, 40, 40], [0, 40])

    return fs.FS(name, params, domain)

def create_fuzzy_variable(name:str, domain:list[int], params:list[list], fsnames=["Low","Medium","High"]) -> fs.fuzzyVariable:
    """
    Create a fuzzy set with the given name, domain, parameters and fuzzy set names.

    Args:
        name (str): Name of the fuzzy variable.
        domain (list[int]): Domain of the fuzzy variable.
        params (list[list]): Parameters for the fuzzy sets.
        fsnames (list[str]): Names of the fuzzy sets.

    Returns:
        fs.fuzzyVariable: The created fuzzy set.
    """
    fsets = []
    for p, fsn in zip(params,fsnames):
        fsets.append(fs.FS(fsn, p, domain))

    return  fs.fuzzyVariable(name, fsets)


def create_partititions(partitions, names=None) -> list[fs.FS]:
    print(partitions)
    #partition n_variables + n_terms * n_points (3 for triangular, 4 trapezoidal)
    if names is None:
        names = [f"Variable{i}" for i in range(len(partitions))]
    ex_fuzzy_partitions = []
    for name, data in zip(names,partitions):
        domain = [np.min(data), np.max(data)]
        new_column = data[:,1].reshape(-1, 1)  # IF trinagular...
        new_data = np.insert(data,2,new_column,axis=1)

        variables = create_fuzzy_variable(name, domain, new_data)
        ex_fuzzy_partitions.append(variables)

    from ex_fuzzy.vis_rules import plot_fuzzy_variable

    return ex_fuzzy_partitions