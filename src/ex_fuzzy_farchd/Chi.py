"""
1. F1 Score
2. Algorithm schema
3. Bigger adaptative figure
4. Updated CV
"""
import numpy as np
from sklearn.base import ClassifierMixin

try:
    from ex_fuzzy import fuzzy_sets as fs
    from ex_fuzzy import rules
    from ex_fuzzy import eval_rules as evr
    from ex_fuzzy import vis_rules
    from ex_fuzzy import utils
except ImportError:
    import fuzzy_sets as fs
    import rules
    import eval_rules as evr
    import vis_rules
    import utils


class FitRuleBase():
    vl_names = [  # Linguistic variable names prenamed for some specific cases.
        [],
        [],
        ['Low', 'High'],
        ['Low', 'Medium', 'High'],
        ['Low', 'Medium', 'High', 'Very High'],
        ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    ]


def create_partitions(X, n_partitions=3, partition_type="triangular-equal", fuzzy_type=fs.FUZZY_SETS.t1):
    """
    Create partitions for the given data
    :param X: Data
    :param partition_type: Type of partition: triangular(-equal), trapezoidal or gaussian
    :param fuzzy_type: Type of fuzzy set
    :return: List of partitions
    """

    points = lambda start,end,n : [start+i*(end-start)/(n-1) for i in range(n)]

    if partition_type == "triangular-equal":  # triangular and equal
        X = np.array([np.min(X, axis=0), np.max(X, axis=0)])
        fuzzy_variables = []
        for x1, x2 in X.T:
            fuzzy_sets = []
            critical_points = points(x1, x2, n_partitions)
            fuzzy_sets.append(fs.FS("0",[critical_points[0], critical_points[0], critical_points[0], critical_points[1]],X.T[0]))
            for i in range(1,n_partitions-1):
                fuzzy_sets.append(fs.FS(str(i),[critical_points[i-1],critical_points[i], critical_points[i], critical_points[i+1]],X.T[i]))
            fuzzy_sets.append(fs.FS(str(n_partitions-1),[critical_points[-2], critical_points[-1], critical_points[-1], critical_points[-1]],X.T[-1]))
            fuzzy_variables.append(fs.fuzzyVariable("name", fuzzy_sets))
        return fuzzy_variables
    if partition_type == "triangular":
        fuzzy_variables = utils.construct_partitions(X, fuzzy_type)
        for fv in fuzzy_variables:
            for fset in fv:
                if fset.membership_parameters[0] == fset.membership_parameters[1] and fset.membership_parameters[1] != fset.membership_parameters[2]:
                    fset.membership_parameters[2] = fset.membership_parameters[1]  # This special case is -\ and we convert to \
                elif fset.membership_parameters[2] == fset.membership_parameters[3] and fset.membership_parameters[1] != fset.membership_parameters[2]:
                    fset.membership_parameters[1] = fset.membership_parameters[2]  # This special case is /- and we convert to /
                else: # /-\ => /\
                    fset.membership_parameters[1] = fset.membership_parameters[2] = (fset.membership_parameters[1] + fset.membership_parameters[2])
        return fuzzy_variables
        # FIXME: Not defined anymore
        # return utils._triangular_construct_partitions(X, fuzzy_type)
    else:  # TODO: Gaussian. ..
        return utils.construct_partitions(X, fuzzy_type)


def create_rule_base(partitions, rules_by_keys, consequent_names) -> rules.MasterRuleBase:
    # NOTE: I add consequent names becasue it can happen that in some cases
    # There is a consequent missing and so the new rule base is created wrongly
    rule_bases = []
    for key in consequent_names:
        rule_lst = []
        if key in rules_by_keys:
            for rule_i in rules_by_keys[key]:
                # FIXME where to store the original rules with indices
                R = rules.RuleSimple(rule_i["var_idx"], key)
                # R.score = rule_i["score"]  # IF we add score. then it does not get update on the client
                rule_lst.append(R)
        RB = rules.RuleBaseT1(antecedents=partitions, rules=rule_lst)  # , consequent=key
        RB.consequent = key
        rule_bases.append(RB)
    master_rule_base = rules.MasterRuleBase(rule_bases, consequent_names)
    return master_rule_base


def delete_conflicting_rules(rules: np.array, rules_count: np.array = None,
                             keep_threshold: float = np.inf) -> np.array:
    """
    Delete conflicting rules from the rule base, those with the same antecedents but different consequents
    :param rules: Rule base
    :param rules_count: Count of each rule
    :param keep_threshold: Threshold to keep a conflicting rule. i.e. how many times one rule should appear more than the other to keep it
    :return: Rule base without conflicting rules
    """
    rules_base = rules.copy()
    n_rules = rules_base.shape[0]
    dups = []
    for i in range(n_rules):
        for j in range(i + 1, n_rules):
            if np.array_equal(rules_base[i, :-1], rules_base[j, :-1]) and rules_base[i, -1] != rules_base[j, -1]:
                if rules_count is not None:
                    if rules_count[i] > rules_count[j] and rules_count[i] / rules_count[j] > keep_threshold:
                        dups.append(j)
                    elif rules_count[j] > rules_count[i] and rules_count[j] / rules_count[i] > keep_threshold:
                        dups.append(i)
                    else:
                        dups.append(i)
                        dups.append(j)
                else:
                    dups.append(j)
                    dups.append(i)
    rules_base = np.delete(rules_base, dups, axis=0)
    return rules_base


def rule_base_from_data(X: np.ndarray, y, fz_type_studied, consequent_names: list,
                        precomputed_partitions: list[fs.fuzzyVariable] = None):
    if precomputed_partitions is None:
        precomputed_partitions = utils.construct_partitions(X, fz_type_studied)
    memberships = rules.compute_antecedents_memberships(precomputed_partitions, X)
    # FIXME: categorical variables may have a different number of fuzzy sets
    antecedents = np.argmax(np.array(memberships), axis=1).T
    antecedents_consequents = np.column_stack([antecedents, y])
    rules_base, rules_count = np.unique(antecedents_consequents, axis=0, return_counts=True)
    # TODO: Deal with conflicting rules (i.e. same antecedents different consequent)

    rules_base = delete_conflicting_rules(rules_base)
    # rules_base = delete_conflicting_rules(rules_base, rules_count, keep_threshold=10.0)

    rules_by_keys = {}
    consequents = np.unique(rules_base[:, -1])
    for consequent in consequents:
        con_idx = np.where(rules_base[:, -1] == consequent)
        rulesonekey = []
        for rule in rules_base[con_idx]:
            rdic = {"var_idx": rule[:-1], "consequent": rule[-1], "score": 1}
            rulesonekey.append(rdic)
        rules_by_keys[consequent] = rulesonekey

    master_rule_base = create_rule_base(precomputed_partitions, rules_by_keys, consequents)
    for rule_base in master_rule_base:
        for rule in rule_base:
            rule.score = 1
    return master_rule_base


class ChiFuzzyRulesClassifier(ClassifierMixin):
    '''
    Class that is used as a classifier for a fuzzy rule based system. Supports precomputed and optimization of the linguistic variables.
    '''

    def __init__(self, nRules: int = 30, nAnts: int = 4,
                 fuzzy_type: fs.FUZZY_SETS = fs.FUZZY_SETS.t1, tolerance: float = 0.0, class_names: list[str] = None,
                 n_linguistic_variables: list[int] | int = 3, verbose=False,
                 linguistic_variables: list[fs.fuzzyVariable] = None,
                 domain: list[float] = None, n_class: int = None,
                 precomputed_rules: rules.MasterRuleBase = None, runner: int = 1, ds_mode: int = 0,
                 fuzzy_modifiers: bool = False, allow_unknown: bool = False) -> None:
        '''
        Inits the optimizer with the corresponding parameters.

        :param nRules: number of rules to optimize.
        :param nAnts: max number of antecedents to use.
        :param fuzzy type: FUZZY_SET enum type in fuzzy_sets module. The kind of fuzzy set used.
        :param tolerance: tolerance for the dominance score of the rules.
        :param n_linguist_variables: number of linguistic variables per antecedent.
        :param verbose: if True, prints the progress of the optimization.
        :param linguistic_variables: list of fuzzyVariables type. If None (default) the optimization process will init+optimize them.
        :param domain: list of the limits for each variable. If None (default) the classifier will compute them empirically.
        :param n_class: names of the classes in the problem. If None (default) the classifier will compute it empirically.
        :param precomputed_rules: MasterRuleBase object. If not None, the classifier will use the rules in the object and ignore the conflicting parameters.
        :param runner: number of threads to use. If None (default) the classifier will use 1 thread.
        :param ds_mode: mode for the dominance score. 0: normal dominance score, 1: rules without weights, 2: weights optimized for each rule based on the data.
        :param fuzzy_modifiers: if True, the classifier will use the modifiers in the optimization process.
        :param allow_unknown: if True, the classifier will allow the unknown class in the classification process. (Which would be a -1 value)
        '''

        if precomputed_rules is not None:
            self.nRules = len(precomputed_rules.get_rules())
            self.nAnts = len(precomputed_rules.get_rules()[0].antecedents)
            self.n_class = len(precomputed_rules)
            self.nclasses_ = len(precomputed_rules.consequent_names)
            self.classes_names = precomputed_rules.consequent_names
            self.rule_base = precomputed_rules
        else:
            self.nRules = nRules
            self.nAnts = nAnts
            self.nclasses_ = n_class
            if not (class_names is None):
                if isinstance(class_names, np.ndarray):
                    self.classes_names = list(class_names)
                else:
                    self.classes_names = class_names
            else:
                self.classes_names = class_names

        self.verbose = verbose
        self.tolerance = tolerance
        self.ds_mode = ds_mode
        self.fuzzy_modifiers = fuzzy_modifiers
        self.allow_unknown = allow_unknown

        if linguistic_variables is not None:
            # If the linguistic variables are precomputed then we act accordingly
            self.lvs = linguistic_variables
            self.n_linguist_variables = [len(lv.linguistic_variable_names()) for lv in self.lvs]
            self.domain = None
            self.fuzzy_type = self.lvs[0].fuzzy_type()

            if self.nAnts > len(linguistic_variables):
                self.nAnts = len(linguistic_variables)
                if verbose:
                    print(
                        'Warning: The number of antecedents is higher than the number of variables. Setting nAnts to the number of linguistic variables. (' + str(
                            len(linguistic_variables)) + ')')

        else:
            # If not, then we need the parameters sumistered by the user.
            self.lvs = None
            self.fuzzy_type = fuzzy_type
            self.n_linguist_variables = n_linguistic_variables
            self.domain = domain

    def fit(self, X: np.array, y: np.array,
            candidate_rules: rules.MasterRuleBase = None,
            initial_rules: rules.MasterRuleBase = None,
            random_state: int = 33,
            bootstrap_size=1000,
            p_value_compute=False) -> None:
        '''
        Fits a fuzzy rule based classifier using a genetic algorithm to the given data.

        :param X: numpy array samples x features
        :param y: labels. integer array samples (x 1)
        :param candidate_rules: if these rules exist, the optimization process will choose the best rules from this set. If None (default) the rules will be generated from scratch.
        :param initial_rules: if these rules exist, the optimization process will start from this set. If None (default) the rules will be generated from scratch.
        :param random_state: integer. Random seed for the optimization process.

        :return: None. The classifier is fitted to the data.
        '''
        # fIXME: only initial_rules or candidate_rules

        if self.classes_names is None:
            self.classes_names = [aux for aux in np.unique(y)]

        if self.nclasses_ is None:
            self.nclasses_ = len(self.classes_names)

        if isinstance(y[0], str):
            y = np.array([self.classes_names.index(str(aux)) for aux in y])

        if candidate_rules is None:
            if initial_rules is not None:
                self.fuzzy_type = initial_rules.fuzzy_type()
                self.n_linguist_variables = initial_rules.n_linguistic_variables()
                self.domain = [fv.domain for fv in initial_rules[0].antecedents]
                self.nRules = len(initial_rules.get_rules())
                self.nAnts = len(initial_rules.get_rules()[0].antecedents)

            if self.lvs is None:
                # Check if self.n_linguist_variables is a list or a single value.
                if isinstance(self.n_linguist_variables, int):
                    self.n_linguist_variables = [self.n_linguist_variables for _ in range(X.shape[1])]

                if self.nAnts > X.shape[1]:
                    self.nAnts = X.shape[1]
                    if self.verbose:
                        print(
                            'Warning: The number of antecedents is higher than the number of variables. Setting nAnts to the number of variables. (' + str(
                                X.shape[1]) + ')')

                self.lvs = utils.construct_partitions(X, self.fuzzy_type)
            else:
                # If Fuzzy variables are already precomputed.
                pass  # Delete this else
        else:  # we add new rules to the providing one, expanding the rule base
            self.fuzzy_type = candidate_rules.fuzzy_type()
            self.n_linguist_variables = candidate_rules.n_linguistic_variables()
            pass

        try:
            self.var_names = list(X.columns)
            self.X = X.values
        except AttributeError:
            self.X = X
            self.var_names = [str(ix) for ix in range(X.shape[1])]

        precomputed_partitions = None  # For now
        consequent_names = self.classes_names
        self.rule_base = rule_base_from_data(X, y, self.fuzzy_type,
                                             consequent_names, self.lvs)

        self.performance = 1  # TODO: Compute performance of the training set # 1 - fitness_last_gen[best_solution]

        self.eval_performance = evr.evalRuleBase(
            self.rule_base, np.array(X), y)
        self.eval_performance.add_full_evaluation()
        # FIXME: I dont want to purge for chi, I want to generate all possible rules
        # self.rule_base.purge_rules(self.tolerance)
        self.eval_performance.add_full_evaluation()  # After purging the bad rules we update the metrics.

        if p_value_compute:
            self.p_value_validation(bootstrap_size)

        self.rule_base.rename_cons(self.classes_names)
        if self.lvs is None:
            self.rename_fuzzy_variables()

    def p_value_validation(self, bootstrap_size: int = 100):
        '''
        Computes the permutation and bootstrapping p-values for the classifier and its rules.

        :param bootstrap_size: integer. Number of bootstraps samples to use.
        '''
        self.p_value_class_structure, self.p_value_feature_coalitions = self.eval_performance.p_permutation_classifier_validation()

        self.eval_performance.p_bootstrapping_rules_validation(bootstrap_size)

    def load_master_rule_base(self, rule_base: rules.MasterRuleBase) -> None:
        '''
        Loads a master rule base to be used in the prediction process.

        :param rule_base: ruleBase object.
        :return: None
        '''
        self.rule_base = rule_base
        self.nRules = len(rule_base.get_rules())
        self.nAnts = len(rule_base.get_rules()[0].antecedents)
        self.nclasses_ = len(rule_base)

    def forward(self, X: np.array, out_class_names=False) -> np.array:
        '''
        Returns the predicted class for each sample.

        :param X: np array samples x features.
        :param out_class_names: if True, the output will be the class names instead of the class index.
        :return: np array samples (x 1) with the predicted class.
        '''
        try:
            X = X.values  # If X was a pandas dataframe
        except AttributeError:
            pass

        return self.rule_base.winning_rule_predict(X, out_class_names=out_class_names)

    def predict(self, X: np.array, out_class_names=False) -> np.array:
        '''
        Returns the predicted class for each sample.

        :param X: np array samples x features.
        :param out_class_names: if True, the output will be the class names instead of the class index.
        :return: np array samples (x 1) with the predicted class.
        '''
        return self.forward(X, out_class_names=out_class_names)

    def predict_proba(self, X: np.array) -> np.array:
        '''
        Returns the predicted class probabilities for each sample.

        :param X: np array samples x features.
        :return: np array samples x classes with the predicted class probabilities.
        '''
        try:
            X = X.values  # If X was a pandas dataframe
        except AttributeError:
            pass

        return self.rule_base.compute_association_degrees(X)

    def print_rules(self, return_rules: bool = False) -> None:
        '''
        Print the rules contained in the fitted rulebase.
        '''
        return self.rule_base.print_rules(return_rules)

    def plot_fuzzy_variables(self) -> None:
        '''
        Plot the fuzzy partitions in each fuzzy variable.
        '''
        fuzzy_variables = self.rule_base.rule_bases[0].antecedents

        for ix, fv in enumerate(fuzzy_variables):
            vis_rules.plot_fuzzy_variable(fv)

    def rename_fuzzy_variables(self) -> None:
        '''
        Renames the linguist labels so that high, low and so on are consistent. It does so usually after an optimization process.

        :return: None. Names are sorted accorded to the central point of the fuzzy memberships.
        '''

        for ix in range(len(self.rule_base)):
            fuzzy_variables = self.rule_base.rule_bases[ix].antecedents

            for jx, fv in enumerate(fuzzy_variables):
                new_order_values = []
                possible_names = FitRuleBase.vl_names[self.n_linguist_variables[jx]]

                for zx, fuzzy_set in enumerate(fv.linguistic_variables):
                    studied_fz = fuzzy_set.type()

                    if studied_fz == fs.FUZZY_SETS.temporal:
                        studied_fz = fuzzy_set.inside_type()

                    if studied_fz == fs.FUZZY_SETS.t1:
                        f1 = np.mean(
                            fuzzy_set.membership_parameters[0] + fuzzy_set.membership_parameters[1])
                    elif (studied_fz == fs.FUZZY_SETS.t2):
                        f1 = np.mean(
                            fuzzy_set.secondMF_upper[0] + fuzzy_set.secondMF_upper[1])
                    elif studied_fz == fs.FUZZY_SETS.gt2:
                        sec_memberships = fuzzy_set.secondary_memberships.values()
                        f1 = float(list(fuzzy_set.secondary_memberships.keys())[np.argmax(
                            [fzm.membership_parameters[2] for ix, fzm in enumerate(sec_memberships)])])

                    new_order_values.append(f1)

                new_order = np.argsort(np.array(new_order_values))
                fuzzy_sets_vl = fv.linguistic_variables

                for jx, x in enumerate(new_order):
                    fuzzy_sets_vl[x].name = possible_names[jx]

    def get_rulebase(self) -> list[np.array]:
        '''
        Get the rulebase obtained after fitting the classifier to the data.

        :return: a matrix format for the rulebase.
        '''
        return self.rule_base.get_rulebase_matrix()

    def __call__(self, X: np.array) -> np.array:
        '''
        Returns the predicted class for each sample.

        :param X: np array samples x features.
        :return: np array samples (x 1) with the predicted class.
        '''
        return self.predict(X)


def print_linguistic_variables(lvs):
    for lv_i in lvs:
        lingustic_variables = lv_i.get_linguistic_variables()
        print(f"lingustic variable: {lv_i.name}")
        for lv in lingustic_variables:
            print(lv)

if __name__ == "__main__":

    def test_random_classification(fs_type=fs.FUZZY_SETS.t1):
        sample_size = 1000

        sample = np.random.random_sample((sample_size, 5))
        targets = np.random.randint(0, 2, sample_size)

        p = create_partitions(sample)
        model = ChiFuzzyRulesClassifier(10, 3, fs_type, verbose=False, tolerance=0.0, n_linguistic_variables=3,
                                        linguistic_variables=p)
        model.fit(sample, targets)
        predictions = model.predict(sample)
        import math
        assert math.isclose(np.mean(np.equal(predictions, targets)), 0.5, abs_tol=0.1)


    X = np.random.random_sample((3, 5))
    p = create_partitions(X)
    print_linguistic_variables(p)

    test_random_classification()
    print("test_random_classification() passed")
