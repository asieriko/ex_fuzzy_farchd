import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from copy import deepcopy
import warnings

try:
    from ex_fuzzy import fuzzy_sets as fs
except ImportError:
    import fuzzy_sets as fs


class FuzzyAntecedent:
    """Wrapper class to store fuzzy set antecedent with metadata."""

    def __init__(self, fuzzy_set, attr_idx):
        """
        Initialize a fuzzy antecedent.

        Parameters
        ----------
        fuzzy_set : FS or categoricalFS
            The fuzzy set representing this antecedent
        attr_idx : int
            The attribute index this antecedent applies to
        """
        self.fuzzy_set = fuzzy_set
        self.attr_idx = attr_idx
        self.confidence = 0.0

    def membership(self, instance):
        """Return degree of membership [0,1] for the instance."""
        x = instance[self.attr_idx]
        if np.isnan(x):
            return 0.0
        return float(self.fuzzy_set.membership(np.array([x]))[0])

    def __str__(self):
        if type(self.fuzzy_set) == fs.FS:
            if self.fuzzy_set.membership_parameters[0] <= self.fuzzy_set.domain[0]:
                return f"X{self.attr_idx}: {self.fuzzy_set.domain[0]}-{self.fuzzy_set.membership_parameters[2]:.4f}->{self.fuzzy_set.membership_parameters[3]:.4f}"
            else:
                return f"X{self.attr_idx}: {self.fuzzy_set.membership_parameters[0]:.4f}->{self.fuzzy_set.membership_parameters[1]:.4f}-{self.fuzzy_set.domain[1]}"

        return f"X{self.attr_idx}: {self.fuzzy_set.name}"


class FuzzyRule:
    """A single fuzzy rule."""
    
    def __init__(self, consequent):
        self.antecedents = []
        self.consequent = consequent
        
    def add_antecedent(self, antecedent):
        """Add an antecedent to the rule."""
        self.antecedents.append(antecedent)
        
    def coverage_degree(self, instance):
        """Calculate degree of coverage using product t-norm."""
        degree = 1.0
        for antd in self.antecedents:
            degree *= antd.membership(instance)
        return degree
    
    def covers(self, instance):
        """Check if rule covers instance (degree > 0)."""
        return self.coverage_degree(instance) > 0
    
    def get_confidence(self):
        """Get rule confidence (m-estimate)."""
        if len(self.antecedents) == 0:
            return np.nan
        return self.antecedents[-1].confidence
    
    def compute_confidences(self, X, y, prior=None, m=2.0):
        """
        Compute confidence for each antecedent prefix using m-estimate.

        This method calculates the reliability of each rule prefix (partial rule)
        using the m-estimate formula. It should be called AFTER the rule is fully
        built (all antecedents added).

        The confidence represents: "Of all instances covered by the first i+1
        antecedents, what proportion belong to the target class?"

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target labels
        prior : float, optional
            Prior probability of the consequent class. If None, computed from y.
        m : float, default=2.0
            Laplace smoothing parameter for m-estimate

        Notes
        -----
        For each antecedent at position i, computes:
            confidence = (acc + m × prior) / (cov + m)
        where:
            - acc: sum of fuzzy memberships for correct class instances
            - cov: sum of fuzzy memberships for all instances
            - m: smoothing parameter
            - prior: P(consequent class)

        Example
        -------
        >>> rule.add_antecedent(fs_low, attr_idx=2)
        >>> rule.add_antecedent(fs_high, attr_idx=0)
        >>> rule.compute_confidences(X_train, y_train)
        >>> print(rule.antecedent_confidences)
        [0.85, 0.92]  # First condition: 85% confident, both: 92% confident
        """
        self.antecedent_confidences = []

        # Compute prior if not provided
        if prior is None:
            prior = np.sum(y == self.consequent) / len(y) if len(y) > 0 else 0.5

        # Calculate confidence for each prefix
        for i in range(len(self.antecedents)):
            acc = 0.0  # Accurate coverage (correct class)
            cov = 0.0  # Total coverage

            # Evaluate coverage with first i+1 antecedents
            for j in range(len(X)):
                instance = X[j]
                label = y[j]

                # Compute fuzzy coverage degree
                degree = 1.0
                for k in range(i + 1):
                    x = instance[self.antecedent_indices[k]]

                    # Handle missing values
                    if np.isnan(x):
                        degree = 0.0
                        break

                    # Get membership
                    membership = self.antecedents[k].membership(np.array([x]))
                    membership_value = float(membership[0]) if isinstance(membership, np.ndarray) else float(membership)
                    degree *= membership_value

                # Accumulate coverage
                cov += degree
                if label == self.consequent:
                    acc += degree

            # M-estimate formula
            confidence = (acc + m * prior) / (cov + m) if (cov + m) > 0 else 0.5
            self.antecedent_confidences.append(confidence)

    def __str__(self):
        if len(self.antecedents) == 0:
            return f"=> Class {self.consequent}"
        antd_str = " AND ".join(str(a) for a in self.antecedents)
        return f"({antd_str}) => Class {self.consequent}"


class FuriaRule:
    """
    FURIA rule using ex-fuzzy FS objects directly.

    This class represents a fuzzy rule with:
    - Antecedents as ex-fuzzy FS objects
    - Antecedent indices mapping each FS to a feature
    - Consequent as the target class

    Attributes
    ----------
    antecedents : list[FS]
        List of fuzzy sets (FS or categoricalFS objects) representing conditions
    antecedent_indices : list[int]
        List of feature indices, where antecedent_indices[i] is the feature index
        for antecedents[i]
    antecedent_confidences : list[float]
        Confidence value for each antecedent prefix (from m-estimate)
    consequent : int or any
        The class/output value this rule predicts

    Example
    -------
    >>> # Rule: IF (X2 is low AND X0 is high) THEN Class 1
    >>> fs1 = fs.FS("low", [-1e10, -1e10, 1.9, 2.0])
    >>> fs2 = fs.FS("high", [5.0, 5.5, 1e10, 1e10])
    >>> rule = FuriaRule(
    ...     antecedents=[fs1, fs2],
    ...     antecedent_indices=[2, 0],
    ...     consequent=1
    ... )
    >>> instance = [5.8, 3.2, 1.4, 0.3]
    >>> coverage = rule.coverage_degree(instance)
    """

    def __init__(self, antecedents=None, antecedent_indices=None, consequent=None):
        """
        Initialize a FURIA rule.

        Parameters
        ----------
        antecedents : list[FS], optional
            List of fuzzy sets representing rule conditions
        antecedent_indices : list[int], optional
            List of feature indices for each antecedent
        consequent : int or any, optional
            The class/output predicted by this rule
        """
        self.antecedents = antecedents if antecedents is not None else []
        self.antecedent_indices = antecedent_indices if antecedent_indices is not None else []
        self.antecedent_confidences = []  # Confidence for each antecedent prefix
        self.consequent = consequent

        # Validate that antecedents and indices have same length
        if len(self.antecedents) != len(self.antecedent_indices):
            raise ValueError(
                f"Length mismatch: {len(self.antecedents)} antecedents but "
                f"{len(self.antecedent_indices)} indices"
            )

    def add_antecedent(self, fuzzy_set, attr_idx):
        """
        Add an antecedent to the rule.

        Parameters
        ----------
        fuzzy_set : FS or categoricalFS
            The fuzzy set representing the condition
        attr_idx : int
            The feature index this fuzzy set applies to
        """
        self.antecedents.append(fuzzy_set)
        self.antecedent_indices.append(attr_idx)

    def coverage_degree(self, instance):
        """
        Calculate the degree of coverage for an instance using product t-norm.

        This computes how well the instance satisfies the rule by:
        1. For each antecedent, extract the value at its corresponding feature index
        2. Compute the membership degree of that value in the fuzzy set
        3. Multiply all membership degrees together (product t-norm)

        Parameters
        ----------
        instance : array-like
            Feature vector to evaluate (e.g., [5.1, 3.5, 1.4, 0.2])

        Returns
        -------
        float
            Coverage degree in [0, 1], where:
            - 1.0 = instance fully satisfies all conditions
            - 0.0 = instance doesn't satisfy at least one condition
            - (0, 1) = instance partially satisfies conditions (fuzzy)

        Example
        -------
        >>> # Rule: IF (X2 ≤ 1.9 AND X0 ≥ 5.0) THEN Class 0
        >>> instance = [5.5, 3.0, 1.2, 0.3]
        >>> # X2=1.2 ≤ 1.9 → membership = 1.0
        >>> # X0=5.5 ≥ 5.0 → membership = 1.0
        >>> # coverage = 1.0 * 1.0 = 1.0
        """
        degree = 1.0

        for fuzzy_set, attr_idx in zip(self.antecedents, self.antecedent_indices):
            # Extract feature value at the specified index
            x = instance[attr_idx]

            # Handle missing values
            if np.isnan(x):
                return 0.0

            # Compute membership degree for this antecedent
            membership = fuzzy_set.membership(np.array([x]))

            # Convert to float (handle both numpy array and scalar returns)
            if isinstance(membership, np.ndarray):
                membership_value = float(membership[0])
            else:
                membership_value = float(membership)

            # Apply product t-norm
            degree *= membership_value

            # Early termination if any antecedent has 0 membership
            if degree == 0.0:
                return 0.0

        return degree

    def covers(self, instance):
        """
        Check if the rule covers an instance (coverage degree > 0).

        Parameters
        ----------
        instance : array-like
            Feature vector to check

        Returns
        -------
        bool
            True if coverage_degree > 0, False otherwise
        """
        return self.coverage_degree(instance) > 0

    def get_confidence(self):
        """
        Get rule confidence (m-estimate).

        The confidence is taken from the last antecedent's confidence value,
        which represents the reliability of the complete rule.

        Returns
        -------
        float
            Confidence value in [0, 1], or np.nan if no antecedents
        """
        if len(self.antecedent_confidences) == 0:
            return np.nan
        return self.antecedent_confidences[-1]

    def compute_confidences(self, X, y, prior=None, m=2.0):
        """
        Compute confidence for each antecedent prefix using m-estimate.

        This method calculates the reliability of each rule prefix (partial rule)
        using the m-estimate formula. It should be called AFTER the rule is fully
        built (all antecedents added).

        The confidence represents: "Of all instances covered by the first i+1
        antecedents, what proportion belong to the target class?"

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target labels
        prior : float, optional
            Prior probability of the consequent class. If None, computed from y.
        m : float, default=2.0
            Laplace smoothing parameter for m-estimate

        Notes
        -----
        For each antecedent at position i, computes:
            confidence = (acc + m × prior) / (cov + m)
        where:
            - acc: sum of fuzzy memberships for correct class instances
            - cov: sum of fuzzy memberships for all instances
            - m: smoothing parameter
            - prior: P(consequent class)

        Example
        -------
        >>> rule.add_antecedent(fs_low, attr_idx=2)
        >>> rule.add_antecedent(fs_high, attr_idx=0)
        >>> rule.compute_confidences(X_train, y_train)
        >>> print(rule.antecedent_confidences)
        [0.85, 0.92]  # First condition: 85% confident, both: 92% confident
        """
        self.antecedent_confidences = []

        # Compute prior if not provided
        if prior is None:
            prior = np.sum(y == self.consequent) / len(y) if len(y) > 0 else 0.5

        # Calculate confidence for each prefix
        for i in range(len(self.antecedents)):
            acc = 0.0  # Accurate coverage (correct class)
            cov = 0.0  # Total coverage

            # Evaluate coverage with first i+1 antecedents
            for j in range(len(X)):
                instance = X[j]
                label = y[j]

                # Compute fuzzy coverage degree
                degree = 1.0
                for k in range(i + 1):
                    x = instance[self.antecedent_indices[k]]

                    # Handle missing values
                    if np.isnan(x):
                        degree = 0.0
                        break

                    # Get membership
                    membership = self.antecedents[k].membership(np.array([x]))
                    membership_value = float(membership[0]) if isinstance(membership, np.ndarray) else float(membership)
                    degree *= membership_value

                # Accumulate coverage
                cov += degree
                if label == self.consequent:
                    acc += degree

            # M-estimate formula
            confidence = (acc + m * prior) / (cov + m) if (cov + m) > 0 else 0.5
            self.antecedent_confidences.append(confidence)

    def __str__(self):
        """String representation of the rule."""
        if len(self.antecedents) == 0:
            return f"=> Class {self.consequent}"

        conditions = []
        for fuzzy_set, attr_idx in zip(self.antecedents, self.antecedent_indices):
            conditions.append(f"X{attr_idx} is {fuzzy_set.name}")

        antecedent_str = " AND ".join(conditions)
        confidence = self.get_confidence()
        conf_str = f" (CF={confidence:.3f})" if not np.isnan(confidence) else ""
        return f"IF ({antecedent_str}) THEN Class {self.consequent}{conf_str}"

    def __repr__(self):
        """Detailed representation for debugging."""
        confidence = self.get_confidence()
        conf_val = f"{confidence:.3f}" if not np.isnan(confidence) else "N/A"
        return (
            f"FuriaRule("
            f"antecedents={len(self.antecedents)}, "
            f"indices={self.antecedent_indices}, "
            f"consequent={self.consequent}, "
            f"confidence={conf_val}"
            f")"
        )


class FURIA(BaseEstimator, ClassifierMixin):
    """
    FURIA: Fuzzy Unordered Rule Induction Algorithm
    
    Parameters
    ----------
    n_folds : int, default=3
        Number of folds for data split (growing/pruning)
    
    min_no : float, default=2.0
        Minimum total weight of instances in a rule
    
    n_optimizations : int, default=2
        Number of optimization runs
    
    check_error_rate : bool, default=True
        Whether to check error rate >= 0.5 in stopping criteria
    
    use_rule_stretching : bool, default=True
        Whether to use rule stretching for uncovered instances
    
    random_state : int, default=None
        Random seed for reproducibility
    """
    
    def __init__(self, n_folds=3, min_no=2.0, n_optimizations=2,
                 check_error_rate=True, use_rule_stretching=True,
                 random_state=None):
        self.n_folds = n_folds
        self.min_no = min_no
        self.n_optimizations = n_optimizations
        self.check_error_rate = check_error_rate
        self.use_rule_stretching = use_rule_stretching
        self.random_state = random_state
        
    def fit(self, X, y):
        """
        Fit the FURIA classifier.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : object
            Fitted estimator
        """
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]
        
        # Calculate class distribution
        self.class_distribution_ = np.bincount(y) / len(y)
        
        # Initialize random state
        self.rng_ = np.random.RandomState(self.random_state)
        
        # Build ruleset for each class (one-vs-rest)
        self.ruleset_ = []
        
        for class_idx in range(len(self.classes_)):
            class_label = self.classes_[class_idx]
            
            # Build rules for this class
            rules = self._build_ruleset_for_class(X, y, class_label)
            self.ruleset_.extend(rules)
        
        # Fuzzify all rules
        self._fuzzify_rules(X, y)
        
        # Calculate confidences
        self._calculate_confidences(X, y)
        
        return self
    
    def _build_ruleset_for_class(self, X, y, class_label):
        """Build ruleset for a single class."""
        rules = []
        data_indices = np.arange(len(X))
        
        # Building stage
        while len(data_indices) > 0:
            # Check if positive examples remain
            pos_mask = y[data_indices] == class_label
            if not np.any(pos_mask):
                break
            
            # Grow a rule
            rule = self._grow_rule(X[data_indices], y[data_indices], class_label)
            
            if rule is None or len(rule.antecedents) == 0:
                break
            
            # Check stopping criteria
            covered = np.array([rule.covers(X[i]) for i in data_indices])
            pos_covered = np.sum(covered & pos_mask)
            neg_covered = np.sum(covered & ~pos_mask)
            
            if pos_covered == 0:
                break
            
            error_rate = neg_covered / (pos_covered + neg_covered) if (pos_covered + neg_covered) > 0 else 0.5
            
            if self.check_error_rate and error_rate >= 0.5:
                break
            
            rules.append(rule)
            
            # Remove covered instances
            data_indices = data_indices[~covered]
        
        # Optimization stage (simplified)
        for _ in range(self.n_optimizations):
            rules = self._optimize_rules(X, y, rules, class_label)
        
        return rules
    
    def _grow_rule(self, X, y, class_label):
        """Grow a single rule using FOIL information gain."""
        rule = FuzzyRule(class_label)
        
        # Calculate default accuracy
        pos_count = np.sum(y == class_label)
        total_count = len(y)
        def_acc_rate = (pos_count + 1.0) / (total_count + 1.0)
        
        covered_indices = np.arange(len(X))
        used_attrs = set()
        
        while len(covered_indices) > 0 and def_acc_rate < 1.0:
            X_covered = X[covered_indices]
            y_covered = y[covered_indices]
            
            best_antd = None
            best_gain = 0.0
            best_covered = None
            
            # Try each attribute
            for attr_idx in range(X.shape[1]):
                if attr_idx in used_attrs:
                    continue
                
                # Find best split for this attribute
                antd, gain, new_covered = self._find_best_split(
                    X_covered, y_covered, attr_idx, class_label, def_acc_rate
                )
                
                if antd is not None and gain > best_gain:
                    best_gain = gain
                    best_antd = antd
                    best_covered = new_covered
            
            if best_antd is None or best_gain <= 0:
                break
            
            # Add best antecedent
            rule.add_antecedent(best_antd)
            used_attrs.add(best_antd.attr_idx)
            
            # Update covered data
            covered_indices = covered_indices[best_covered]
            
            if len(covered_indices) < self.min_no:
                break
            
            # Update accuracy rate
            pos_count = np.sum(y[covered_indices] == class_label)
            total_count = len(covered_indices)
            def_acc_rate = (pos_count + 1.0) / (total_count + 1.0)
        
        return rule if len(rule.antecedents) > 0 else None
    
    def _find_best_split(self, X, y, attr_idx, class_label, def_acc_rate):
        """Find best split point for an attribute using information gain."""
        values = X[:, attr_idx]
        
        # Remove missing values
        valid_mask = ~np.isnan(values)
        if not np.any(valid_mask):
            return None, 0.0, None
        
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        values_valid = values[valid_mask]
        
        # Sort by attribute value
        sort_idx = np.argsort(values_valid)
        values_sorted = values_valid[sort_idx]
        y_sorted = y_valid[sort_idx]
        
        best_gain = 0.0
        best_split = None
        best_value = None
        best_covered = None
        
        # Try each unique value as split point
        unique_vals = np.unique(values_sorted)
        
        for i in range(len(unique_vals)):
            split_point = unique_vals[i]
            
            # Try <= split_point (bag 0)
            mask_0 = values_valid <= split_point
            if np.sum(mask_0) >= self.min_no:
                gain, acc_rate = self._calculate_info_gain(
                    y_valid[mask_0], class_label, def_acc_rate
                )
                
                if gain > best_gain:
                    best_gain = gain
                    best_split = split_point
                    best_value = 0
                    best_covered = valid_mask.copy()
                    best_covered[valid_mask] = mask_0
            
            # Try >= split_point (bag 1)
            mask_1 = values_valid >= split_point
            if np.sum(mask_1) >= self.min_no:
                gain, acc_rate = self._calculate_info_gain(
                    y_valid[mask_1], class_label, def_acc_rate
                )
                
                if gain > best_gain:
                    best_gain = gain
                    best_split = split_point
                    best_value = 1
                    best_covered = valid_mask.copy()
                    best_covered[valid_mask] = mask_1
        
        if best_split is not None:
            # Create FS object based on split direction
            if best_value == 0:  # <= split_point
                # Trapezoidal: [-inf, -inf, split_point, split_point]
                # Initially crisp, will be fuzzified later
                membership_params = [-1e10, -1e10, best_split, best_split]
            else:  # >= split_point
                # Trapezoidal: [split_point, split_point, +inf, +inf]
                membership_params = [best_split, best_split, 1e10, 1e10]

            name = f"X{attr_idx}_{'le' if best_value == 0 else 'ge'}_{best_split:.4f}"
            fuzzy_set = fs.FS(name=name, membership_parameters=membership_params,domain=[-1e10,1e10])

            # Create FuzzyAntecedent wrapper with metadata
            antd = FuzzyAntecedent(fuzzy_set, attr_idx)
            # Store metadata for fuzzification
            antd.split_point = best_split
            antd.value = best_value
            antd.support_bound = best_split
            antd.fuzzy = False

            return antd, best_gain, best_covered
        
        return None, 0.0, None
    
    def _calculate_info_gain(self, y_subset, class_label, def_acc_rate):
        """Calculate FOIL information gain."""
        pos_count = np.sum(y_subset == class_label)
        total_count = len(y_subset)
        
        if total_count == 0:
            return 0.0, 0.0
        
        acc_rate = (pos_count + 1.0) / (total_count + 1.0)
        
        if acc_rate <= 0 or def_acc_rate <= 0:
            return 0.0, acc_rate
        
        info_gain = pos_count * (np.log2(acc_rate) - np.log2(def_acc_rate))
        
        return info_gain, acc_rate
    
    def _optimize_rules(self, X, y, rules, class_label):
        """Simplified optimization stage."""
        # In a full implementation, this would perform replacement and revision
        # For now, we just return the rules as-is
        return rules
    
    def _fuzzify_rules(self, X, y):
        """Fuzzify all rules by finding optimal support bounds."""
        for rule in self.ruleset_:
            self._fuzzify_rule(rule, X, y)
    
    def _fuzzify_rule(self, rule, X, y):
        """Fuzzify a single rule using greedy strategy."""
        n_antecedents = len(rule.antecedents)
        fuzzified = [False] * n_antecedents
        
        for iteration in range(n_antecedents):
            best_idx = -1
            best_purity = -1
            best_support = None
            
            for idx, antd in enumerate(rule.antecedents):
                # Check if it's a numeric fuzzy set (has metadata for fuzzification)
                if fuzzified[idx] or not hasattr(antd, 'split_point'):
                    continue
                
                # Get relevant data (covered by other antecedents)
                relevant_mask = np.ones(len(X), dtype=bool)
                for j, other_antd in enumerate(rule.antecedents):
                    if j != idx:
                        coverage = np.array([other_antd.membership(X[i]) for i in range(len(X))])
                        relevant_mask &= (coverage > 0)
                
                X_relevant = X[relevant_mask]
                y_relevant = y[relevant_mask]
                
                if len(X_relevant) == 0:
                    continue
                
                # Find best support bound
                support, purity = self._find_best_support_bound(
                    antd, X_relevant, y_relevant, rule.consequent
                )
                
                if purity > best_purity:
                    best_purity = purity
                    best_idx = idx
                    best_support = support
            
            if best_idx == -1:
                break
            
            # Apply best fuzzification
            if best_support is not None:
                antd = rule.antecedents[best_idx]
                antd.support_bound = best_support
                antd.fuzzy = True

                # Update the FS membership parameters
                if antd.value == 0:  # <= split_point
                    membership_params = [-1e10, -1e10, antd.split_point, best_support]
                else:  # >= split_point
                    membership_params = [best_support, antd.split_point, 1e10, 1e10]

                # Create new FS with updated parameters
                name = f"X{antd.attr_idx}_{'le' if antd.value == 0 else 'ge'}_{antd.split_point:.4f}_fuzzy"
                antd.fuzzy_set = fs.FS(name=name, membership_parameters=membership_params,domain=[-1e10,1e10])

            fuzzified[best_idx] = True
    
    def _find_best_support_bound(self, antd, X, y, class_label):
        """Find optimal support bound for a numeric antecedent."""
        values = X[:, antd.attr_idx]
        valid_mask = ~np.isnan(values)
        
        if not np.any(valid_mask):
            return antd.split_point, 0.0
        
        values_valid = values[valid_mask]
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        
        # Get candidates based on value (0 for <=, 1 for >=)
        if antd.value == 0:  # <=
            candidates = values_valid[values_valid > antd.split_point]
        else:  # >=
            candidates = values_valid[values_valid < antd.split_point]
        
        if len(candidates) == 0:
            # Trivial fuzzification
            sorted_vals = np.sort(values_valid)
            if antd.value == 0:
                idx = np.searchsorted(sorted_vals, antd.split_point, side='right')
                if idx < len(sorted_vals):
                    return sorted_vals[idx], 1.0
            else:
                idx = np.searchsorted(sorted_vals, antd.split_point, side='left')
                if idx > 0:
                    return sorted_vals[idx - 1], 1.0
            return antd.split_point, 1.0
        
        best_purity = 0.0
        best_support = antd.split_point
        
        for candidate in np.unique(candidates):
            # Create temporary fuzzified FS
            if antd.value == 0:  # <= split_point
                membership_params = [-1e10, -1e10, antd.split_point, candidate]
            else:  # >= split_point
                membership_params = [candidate, antd.split_point, 1e10, 1e10]

            temp_fuzzy_set = fs.FS(name="temp", membership_parameters=membership_params,domain=[-1e10,1e10])

            # Calculate purity
            pos_weight = 0.0
            total_weight = 0.0
            
            for i in range(len(X_valid)):
                x_val = X_valid[i, antd.attr_idx]
                coverage = float(temp_fuzzy_set.membership(np.array([x_val]))[0])
                total_weight += coverage
                if y_valid[i] == class_label:
                    pos_weight += coverage
            
            purity = pos_weight / total_weight if total_weight > 0 else 0.0
            
            if purity > best_purity or (purity == best_purity and abs(candidate - antd.split_point) > abs(best_support - antd.split_point)):
                best_purity = purity
                best_support = candidate
        
        return best_support, best_purity
    
    def _calculate_confidences(self, X, y):
        """Calculate confidence factors for all rules using m-estimate."""
        m = 2.0
        
        for rule in self.ruleset_:
            if len(rule.antecedents) == 0:
                continue
            
            # Calculate for each prefix of the rule
            for i in range(len(rule.antecedents)):
                acc = 0.0
                cov = 0.0
                
                for j in range(len(X)):
                    # Coverage by first i+1 antecedents
                    degree = 1.0
                    for k in range(i + 1):
                        degree *= rule.antecedents[k].membership(X[j])

                    cov += degree
                    if y[j] == rule.consequent:
                        acc += degree
                
                # M-estimate
                prior = self.class_distribution_[rule.consequent]
                confidence = (acc + m * prior) / (cov + m) if cov + m > 0 else 0.5
                rule.antecedents[i].confidence = confidence
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples
            
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities
        """
        check_is_fitted(self)
        X = check_array(X)
        
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        proba = np.zeros((n_samples, n_classes))
        
        for i in range(n_samples):
            instance = X[i]
            class_support = np.zeros(n_classes)
            
            # Calculate support from each rule
            for rule in self.ruleset_:
                if rule.covers(instance):
                    coverage = rule.coverage_degree(instance)
                    confidence = rule.get_confidence()
                    if not np.isnan(confidence):
                        class_support[rule.consequent] += coverage * confidence
            
            # If no rule covers, use rule stretching
            if np.sum(class_support) == 0 and self.use_rule_stretching:
                class_support = self._stretch_rules(instance)
            
            # If still no coverage, use prior
            if np.sum(class_support) == 0:
                class_support = self.class_distribution_.copy()
            
            # Handle ties with prior
            max_support = np.max(class_support)
            if max_support > 0:
                ties = class_support == max_support
                if np.sum(ties) > 1:
                    class_support = ties * self.class_distribution_
            
            # Normalize
            if np.sum(class_support) > 0:
                proba[i] = class_support / np.sum(class_support)
            else:
                proba[i] = self.class_distribution_
        
        return proba
    
    def _stretch_rules(self, instance):
        """Apply rule stretching for uncovered instance."""
        n_classes = len(self.classes_)
        class_support = np.zeros(n_classes)
        max_confidence = -np.inf
        
        for rule in self.ruleset_:
            if len(rule.antecedents) == 0:
                continue
            
            # Find first non-covering antecedent
            first_fail = len(rule.antecedents)
            for j, antd in enumerate(rule.antecedents):
                if antd.membership(instance) == 0:
                    first_fail = j
                    break
            
            if first_fail == 0:
                continue
            
            # Calculate stretched rule confidence
            remaining_antds = first_fail
            total_antds = len(rule.antecedents)
            
            # Coverage degree with remaining antecedents
            coverage = 1.0
            for j in range(remaining_antds):
                coverage *= rule.antecedents[j].membership(instance)

            # Confidence of last remaining antecedent
            base_confidence = rule.antecedents[remaining_antds - 1].confidence if remaining_antds > 0 else 0.5
            
            # Weight by fraction of antecedents remaining (with Laplace correction)
            stretch_weight = (remaining_antds + 1.0) / (total_antds + 2.0)
            
            combined_confidence = base_confidence * stretch_weight * coverage
            
            if combined_confidence >= max_confidence:
                if combined_confidence > max_confidence:
                    class_support = np.zeros(n_classes)
                    max_confidence = combined_confidence
                class_support[rule.consequent] = 1.0
        
        return class_support
    
    def predict(self, X):
        """
        Predict class labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples
            
        Returns
        -------
        y : ndarray of shape (n_samples,)
            Predicted class labels
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
    
    def __str__(self):
        """String representation of the classifier."""
        if not hasattr(self, 'ruleset_'):
            return "FURIA: No model built yet."
        
        lines = ["FURIA Rules:", "=" * 50]
        for i, rule in enumerate(self.ruleset_):
            conf = rule.get_confidence()
            conf_str = f"{conf:.2f}" if not np.isnan(conf) else "N/A"
            lines.append(f"Rule {i+1}: {rule} (CF={conf_str})")
        lines.append(f"\nTotal rules: {len(self.ruleset_)}")
        return "\n".join(lines)


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    
    # Load data
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train FURIA
    furia = FURIA(n_folds=3, n_optimizations=2, random_state=42)
    furia.fit(X_train, y_train)
    
    # Predict
    y_pred = furia.predict(X_test)
    
    # Evaluate
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\n" + str(furia))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
