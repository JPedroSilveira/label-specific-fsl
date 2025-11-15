import numpy as np
from sklearn.utils.validation import check_random_state
from sklearn.utils import shuffle as util_shuffle
from sklearn.utils.random import sample_without_replacement

"""
Generate samples of synthetic data sets.
"""

# Adapted from authors: B. Thirion, G. Varoquaux, A. Gramfort, V. Michel, O. Grisel,
#          G. Louppe, J. Nothman
# License: BSD 3 clause
def _custom_generate_hypercube(samples, dimensions, rng):
    """Returns distinct binary samples of length dimensions."""
    if dimensions > 30:
        return np.hstack(
            [
                rng.randint(2, size=(samples, dimensions - 30)),
                _custom_generate_hypercube(samples, 30, rng),
            ]
        )
    out = sample_without_replacement(2**dimensions, samples, random_state=rng).astype(
        dtype=">u4", copy=False
    )
    out = np.unpackbits(out.view(">u1")).reshape((-1, 32))[:, -dimensions:]
    return out

def custom_make_classification(
    n_samples=100,
    n_features=20,
    *,
    n_informative=2,
    n_informative_features_per_class=2,
    n_redundant=0,
    n_repeated=0,
    n_classes=2,
    n_clusters_per_class=2,
    weights=None,
    flip_y=0.01,
    class_sep=1.0,
    hypercube=True,
    shift=0.0,
    scale=1.0,
    shuffle=True,
    random_state=None,
):
    generator = check_random_state(random_state)

    # Count features, clusters and samples
    if n_informative + n_redundant + n_repeated > n_features:
        raise ValueError(
            "Number of informative, redundant and repeated "
            "features must sum to less than the number of total"
            " features"
        )
    # Use log2 to avoid overflow errors
    if n_informative < np.log2(n_classes * n_clusters_per_class):
        msg = "n_classes({}) * n_clusters_per_class({}) must be"
        msg += " smaller or equal 2**n_informative({})={}"
        raise ValueError(
            msg.format(n_classes, n_clusters_per_class, n_informative, 2**n_informative)
        )

    if weights is not None:
        if len(weights) not in [n_classes, n_classes - 1]:
            raise ValueError(
                "Weights specified but incompatible with number of classes."
            )
        if len(weights) == n_classes - 1:
            if isinstance(weights, list):
                weights = weights + [1.0 - sum(weights)]
            else:
                weights = np.resize(weights, n_classes)
                weights[-1] = 1.0 - sum(weights[:-1])
    else:
        weights = [1.0 / n_classes] * n_classes

    n_useless = n_features - n_informative - n_redundant - n_repeated
    n_clusters = n_classes * n_clusters_per_class

    # Distribute samples among clusters by weight
    n_samples_per_cluster = [
        int(n_samples * weights[k % n_classes] / n_clusters_per_class)
        for k in range(n_clusters)
    ]

    for i in range(n_samples - sum(n_samples_per_cluster)):
        n_samples_per_cluster[i % n_clusters] += 1

    # Initialize X and y
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples, dtype=int)

    # Build the polytope whose vertices become cluster centroids
    centroids = _custom_generate_hypercube(n_clusters, n_informative, generator).astype(
        float, copy=False
    )

    centroids *= 2 * class_sep
    centroids -= class_sep
    if not hypercube:
        centroids *= generator.uniform(size=(n_clusters, 1))
        centroids *= generator.uniform(size=(1, n_informative))

    # Initially draw informative features from the standard normal
    X[:, :n_informative] = generator.standard_normal(size=(n_samples, n_informative))

    feature_generator = check_random_state(random_state)

    # Define features per class
    features_by_class = []
    non_used_features = list(range(0, n_informative))
    features_classes = {}

    for c in range(0, n_classes):
        features = sample_without_replacement(n_informative, n_informative_features_per_class, random_state=feature_generator)
        for feature in features:
            if feature in non_used_features:
                non_used_features.remove(feature)
            if feature not in features_classes:
                features_classes[feature] = [c]
            else:
                features_classes[feature].append(c)
        features_by_class.append(features)
    for non_used_feature in non_used_features:
        added = False
        for feature, classes in features_classes.items():
            if len(classes) > 1:
                class_to_add = classes[0]
                class_features = features_by_class[class_to_add]
                class_features = class_features[class_features != feature]
                class_features = np.insert(class_features, 0, non_used_feature)
                features_by_class[class_to_add] = class_features
                classes.remove(class_to_add)
                features_classes[feature] = classes
                features_classes[non_used_feature] = [class_to_add]
                added = True
                break
        if not added:
            raise ValueError("Fail to use all informative features, number of feature and features per class don't match")
    for c, features in enumerate(features_by_class):
        features.sort()

    # Create each cluster; a variant of make_blobs
    stop = 0
    for k, centroid in enumerate(centroids):
        # pdb.set_trace()
        current_class = k % n_classes
        class_features = features_by_class[current_class]
        start, stop = stop, stop + n_samples_per_cluster[k]
        number_of_samples = stop - start
        y[start:stop] = current_class  # assign labels

        A = 2 * generator.uniform(size=(n_informative_features_per_class, n_informative_features_per_class)) - 1
        X[start:stop, class_features] = np.dot(X[start:stop, class_features], A)  # introduce random covariance
        X[start:stop, class_features] += centroid[class_features]  # shift the cluster to a vertex

        # Set values for useless features given the class
        useless_features = [i for i in range(0, n_informative) if i not in class_features]
        X[start:stop, useless_features] = generator.standard_normal(size=(number_of_samples, len(useless_features)))


    # Create redundant features
    if n_redundant > 0:
        B = 2 * generator.uniform(size=(n_informative, n_redundant)) - 1
        X[:, n_informative : n_informative + n_redundant] = np.dot(
            X[:, :n_informative], B
        )

    # Repeat some features
    if n_repeated > 0:
        n = n_informative + n_redundant
        indices = ((n - 1) * generator.uniform(size=n_repeated) + 0.5).astype(np.intp)
        X[:, n : n + n_repeated] = X[:, indices]

    # Fill useless features
    if n_useless > 0:
        X[:, -n_useless:] = generator.standard_normal(size=(n_samples, n_useless))

    # Randomly replace labels
    if flip_y >= 0.0:
        flip_mask = generator.uniform(size=n_samples) < flip_y
        y[flip_mask] = generator.randint(n_classes, size=flip_mask.sum())

    # Randomly shift and scale
    if shift is None:
        shift = (2 * generator.uniform(size=n_features) - 1) * class_sep
    X += shift

    if scale is None:
        scale = 1 + 100 * generator.uniform(size=n_features)
    X *= scale

    if shuffle:
        # Randomly permute samples
        X, y = util_shuffle(X, y, random_state=generator)

        # Randomly permute features
        indices = np.arange(n_features)
        generator.shuffle(indices)
        X[:, :] = X[:, indices]

    return X, y, features_by_class