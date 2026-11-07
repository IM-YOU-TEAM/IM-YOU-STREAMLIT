import numpy as np


def _first_matching_column(row, values):
    """Return the first column index whose value is in `values` (NumPy 2-safe)."""
    matches = np.flatnonzero(np.isin(row, values))
    if len(matches) == 0:
        raise ValueError(f'No column matched any of: {values}')
    return int(matches[0])


def get_activity_index_test(test_labels, activity):
    activity_index = []
    activity_column = _first_matching_column(
        test_labels[0][0],
        ['levelground1', 'levelground2', 'stairascent', 'stairdescent', 'rampascent', 'rampdescent'],
    )
    for i, label in enumerate(test_labels):
        if np.all(label[:, activity_column] == activity):
            activity_index.append(i)
    return activity_index


def get_exact_activity_index_test(test_labels, exact_activity):
    exact_activity_column = 1
    exact_activity_index = []
    for i, label in enumerate(test_labels):
        if np.all(label[:, exact_activity_column] == exact_activity):
            exact_activity_index.append(i)
    return exact_activity_index


def get_subject_index_test(test_labels, subject):
    subject_index = []
    subject_column = _first_matching_column(
        test_labels[0][0],
        ["AB06", "AB07", "AB08", "AB09", "AB10", "AB11", "AB12", "AB13", "AB14", "AB15",
         "AB16", "AB17", "AB18", "AB19", "AB20", "AB21", "AB23", "AB24", "AB25"],
    )
    for i, label in enumerate(test_labels):
        if np.all(label[:, subject_column] == subject):
            subject_index.append(i)
    return subject_index


def get_model_name_from_activites(train_activity, test_activity):
    original_name = {"levelground1": "g1", "levelground2": "g2",
                     "rampascent": "ra", "rampdescent": "rd",
                     "stairascent": "sa", "stairdescent": "sd"}
    model_train_activity = []
    for activity in train_activity:
        model_train_activity.append(original_name[activity])
    model_test_activity = []
    for activity in test_activity:
        model_test_activity.append(original_name[activity])
    return model_train_activity, model_test_activity