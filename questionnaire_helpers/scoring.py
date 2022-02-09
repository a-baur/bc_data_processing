from collections import defaultdict
import pandas as pd

## Scoring keys -> {question_1: [(category_1, to_be_reversed), (category_2, to_be_reversed), ...], ...}

# scoring key for bfi-s
sk_bfi = {
    "Worries a lot": [("Neuroticism", "")],
    "Gets nervous easily": [("Neuroticism", "")],
    "Remains calm in tense situations": [("Neuroticism", "R")],
    "Is talkative": [("Extraversion", "")],
    "Is outgoing, sociable": [("Extraversion", "")],
    "Is reserved": [("Extraversion", "R")],
    "Is original, comes up with new ideas": [("Openness", "")],
    "Values artistic, aesthetic experiences": [("Openness", "")],
    "Has an active imagination": [("Openness", "")],
    "Is sometimes rude to others": [("Agreeableness", "R")],
    "Has a forgiving nature": [("Agreeableness", "")],
    "Is considerate and kind to almost everyone": [("Agreeableness", "")],
    "Does a thorough job": [("Conscientiousness", "")],
    "Tends to be lazy": [("Conscientiousness", "R")],
    "Does things efficiently": [("Conscientiousness", "")]
}

sk_ssq = {
    "General discomfort": [("Nausea", ""), ("Oculomotor", ""), ("Total", "")],
    "Fatigue": [("Oculomotor", ""), ("Total", "")],
    "Headache": [("Oculomotor", ""), ("Total", "")],
    "Eye strain": [("Oculomotor", ""), ("Total", "")],
    "Difficulty focusing": [("Disorientation", ""), ("Oculomotor", ""), ("Total", "")],
    "Sweating": [("Nausea", ""), ("Total", "")],
    "Nausea": [("Disorientation", ""), ("Nausea", ""), ("Total", "")],
    "Difficulty concentrating": [("Nausea", ""), ("Oculomotor", ""), ("Total", "")],
    "Fullness of head": [("Disorientation", ""), ("Total", "")],
    "Blurred vision": [("Disorientation", ""), ("Oculomotor", ""), ("Total", "")],
    "Dizzy (eyes open)": [("Disorientation", ""), ("Total", "")],
    "Dizzy (eyes closed)": [("Disorientation", ""), ("Total", "")],
    "Vertigo": [("Disorientation", ""), ("Total", "")],
    "Stomach awareness": [("Nausea", ""), ("Total", "")],
    "BurpingPost-Negotiation": [("Nausea", ""), ("Total", "")]  # LimeSurvey bug inserts 'Post-Negotiation'
}

sk_iqq = {
    "I think my negotiation partner is very unreliable.": [("Trust", "R")],
    "I think the interaction with my negotiation partner was smooth.": [("Smoothness", "")],
    "I think my negotiation partner is very unpleasant.": [("Rapport", "R")],
    "I think my negotiation partner is very similar to me.": [("Similarity", "")],
    "I think my negotiation partner is very likeable.": [("Rapport", "")],
    "I think my negotiation partner is very insincere.": [("Trust", "R")],
    "I think my negotiation partner is very responsible.": [("Trust", "")],
    "I think my negotiation partner is very engaging.": [("Rapport", "")],
    "I think my negotiation partner is very unfriendly.": [("Rapport", "R")],
    "I think my negotiation partner is very trustworthy.": [("Trust", "")],
    "I think my negotiation partner is very kind.": [("Rapport", "")],
    "I think the interaction with my negotiation partner was awkward.": [("Smoothness", "R")],
    "I think my negotiation partner is very different from me.": [("Similarity", "R")],
    "I think my negotiation partner is very honest.": [("Trust", "")]
}

def score(s, scoring_key, max_score, str_fmt="{}", ):
    """ Score series by scoring key.
    :param s: numpy.Series with index being the questions and values the answers.
    :param scoring_key: Mapping of questions to tuples with (category, to_be_reversed).
    :param max_score: Maximum possible score for inversion of scores.
    :param str_fmt: Format string if answers are embedded in it
                    (e.g. "I am someone who {}" with {} containing actual scored question)
    :return numpy.Series with category as index and score as value
    """
    result = defaultdict(int)
    for col, categories in scoring_key.items():
        col = str_fmt.format(col)
        for cat, reverse in categories:
            if reverse == 'R':
                result[cat] += max_score - s.loc[col]
            else:
                result[cat] += s.loc[col]
    return pd.Series(result)

def score_df(df, scoring_key, max_score, str_fmt="{}"):
    """ Score series by scoring key.
    :param df: pandas.DataFrame with questions as columns and the values as answers.
    :param scoring_key: Mapping of questions to tuples with (category, to_be_reversed).
    :param max_score: Maximum possible score for inversion of scores.
    :param str_fmt: Format string if answers are embedded in it
                    (e.g. "I am someone who {}" with {} containing actual scored question)
    :return pandas.DataFrame with category as column and scores as value.
    """
    df_res = pd.DataFrame()
    for subj, row in df.iterrows():
        scores = score(row, scoring_key, max_score, str_fmt)
        scores.name = subj
        df_res = df_res.append(scores)
    return df_res