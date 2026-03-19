def create_features(df):

    # Family size
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    # Is alone
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Fare per person
    df["FarePerPerson"] = df["Fare"] / df["FamilySize"]

    return df
