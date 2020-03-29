import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# visualise distribution
def plot_distribution(data, feature):
    sns.set_style("ticks")
    s = sns.FacetGrid(data, hue="churn", aspect=2.5, palette={1: 'Lightblue', 0: "pink"})
    s.map(sns.kdeplot, feature, shade=True, alpha=0.8)
    s.set(xlim=(0, data[feature].max()))
    s.add_legend()
    s.set_axis_labels(feature, "Proportion")
    plt.show()


def categorical_visualisation(data, feature, churn_col_name="churn"):
    index_value = data[feature].unique().tolist()

    churn = []
    stay = []

    for value in index_value:
        col_data = data.loc[data[feature] == value][churn_col_name]

        churn.append(sum(col_data))
        stay.append(len(col_data) - sum(col_data))

    new_df = pd.DataFrame({"churn": churn, "stay": stay}, index=index_value)

    new_df.plot.bar(rot=0)

    plt.title("Churn rate by " + feature)
    plt.show()