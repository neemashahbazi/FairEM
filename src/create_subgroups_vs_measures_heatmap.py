import matplotlib.pyplot as plt
import numpy as np

def make_acronym(word, delim):
    res = ""
    for spl in word.split(delim):
        res += spl[0].capitalize()
    return res


def plot_results_in_2d_heatmap(dataset, data, xlabels, ylabels, title, figsize = (10,5), 
                                x_font = 11, y_font = 12, vmin=-1.0, vmax=0.6, cmap="RdYlGn"):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)

    fig.colorbar(im, shrink=0.5)
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(labels=xlabels, fontdict={'fontsize': x_font})
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(labels=ylabels, fontdict={'fontsize': y_font})

    plt.setp(ax.get_xticklabels(), rotation=60, ha="right",
            rotation_mode="anchor")

    thr = round((vmin + vmax)/2, 2)
    print("thr = ", thr)
    for i in range(len(data)):
        for j in range(len(data[0])):
            if abs(data[i][j] - thr) > 0.2:
                color = "w"
            else:
                color = "black"
            text = ax.text(j, i, round(data[i][j],2),
                        ha="center", va="center", color=color)
            
    # ax.set_title(title, fontdict={'fontsize': 24})
    fig.tight_layout()
    plt.savefig("../experiments/" + dataset + "/" + title.replace("\n","") + ".png", dpi=100)
    plt.close()

attributes =  ['Dance', 'Electronic', 'Music', 'Pop', 'R&B / Soul', 'Rock', 'other', 'Contemporary Country', 'Country', 'Dance & Electronic', 'Alternative', 'Hip-Hop/Rap', 'Rap', 'Rap & Hip-Hop', 'East Coast Rap', 'Hardcore Rap', 'House']
measures =  ['accuracy_parity',
             'statistical_parity',
             'true_positive_rate_parity',
             'false_positive_rate_parity',
             'false_negative_rate_parity',
             'true_negative_rate_parity',
             'negative_predictive_value_parity',
             'false_discovery_rate_parity',
             'false_omission_rate_parity']

actual_fairness =  [[0.00523349436392917, -0.021164021164021163, 0.0, -0.16517323775388293, -0.34259259259259256, -0.002592592592592591, -0.04894179894179895, -0.15028490028490027, -0.19973544973544965, 0.036195286195286225, 0.15740740740740744, 0.15740740740740744, 0.1174074074074074, 0.15740740740740744, 0.15740740740740744, 0.15740740740740744, -0.05687830687830686], [-0.023349436392914646, -0.06216931216931215, 0.0, 0.04958183990442058, -0.040740740740740716, 0.0392592592592593, 0.0291005291005291, 0.29772079772079774, 0.2592592592592593, -0.0286195286195286, 0.09259259259259259, -0.08074074074074072, -0.08074074074074072, -0.09259259259259259, -0.08689458689458687, -0.1074074074074074, -0.09788359788359788], [-0.005376344086021501, -0.005376344086021501, 0.0, -0.08870967741935487, -0.33870967741935487, 0.03629032258064513, -0.06598240469208216, 0.03629032258064513, 0.03629032258064513, 0.03629032258064513, 0.16129032258064513, 0.16129032258064513, 0.16129032258064513, 0.16129032258064513, 0.16129032258064513, 0.16129032258064513, -0.17204301075268824], [-0.008785332314744071, 0.025974025974025983, 0.0, 0.2125768967874231, 0.34415584415584416, 0.02062643239113829, 0.03927779537535636, 0.44415584415584414, 0.5108225108225108, -0.035844155844155845, -0.15584415584415584, -0.15584415584415584, -0.10822510822510822, -0.15584415584415584, -0.15584415584415584, -0.15584415584415584, 0.025974025974025983], [0.005376344086021501, 0.005376344086021501, 0.0, 0.08870967741935484, 0.33870967741935487, -0.03629032258064516, 0.06598240469208211, -0.03629032258064516, -0.03629032258064516, -0.03629032258064516, -0.16129032258064516, -0.16129032258064516, -0.16129032258064516, -0.16129032258064516, -0.16129032258064516, -0.16129032258064516, 0.17204301075268816], [0.008785332314744099, -0.02597402597402587, 0.0, -0.2125768967874231, -0.3441558441558441, -0.02062643239113826, -0.03927779537535625, -0.4441558441558441, -0.5108225108225108, 0.0358441558441559, 0.1558441558441559, 0.1558441558441559, 0.10822510822510822, 0.1558441558441559, 0.1558441558441559, 0.1558441558441559, -0.02597402597402587], [0.006912442396313279, 0.018796992481202923, 0.0, -0.12857142857142856, -0.3285714285714286, 0.004761904761904745, -0.06015037593984962, -0.261904761904762, -0.261904761904762, 0.02795031055900621, 0.0714285714285714, 0.0714285714285714, 0.0714285714285714, 0.0714285714285714, 0.0714285714285714, 0.0714285714285714, -0.02857142857142858], [0.017543859649122806, 0.1286549707602339, 0.0, 0.12171052631578949, 0.28421052631578947, -0.01578947368421052, 0.004210526315789498, -0.01578947368421052, 0.04784688995215314, -0.01578947368421052, -0.3157894736842105, -0.3157894736842105, -0.1157894736842105, -0.3157894736842105, -0.3157894736842105, -0.3157894736842105, 0.1842105263157895], [-0.006912442396313362, -0.018796992481203006, 0.0, 0.1285714285714286, 0.3285714285714286, -0.004761904761904759, 0.06015037593984962, 0.26190476190476186, 0.26190476190476186, -0.02795031055900621, -0.07142857142857142, -0.07142857142857142, -0.07142857142857142, -0.07142857142857142, -0.07142857142857142, -0.07142857142857142, 0.02857142857142858]]

good_measures = ["accuracy_parity",
            "statistical_parity",
            "true_positive_rate_parity",
            "true_negative_rate_parity",
            "negative_predictive_value_parity"]

bad_measures = ["false_positive_rate_parity",
            "false_negative_rate_parity",
            "false_discovery_rate_parity",
            "false_omission_rate_parity"]


good_measures_acronymes = [make_acronym(x, "_") for x in good_measures]
bad_measures_acronymes = [make_acronym(x, "_") for x in bad_measures]
good_measures_fairness = []
bad_measures_fairness = []


for i in range(len(measures)):
    measure = measures[i]
    if measure in set(good_measures):
        good_measures_fairness.append(actual_fairness[i])
    else:
        bad_measures_fairness.append(actual_fairness[i])
    

plot_results_in_2d_heatmap("itunes-amazon", good_measures_fairness, 
                            attributes, good_measures_acronymes,
                            "Actual Fairness for Positive Measures",
                            vmin=-0.7, vmax=0.3, cmap="RdYlGn")

plot_results_in_2d_heatmap("itunes-amazon", bad_measures_fairness, 
                            attributes, bad_measures_acronymes,
                            "Actual Fairness for Negative Measures",
                            vmin=-0.3, vmax=0.7, cmap="RdYlGn_r")