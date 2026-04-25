import matplotlib.pyplot as plt
import seaborn as sns

def plot_emb_results(results, dataset):
    models = results['model'].unique()
    colors = sns.color_palette("hls", n_colors=len(models))
    palette = dict(zip(models, colors))
    
    summary = (
        results.groupby("model", as_index=False)['f1-score']
        .median()
        .sort_values("f1-score", ascending=False)
    )

    plt.figure(figsize=(12,6))
    ax = sns.barplot(
        data=summary,
        x='model',
        y='f1-score',
        palette=palette,
        legend=False,
        hue='model'
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=2, label_type='center',fontweight='bold')
    plt.title(f"{dataset} | Model performance (Median F1)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(12,6))
    sns.boxplot(
        data=results,
        x='model',
        y='f1-score',
        palette=palette,
        hue='model',
        legend=False
    )
    plt.title(f"{dataset} | Model F1-score distributions")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
           