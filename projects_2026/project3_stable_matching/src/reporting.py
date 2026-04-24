from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from math import pi

def summary_to_row(task, model, instances, summary):
    return {
        "Task": task,
        "Model": model,
        "Instances": instances,
        "Parsed_Count": summary.get("parsed_count"),
        "Valid_Count": summary.get("valid_count"),
        "Stable_Count": summary.get("stable_count"),
        "Exact_Match_Count": summary.get("exact_match_count"),
        "Correct_Count": summary.get("correct_count"),
        "Accuracy": summary.get("accuracy"),
        "Avg_Blocking_Pairs": summary.get("avg_blocking_pairs"),
        "Total_Instances": summary.get("total_instances")
    }

def build_final_summary_table(summary_map):
    rows = []
    # Build a much easier, standardized comparative table
    for (task, model, instances), summary in summary_map.items():
        if task == "Task 1":
            task_name = "Generation"
            metric = "Stability Score"
            score = (summary.get("stable_count", 0) / summary.get("total_instances", 1)) * 100
        elif task == "Task 2":
            task_name = "Detection"
            metric = "Accuracy Score"
            score = summary.get("accuracy", 0) * 100
        elif task == "Task 3":
            task_name = "Resolution"
            metric = "Stability Score"
            score = (summary.get("stable_count", 0) / summary.get("total_instances", 1)) * 100
        elif task == "Task 4":
            task_name = "Reasoning"
            metric = "Accuracy Score"
            score = summary.get("accuracy", 0) * 100
        else:
            continue
            
        rows.append({
            "Task": task_name,
            "Instances": instances,
            "Model": model,
            "Metric": metric,
            "Performance": f"{score:.0f}%"
        })
        
    if not rows: return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    # Pivot so Basic and Reasoning are side-by-side columns matching the user's layout
    pivot_df = df.pivot_table(index=["Task", "Metric", "Instances"], columns="Model", values="Performance", aggfunc='first')
    
    # Flatten multi-index columns safely
    if isinstance(pivot_df.columns, pd.MultiIndex):
        pivot_df.columns = [col[1] if isinstance(col, tuple) else col for col in pivot_df.columns]
        
    pivot_df = pivot_df.reset_index()
    
    # Order Tasks natively
    task_order = ["Generation", "Detection", "Resolution", "Reasoning"]
    pivot_df["Task"] = pd.Categorical(pivot_df["Task"], categories=task_order, ordered=True)
    pivot_df = pivot_df.sort_values(by=["Task", "Instances"]).reset_index(drop=True)
    
    return pivot_df


def plot_radar_chart(chart_df, save_path=None):
    if chart_df.empty: return None
    # We will plot the 20 Instance variants for the radar chart to represent maximum logical breadth
    df20 = chart_df[chart_df["Instances"] == 20].copy()
    if df20.empty:
        # Fall back to whatever is available if 20 wasn't run
        df20 = chart_df.copy()
        
    pivot_df = df20.pivot(index="Task", columns="Model", values="Score").fillna(0)
    
    # Map the shortened keys back to the full labels for the Radar chart
    rename_map = {"Gen (20)": "Generation", "Det (20)": "Detection", "Res (20)": "Resolution", "Reason (20)": "Reasoning"}
    pivot_df = pivot_df.rename(index=rename_map)
    
    categories = ['Generation', 'Detection', 'Resolution', 'Reasoning']
    for cat in categories:
        if cat not in pivot_df.index:
            pivot_df.loc[cat] = 0
            
    pivot_df = pivot_df.reindex(categories)
    
    if "Basic" not in pivot_df.columns: pivot_df["Basic"] = 0
    if "Reasoning" not in pivot_df.columns: pivot_df["Reasoning"] = 0
    
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], categories, size=12, fontweight="bold")
    ax.set_rlabel_position(0)
    plt.yticks([25, 50, 75, 100], ["25%","50%","75%","100%"], color="grey", size=10)
    plt.ylim(0, 100)
    
    # Basic
    values_basic = pivot_df["Basic"].values.flatten().tolist()
    values_basic += values_basic[:1]
    ax.plot(angles, values_basic, linewidth=2, linestyle='solid', label='Basic (Llama-3.1-8b)', color='#ff9999')
    ax.fill(angles, values_basic, '#ff9999', alpha=0.25)
    
    # Reasoning
    values_reason = pivot_df["Reasoning"].values.flatten().tolist()
    values_reason += values_reason[:1]
    ax.plot(angles, values_reason, linewidth=2, linestyle='solid', label='Reasoning (Llama-3.3-70b)', color='#66b3ff')
    ax.fill(angles, values_reason, '#66b3ff', alpha=0.25)
    
    plt.title("Performance Multi-Dimensional Radar (20 Instances)", size=16, fontweight="bold", y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    return fig

def plot_scaling_degradation(chart_df, save_path=None):
    if chart_df.empty: return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract root task name backwards from the Instance strings (e.g. 'Gen (10)' -> 'Gen')
    plot_df = chart_df.copy()
    plot_df["BaseTask"] = plot_df["Task"].apply(lambda x: x.split(" (")[0])
    
    tasks = ['Gen', 'Det', 'Res', 'Reason']
    colors = {'Basic': '#ff9999', 'Reasoning': '#66b3ff'}
    markers = {'Basic': 'o', 'Reasoning': 's'}
    
    for task in tasks:
        task_data = plot_df[plot_df["BaseTask"] == task]
        if task_data.empty: continue
        
        for model in ['Basic', 'Reasoning']:
            model_data = task_data[task_data["Model"] == model]
            if model_data.empty: continue
                
            model_data = model_data.sort_values(by="Instances")
            
            ax.plot(
                model_data["Instances"].astype(str), 
                model_data["Score"], 
                marker=markers[model],
                color=colors[model],
                linestyle='--' if model == 'Basic' else '-',
                linewidth=2,
                label=f"{model} ({task})"
            )
            
    ax.set_title("Performance Scaling Degradation Slope (10 vs 20 Instances)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Performance (%)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Instances Evaluated", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Model Profiles", prop={'size': 9})
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    return fig

# Backward compat
plot_grouped_bar_chart = plot_radar_chart
plot_exact_match_chart = plot_scaling_degradation

def plot_blocking_pairs_chart(summary_map, save_path=None):
    chart_rows = []
    for (task, model, instances), summary in summary_map.items():
        if task not in ["Task 1", "Task 3"]: continue
        model = str(model).strip()
        task_name = f"Gen ({instances})" if task == "Task 1" else f"Res ({instances})"
        score = summary.get("avg_blocking_pairs", 0)
        chart_rows.append({"Task": task_name, "Model": model, "Score": score})
        
    if not chart_rows: return
    df = pd.DataFrame(chart_rows)
    df["Task"] = pd.Categorical(df["Task"], categories=["Gen (10)", "Gen (20)", "Res (10)", "Res (20)"], ordered=True)
    pivot_df = df.pivot(index="Task", columns="Model", values="Score").fillna(0)
    
    if "Basic" not in pivot_df.columns: pivot_df["Basic"] = 0
    if "Reasoning" not in pivot_df.columns: pivot_df["Reasoning"] = 0
    pivot_df = pivot_df[["Basic", "Reasoning"]]
    x = np.arange(len(pivot_df.index))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, pivot_df["Basic"], width, label="Basic", color='#ff9999', edgecolor="black")
    bars2 = ax.bar(x + width/2, pivot_df["Reasoning"], width, label="Reasoning", color='#66b3ff', edgecolor="black")
    
    ax.set_title("Average Blocking Pairs (Lower is More Stable)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Avg Blocking Pairs", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_df.index, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9)
            
    plt.tight_layout()
    if save_path: fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    return fig


def plot_grouped_bar_chart(chart_df, save_path=None):
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    if chart_df.empty: return None

    pivot_df = chart_df.pivot(index="Task", columns="Model", values="Score").fillna(0)

    if "Basic" not in pivot_df.columns:
        pivot_df["Basic"] = 0
    if "Reasoning" not in pivot_df.columns:
        pivot_df["Reasoning"] = 0

    pivot_df = pivot_df[["Basic", "Reasoning"]]
    x = np.arange(len(pivot_df.index))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width/2, pivot_df["Basic"], width, label="Basic", edgecolor="black")
    bars2 = ax.bar(x + width/2, pivot_df["Reasoning"], width, label="Reasoning", edgecolor="black")

    ax.set_title("Performance Comparison Across Tasks", fontsize=14, fontweight="bold")
    ax.set_ylabel("Performance Score (%)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_df.index, fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    return fig


def build_chart_dataframe(summary_map):
    import pandas as pd
    chart_rows = []
    
    for (task, model, instances), summary in summary_map.items():
        if instances != 20: continue
            
        model = str(model).strip()

        if task == "Task 1":
            score = (summary.get("stable_count", 0) / summary.get("total_instances", 1)) * 100
        elif task == "Task 2":
            score = summary.get("accuracy", 0) * 100
        elif task == "Task 3":
            score = (summary.get("stable_count", 0) / summary.get("total_instances", 1)) * 100
        elif task == "Task 4":
            score = summary.get("accuracy", 0) * 100
        else:
            continue

        chart_rows.append({
            "Task": task,  # Literally just "Task 1", "Task 2"
            "Model": model,
            "Score": score
        })

    if not chart_rows:
        return pd.DataFrame()
        
    df = pd.DataFrame(chart_rows)
    
    task_order = ["Task 1", "Task 2", "Task 3", "Task 4"]
    df["Task"] = pd.Categorical(df["Task"], categories=task_order, ordered=True)
    df = df.sort_values(by=["Task", "Model"]).reset_index(drop=True)
    return df
