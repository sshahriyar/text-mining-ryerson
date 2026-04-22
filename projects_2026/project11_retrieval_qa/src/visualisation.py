"""
visualisation.py
================
All figures for the project:

  1. plot_pipeline()             — RAG pipeline diagram
  2. plot_results()              — 3-strategy baseline chart
  3. plot_results_with_taare()   — 4-strategy chart including TA-ARE
  4. plot_retrieval_accuracy()   — TA-ARE retrieval decision accuracy
  5. plot_paper_architecture()   — Original paper architecture (3-part)

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


NAVY   = '#0D1B2A'
TEAL   = '#0D9488'
PURPLE = '#7C3AED'
AMBER  = '#B45309'
GREEN  = '#059669'
BLUE   = '#1D4ED8'
GRAY   = '#6B7280'
DKGRAY = '#374151'
WHITE  = '#FFFFFF'


# ── 1. RAG PIPELINE DIAGRAM ──────────────────────────────────────────────────

def plot_pipeline(save_path='../diagram/rag_pipeline.png'):
    """Always Retrieval pipeline — 5 sequential steps."""
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')

    boxes = [
        ("User Question",    "",                                               9.2, NAVY),
        ("Step 1: Encode",   "SentenceTransformers\nall-MiniLM-L6-v2 -> 384-dim vector", 7.5, TEAL),
        ("Step 2: Search",   "FAISS IndexFlatIP\nCosine similarity -> top-3 passages",   5.9, PURPLE),
        ("Step 3: Augment",  "Prompt Builder\nContext + Question -> enriched prompt",     4.3, AMBER),
        ("Step 4: Generate", "GPT-3.5-turbo (OpenAI API)\nPrompt -> short answer",        2.7, GREEN),
        ("Step 5: Evaluate", "Exact Match + Token F1\nPrediction vs ground truth",        1.1, NAVY),
    ]

    for title, subtitle, yc, color in boxes:
        ax.add_patch(FancyBboxPatch((2.5, yc-0.55), 5, 1.0,
                                    boxstyle="round,pad=0.1",
                                    facecolor=color, edgecolor=WHITE,
                                    linewidth=2, zorder=3))
        if subtitle:
            ax.text(5, yc-0.05, title, ha='center', va='center',
                    fontsize=11, fontweight='bold', color=WHITE, zorder=4)
            ax.text(5, yc-0.38, subtitle, ha='center', va='center',
                    fontsize=8, color=WHITE, alpha=0.9, zorder=4)
        else:
            ax.text(5, yc, title, ha='center', va='center',
                    fontsize=12, fontweight='bold', color=WHITE, zorder=4)

    for y_start, y_end in [(8.75,8.05),(7.00,6.45),(5.45,4.85),(3.85,3.25),(2.25,1.65)]:
        ax.annotate('', xy=(5, y_end), xytext=(5, y_start),
                    arrowprops=dict(arrowstyle='->', color=TEAL, lw=2.5))

    ax.text(5, 9.8, "Always Retrieval — RAG Pipeline (GPT-3.5-turbo)",
            ha='center', fontsize=13, fontweight='bold', color=NAVY)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    print("Pipeline diagram saved as " + save_path)


# ── 2. BASELINE 3-STRATEGY CHART ─────────────────────────────────────────────

def plot_results(results_df, save_path='../diagram/results_chart.png'):
    """Side-by-side bar chart for 3 baseline strategies."""
    strategies = ['No Retrieval', 'Always Retrieval', 'Adaptive (Oracle)']
    em = [results_df['em_no_ret'].mean(), results_df['em_always'].mean(), results_df['em_adaptive'].mean()]
    f1 = [results_df['f1_no_ret'].mean(), results_df['f1_always'].mean(), results_df['f1_adaptive'].mean()]

    ret_df = results_df[results_df['needs_retrieval'] == 0]
    em_r = [ret_df['em_no_ret'].mean(), ret_df['em_always'].mean(), ret_df['em_adaptive'].mean()]
    f1_r = [ret_df['f1_no_ret'].mean(), ret_df['f1_always'].mean(), ret_df['f1_adaptive'].mean()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('RAG Baseline Strategy Comparison — GPT-3.5-turbo\n'
                 'Nishi Patel (501356244)  ·  Avi Patel (501376903)',
                 fontsize=13, fontweight='bold', color=NAVY, y=1.02)

    x, w = np.arange(3), 0.35
    colors_em = ['#94A3B8', '#0D9488', '#7C3AED']
    colors_f1 = ['#CBD5E1', '#5DCAA5', '#AFA9EC']

    for ax, e, f, title in [
        (axes[0], em, f1, f'Overall ({len(results_df)} questions)'),
        (axes[1], em_r, f1_r, f'Retrieval-Needed (n={len(ret_df)})')
    ]:
        b1 = ax.bar(x-w/2, e, w, label='Exact Match', color=colors_em, edgecolor='white', linewidth=1.5)
        b2 = ax.bar(x+w/2, f, w, label='Token F1',    color=colors_f1, edgecolor='white', linewidth=1.5)
        for bar in b1:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.008,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold', color=NAVY)
        for bar in b2:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.008,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9, color=DKGRAY)
        ax.set_title(title, fontsize=12, fontweight='bold', color=NAVY, pad=12)
        ax.set_xticks(x); ax.set_xticklabels(strategies, fontsize=10)
        ax.set_ylabel('Score', fontsize=11); ax.set_ylim(0, 0.65)
        ax.legend(fontsize=10)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.set_facecolor('#FAFAFA'); ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    print("Results chart saved as " + save_path)


# ── 3. 4-STRATEGY CHART INCLUDING TA-ARE ─────────────────────────────────────

def plot_results_with_taare(results_df, taare_df,
                             save_path='../diagram/results_chart_taare.png'):
    """
    4-strategy comparison chart including TA-ARE.
    Mirrors Figure 1 (top) from the original paper.
    Shows Overall, Retrieval-Needed, and Parametric splits.
    """
    strategies = ['No\nRetrieval', 'Always\nRetrieval', 'Adaptive\n(Oracle)', 'TA-ARE\n(Paper)']
    colors = ['#94A3B8', '#0D9488', '#7C3AED', '#B45309']

    def get_scores(rdf, tdf, split=None):
        if split == 'ret':
            rdf = rdf[rdf['needs_retrieval'] == 0]
            tdf = tdf[tdf['needs_retrieval'] == 0]
        elif split == 'par':
            rdf = rdf[rdf['needs_retrieval'] == 1]
            tdf = tdf[tdf['needs_retrieval'] == 1]
        em = [rdf['em_no_ret'].mean(), rdf['em_always'].mean(),
              rdf['em_adaptive'].mean(), tdf['em_taare'].mean()]
        f1 = [rdf['f1_no_ret'].mean(), rdf['f1_always'].mean(),
              rdf['f1_adaptive'].mean(), tdf['f1_taare'].mean()]
        return em, f1

    ret_df = results_df[results_df['needs_retrieval'] == 0]
    par_df = results_df[results_df['needs_retrieval'] == 1]

    panels = [
        (*get_scores(results_df, taare_df),       f'Overall ({len(results_df)} questions)'),
        (*get_scores(results_df, taare_df, 'ret'), f'Retrieval-Needed (n={len(ret_df)})'),
        (*get_scores(results_df, taare_df, 'par'), f'Parametric Only (n={len(par_df)})'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('4-Strategy Comparison including TA-ARE — GPT-3.5-turbo\n'
                 'Nishi Patel (501356244)  ·  Avi Patel (501376903)',
                 fontsize=13, fontweight='bold', color=NAVY, y=1.02)

    x, w = np.arange(4), 0.35

    for ax, (em, f1, title) in zip(axes, panels):
        b1 = ax.bar(x-w/2, em, w, label='Exact Match', color=colors, edgecolor='white', linewidth=1.5)
        b2 = ax.bar(x+w/2, f1, w, label='Token F1',
                    color=[c+'99' for c in colors], edgecolor='white', linewidth=1.5)

        for bar in b1:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.007,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold', color=NAVY)
        for bar in b2:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.007,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8, color=DKGRAY)

        # Highlight TA-ARE bars with thicker border
        b1[3].set_edgecolor(AMBER); b1[3].set_linewidth(2.5)
        b2[3].set_edgecolor(AMBER); b2[3].set_linewidth(2.5)

        ax.set_title(title, fontsize=11, fontweight='bold', color=NAVY, pad=10)
        ax.set_xticks(x); ax.set_xticklabels(strategies, fontsize=9)
        ax.set_ylabel('Score', fontsize=10); ax.set_ylim(0, 0.75)
        ax.legend(fontsize=9)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.set_facecolor('#FAFAFA'); ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    print("4-strategy chart saved as " + save_path)


# ── 4. RETRIEVAL ACCURACY CHART ──────────────────────────────────────────────

def plot_retrieval_accuracy(taare_df, save_path='../diagram/retrieval_accuracy.png'):
    """
    TA-ARE retrieval decision accuracy chart.
    Left: TP/TN/FP/FN breakdown.
    Right: Comparison against paper's GPT-3.5 benchmarks.
    """
    needs = taare_df['needs_retrieval'] == 0
    did   = taare_df['taare_did_retrieve'] == True

    tp = int(( needs &  did).sum())
    tn = int((~needs & ~did).sum())
    fp = int((~needs &  did).sum())
    fn = int(( needs & ~did).sum())
    acc = (tp + tn) / len(taare_df)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('TA-ARE Retrieval Decision Analysis — GPT-3.5-turbo\n'
                 'Nishi Patel (501356244)  ·  Avi Patel (501376903)',
                 fontsize=13, fontweight='bold', color=NAVY)

    # Left — TP/TN/FP/FN
    ax = axes[0]
    cats   = ['TP\n(Correct\nRetrieve)', 'TN\n(Correct\nSkip)',
              'FP\n(Unnecessary\nRetrieval)', 'FN\n(Missed\nRetrieval)']
    counts = [tp, tn, fp, fn]
    bcolors = ['#059669', '#0D9488', '#F59E0B', '#EF4444']

    bars = ax.bar(cats, counts, color=bcolors, edgecolor='white', linewidth=2)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                str(c), ha='center', va='bottom', fontsize=13, fontweight='bold', color=NAVY)

    ax.set_title('Retrieval Decision Breakdown (TA-ARE)', fontsize=12, fontweight='bold', color=NAVY, pad=10)
    ax.set_ylabel('Number of Questions', fontsize=11)
    ax.set_ylim(0, max(counts)*1.25)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.set_facecolor('#FAFAFA'); ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Right — comparison with paper
    ax2 = axes[1]
    methods  = ['GPT-3.5\nVanilla\n(Paper)', 'GPT-3.5\nTA-ARE\n(Paper)', 'Our\nTA-ARE\n(GPT-3.5)']
    acc_vals = [0.493, 0.863, acc]
    bcolors2 = ['#94A3B8', '#0D9488', '#B45309']

    bars2 = ax2.bar(methods, acc_vals, color=bcolors2, edgecolor='white', linewidth=2, width=0.5)
    for bar, val in zip(bars2, acc_vals):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                 f'{val*100:.1f}%', ha='center', va='bottom', fontsize=13, fontweight='bold', color=NAVY)

    bars2[2].set_edgecolor(AMBER); bars2[2].set_linewidth(3)

    ax2.set_title('Retrieval Accuracy vs Paper Benchmarks', fontsize=12, fontweight='bold', color=NAVY, pad=10)
    ax2.set_ylabel('Retrieval Accuracy', fontsize=11)
    ax2.set_ylim(0, 1.1)
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random baseline (50%)')
    ax2.legend(fontsize=9)
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
    ax2.set_facecolor('#FAFAFA'); ax2.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    print("Retrieval accuracy chart saved as " + save_path)


# ── 5. ORIGINAL PAPER ARCHITECTURE ───────────────────────────────────────────

def plot_paper_architecture(save_path='../diagram/original_paper_architecture.png'):
    """
    3-panel architecture diagram of the original RetrievalQA paper.
    Part 1 — Dataset construction
    Part 2 — ARAG evaluation framework
    Part 3 — Key findings
    """
    fig, axes = plt.subplots(3, 1, figsize=(16, 22))
    fig.patch.set_facecolor('white')
    fig.suptitle("RetrievalQA — Original Paper Architecture\n"
                 "Zhang, Fang & Chen  ·  ACL Findings 2024  ·  arxiv.org/abs/2402.16457",
                 fontsize=14, fontweight='bold', color=NAVY, y=0.99)

    def box(ax, x, y, w, h, color, title, sub='', fontsize=9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                    facecolor=color, edgecolor=WHITE,
                                    linewidth=1.5, zorder=3))
        ty = y + h*0.72 if sub else y + h/2
        ax.text(x+w/2, ty, title, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=WHITE, zorder=4)
        if sub:
            ax.text(x+w/2, y+h/2-0.18, sub, ha='center', va='center',
                    fontsize=fontsize-1.5, color=WHITE, alpha=0.88, zorder=4)

    def arr(ax, x1, y1, x2, y2, color=GRAY):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

    # Part 1 — Dataset
    ax = axes[0]; ax.set_xlim(0,16); ax.set_ylim(0,7); ax.axis('off'); ax.set_facecolor('#F8FFFE')
    ax.text(0.15, 6.65, "Part 1 — Dataset construction pipeline",
            fontsize=12, fontweight='bold', color=TEAL, va='top')
    ax.text(0.15, 6.25, "9,336 raw questions from 5 sources  →  GPT-4 filter  →  2,785 final questions",
            fontsize=9, color=GRAY, va='top')

    for name, count, x in [("RealTimeQA","397 Qs",0.4),("FreshQA","127 Qs",2.0),
                             ("ToolQA","100 Qs",3.6),("PopQA","1,399 Qs",5.2),("TriviaQA","7,313 Qs",6.8)]:
        ax.add_patch(FancyBboxPatch((x,4.6),1.4,0.9,boxstyle="round,pad=0.06",
                                    facecolor=DKGRAY,edgecolor=WHITE,linewidth=1.5,zorder=3))
        ax.text(x+0.7,5.12,name,ha='center',va='center',fontsize=9,fontweight='bold',color=WHITE,zorder=4)
        ax.text(x+0.7,4.85,count,ha='center',va='center',fontsize=8,color=WHITE,alpha=0.85,zorder=4)
        arr(ax,x+0.7,4.6,x+0.7,4.2,GRAY)

    box(ax,3.0,3.1,6.5,0.85,AMBER,"GPT-4 filtering  (closed-book setting)",
        "Keep only questions where Token F1 = 0  (GPT-4 fails without retrieval)")
    for x in [1.1,2.7,4.3,5.9,7.5]:
        arr(ax,x,4.2,6.25,3.95,AMBER)

    ax.annotate('',xy=(2.0,2.4),xytext=(4.5,3.1),arrowprops=dict(arrowstyle='->',color=TEAL,lw=1.5))
    ax.annotate('',xy=(9.5,2.4),xytext=(8.0,3.1),arrowprops=dict(arrowstyle='->',color=GRAY,lw=1.5))
    box(ax,0.4,1.5,3.8,0.8,TEAL,"Retrieval-needed  (1,271 kept)","New world + long-tail  —  must retrieve")
    box(ax,8.2,1.5,3.8,0.8,BLUE,"Parametric  (1,514 added)","Answerable from memory  —  no retrieval")
    ax.annotate('',xy=(6.4,0.75),xytext=(2.3,1.5),arrowprops=dict(arrowstyle='->',color=TEAL,lw=1.5,connectionstyle='arc3,rad=0'))
    ax.annotate('',xy=(8.0,0.75),xytext=(10.1,1.5),arrowprops=dict(arrowstyle='->',color='#185FA5',lw=1.5,connectionstyle='arc3,rad=0'))
    box(ax,5.2,0.05,4.0,0.85,TEAL,"RetrievalQA benchmark  (2,785 total)","1,271 retrieval-needed  +  1,514 parametric")

    # Part 2 — ARAG
    ax = axes[1]; ax.set_xlim(0,16); ax.set_ylim(0,7); ax.axis('off'); ax.set_facecolor('#FEFEFF')
    ax.text(0.15,6.65,"Part 2 — ARAG evaluation framework",fontsize=12,fontweight='bold',color=PURPLE,va='top')
    ax.text(0.15,6.25,"Every question tested through 3 strategies  ×  6 LLMs",fontsize=9,color=GRAY,va='top')

    box(ax,5.5,5.2,4.5,0.8,DKGRAY,"Short-form question  ×","new world or long-tail knowledge")
    arr(ax,6.5,5.2,2.0,4.5,DKGRAY); arr(ax,7.75,5.2,7.75,4.5,PURPLE); arr(ax,9.0,5.2,13.5,4.5,TEAL)
    box(ax,0.4,3.5,3.2,0.8,DKGRAY,"No retrieval","Answer from memory only")
    box(ax,6.1,3.5,3.3,0.8,PURPLE,"Adaptive retrieval","LLM decides whether to retrieve")
    box(ax,11.8,3.5,3.8,0.8,TEAL,"Always retrieval","Always fetch top-5 docs first")
    box(ax,9.0,5.2,3.5,0.8,BLUE,"Retriever","Contriever / Google  —  top-5 docs")
    arr(ax,10.75,5.2,13.7,4.3,'#185FA5'); arr(ax,10.0,5.2,8.0,4.3,'#185FA5')
    arr(ax,7.75,3.5,7.75,2.85,PURPLE)
    box(ax,5.2,1.95,5.1,0.8,'#5B21B6',"TA-ARE  (paper's proposed method)",
        "Today's date  +  2 Yes  +  2 No  in-context examples")

    ax.text(7.75,1.36,"LLMs tested in the paper:",ha='center',fontsize=8.5,color=GRAY)
    for name,size,x in [("TinyLlama","1.1B",0.4),("Phi-2","2.7B",2.55),("Llama-2","7B",4.7),
                         ("Self-RAG","7B",6.85),("GPT-3.5","OpenAI",9.0),("GPT-4","250 samples",11.15)]:
        ax.add_patch(FancyBboxPatch((x,0.2),1.9,0.75,boxstyle="round,pad=0.05",
                                    facecolor=DKGRAY,edgecolor=WHITE,linewidth=1.5,zorder=3))
        ax.text(x+0.95,0.64,name,ha='center',va='center',fontsize=8.5,fontweight='bold',color=WHITE,zorder=4)
        ax.text(x+0.95,0.38,size,ha='center',va='center',fontsize=7.5,color=WHITE,alpha=0.85,zorder=4)

    # Part 3 — Findings
    ax = axes[2]; ax.set_xlim(0,16); ax.set_ylim(0,5); ax.axis('off'); ax.set_facecolor('#F8FFF8')
    ax.text(0.15,4.65,"Part 3 — Key findings",fontsize=12,fontweight='bold',color=GREEN,va='top')
    ax.text(0.15,4.25,"What the paper discovered after running all experiments",fontsize=9,color=GRAY,va='top')

    box(ax,0.4,2.4,4.5,1.4,TEAL,"Finding 1 — Always Retrieval wins",
        "Always Ret. >= Adaptive >= No Ret.  Retrieval helps on hard questions")
    box(ax,5.8,2.4,4.5,1.4,PURPLE,"Finding 2 — TA-ARE beats vanilla",
        "avg +14.9% retrieval accuracy across all LLMs vs vanilla prompting")
    box(ax,11.2,2.4,4.4,1.4,AMBER,"Finding 3 — Main claim",
        "GPT-3.5 fails to retrieve >50% of the time  LLMs misjudge knowledge gaps")

    arr(ax,2.65,2.4,6.5,1.5,GRAY); arr(ax,8.05,2.4,8.05,1.5,GRAY); arr(ax,13.4,2.4,9.6,1.5,GRAY)
    box(ax,4.5,0.3,7.0,1.1,'#3B6D11',"Overall conclusion",
        "Vanilla prompting fails  —  TA-ARE with date + ICL examples fixes it")

    plt.tight_layout(rect=[0,0,1,0.98])
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    print("Original paper architecture saved as " + save_path)
