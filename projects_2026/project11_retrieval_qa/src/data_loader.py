"""
data_loader.py
==============
Load, explore, and sample the RetrievalQA benchmark dataset.

The dataset (hyintell/RetrievalQA on HuggingFace) contains 2,785
short-form QA questions with pre-retrieved context passages and
ground-truth answers. Questions are labelled:
  param_knowledge_answerable = 0  ->  LLM must retrieve to answer
  param_knowledge_answerable = 1  ->  LLM can answer from memory

"""

import json
import pandas as pd
from datasets import load_dataset


def load_retrievalqa_dataset():
    """Load the RetrievalQA benchmark from HuggingFace."""
    dataset = load_dataset("hyintell/RetrievalQA")
    df = dataset['train'].to_pandas()
    print(f"Dataset loaded successfully!")
    print(f"  Total questions : {len(df)}")
    print(f"  Columns         : {df.columns.tolist()}")
    return df


def explore_dataset(df):
    """Print dataset statistics split by retrieval label and data source."""
    needs_retrieval = df[df['param_knowledge_answerable'] == 0]
    parametric_only = df[df['param_knowledge_answerable'] == 1]

    print("=" * 50)
    print(f"Questions needing retrieval   : {len(needs_retrieval)}")
    print(f"Questions answerable in memory: {len(parametric_only)}")
    print(f"Total                         : {len(df)}")
    print("=" * 50)
    print("\nBreakdown by data source:")
    print(df['data_source'].value_counts())

    return needs_retrieval, parametric_only


def show_examples(df, indices=[10, 100, 500]):
    """Print detailed inspection of selected questions."""
    for idx in indices:
        row = df.iloc[idx]
        print("=" * 55)
        print(f"Question       : {row['question']}")
        print(f"Correct answer : {row['ground_truth']}")
        label = 'YES <- must retrieve' if row['param_knowledge_answerable'] == 0 \
                else 'NO  <- AI knows this'
        print(f"Needs retrieval: {label}")
        print(f"Data source    : {row['data_source']}")

        context_list = list(row['context'])
        if len(context_list) > 0:
            ctx = json.loads(context_list[0])
            print(f"Context title  : {ctx.get('title', '(no title)')}")
            print(f"Context text   : {ctx.get('text', '')[:180]}...")
        else:
            print("Context        : (no context available)")
        print()


def create_sample(needs_retrieval, parametric_only,
                  n_retrieval=150, n_parametric=100, random_state=42):
    """
    Create a balanced 250-question sample (150 retrieval + 100 parametric).
    Matches the GPT-4 subset size used in the original paper.
    """
    sample_ret   = needs_retrieval.sample(n_retrieval,  random_state=random_state)
    sample_param = parametric_only.sample(n_parametric, random_state=random_state)

    sample_df = pd.concat([sample_ret, sample_param])
    sample_df = sample_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    print(f"Sample created!")
    print(f"  Total            : {len(sample_df)}")
    print(f"  Needs retrieval  : {len(sample_df[sample_df['param_knowledge_answerable']==0])}")
    print(f"  Parametric only  : {len(sample_df[sample_df['param_knowledge_answerable']==1])}")
    print(f"\nBreakdown by data source:")
    print(sample_df['data_source'].value_counts())

    return sample_df