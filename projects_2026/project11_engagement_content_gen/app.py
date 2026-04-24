from flask import Flask, request, jsonify, send_from_directory
from src.graph import create_graph
from src.experiments import test_single_prompt
from src.optimization import search_best_message, lightweight_ppo

app = Flask(__name__, static_folder='.', static_url_path='')

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json() or {}
    prompt = data.get('prompt', '').strip()
    num_posts = int(data.get('num_posts', 5))

    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400

    G, opinions = create_graph()
    results_df = test_single_prompt(G, opinions, prompt, num_posts=num_posts)
    best_row = results_df.sort_values('Engagement Score', ascending=False).iloc[0]

    response = {
        'prompt': prompt,
        'results': [
            {
                'post_number': int(row['Post Number']),
                'post_text': row['Post Text'],
                'sentiment_score': float(row['Sentiment Score']),
                'engagement_score': int(row['Engagement Score']),
            }
            for _, row in results_df.iterrows()
        ],
        'best_post': {
            'post_number': int(best_row['Post Number']),
            'text': best_row['Post Text'],
            'sentiment_score': float(best_row['Sentiment Score']),
            'engagement_score': int(best_row['Engagement Score']),
        }
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
