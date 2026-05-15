import os
import json
import re
import gradio as gr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
from google import genai

# ── Configuration ─────────────────────────────────────────────────────────────
API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not API_KEY:
    API_KEY = input("Enter your Gemini API Key (https://aistudio.google.com/): ").strip()

client = genai.Client(api_key=API_KEY)

# ── Build Knowledge Graph ──────────────────────────────────────────────────────
G = nx.DiGraph()

medical_edges = [
    # DIABETES
    ("Diabetes", "Frequent Urination", "has_symptom"),
    ("Diabetes", "Excessive Thirst", "has_symptom"),
    ("Diabetes", "Blurred Vision", "has_symptom"),
    ("Diabetes", "Fatigue", "has_symptom"),
    ("Diabetes", "Slow Healing Wounds", "has_symptom"),
    ("Diabetes", "Metformin", "treated_by"),
    ("Diabetes", "Insulin", "treated_by"),
    ("Metformin", "Nausea", "side_effect"),
    ("Metformin", "Diarrhea", "side_effect"),
    ("Metformin", "Alcohol", "interacts_with"),
    ("Insulin", "Low Blood Sugar", "side_effect"),
    ("Diabetes", "Kidney Disease", "causes"),
    # HYPERTENSION
    ("Hypertension", "Headache", "has_symptom"),
    ("Hypertension", "Dizziness", "has_symptom"),
    ("Hypertension", "Chest Pain", "has_symptom"),
    ("Hypertension", "Blurred Vision", "has_symptom"),
    ("Hypertension", "Amlodipine", "treated_by"),
    ("Hypertension", "Lisinopril", "treated_by"),
    ("Amlodipine", "Swollen Ankles", "side_effect"),
    ("Lisinopril", "Dry Cough", "side_effect"),
    ("Hypertension", "Heart Attack", "causes"),
    # INFLUENZA
    ("Influenza", "Fever", "has_symptom"),
    ("Influenza", "Body Ache", "has_symptom"),
    ("Influenza", "Fatigue", "has_symptom"),
    ("Influenza", "Cough", "has_symptom"),
    ("Influenza", "Sore Throat", "has_symptom"),
    ("Influenza", "Paracetamol", "treated_by"),
    ("Paracetamol", "Liver Damage", "side_effect"),
    ("Paracetamol", "Ibuprofen", "interacts_with"),
    # MIGRAINE
    ("Migraine", "Severe Headache", "has_symptom"),
    ("Migraine", "Nausea", "has_symptom"),
    ("Migraine", "Light Sensitivity", "has_symptom"),
    ("Migraine", "Blurred Vision", "has_symptom"),
    ("Migraine", "Sumatriptan", "treated_by"),
    ("Migraine", "Ibuprofen", "treated_by"),
    ("Sumatriptan", "Dizziness", "side_effect"),
    ("Ibuprofen", "Stomach Ulcer", "side_effect"),
    # ASTHMA
    ("Asthma", "Wheezing", "has_symptom"),
    ("Asthma", "Shortness of Breath", "has_symptom"),
    ("Asthma", "Chest Tightness", "has_symptom"),
    ("Asthma", "Cough", "has_symptom"),
    ("Asthma", "Salbutamol", "treated_by"),
    ("Salbutamol", "Tremors", "side_effect"),
    ("Salbutamol", "Rapid Heartbeat", "side_effect"),
    # DEPRESSION
    ("Depression", "Persistent Sadness", "has_symptom"),
    ("Depression", "Fatigue", "has_symptom"),
    ("Depression", "Sleep Problems", "has_symptom"),
    ("Depression", "Loss of Appetite", "has_symptom"),
    ("Depression", "Sertraline", "treated_by"),
    ("Depression", "Fluoxetine", "treated_by"),
    ("Sertraline", "Nausea", "side_effect"),
    ("Sertraline", "Insomnia", "side_effect"),
    ("Sertraline", "Alcohol", "interacts_with"),
    ("Fluoxetine", "Anxiety", "side_effect"),
]

for src, tgt, rel in medical_edges:
    G.add_edge(src, tgt, relation=rel)

# Add lay terms for vector search
lay_terms = [
    ("Frequent Urination", "waking up to pee", "also_known_as"),
    ("Frequent Urination", "peeing a lot", "also_known_as"),
    ("Excessive Thirst", "always thirsty", "also_known_as"),
    ("Headache", "head is pounding", "also_known_as"),
    ("Dizziness", "feel dizzy", "also_known_as"),
    ("Persistent Sadness", "feeling low", "also_known_as"),
    ("Loss of Appetite", "lost interest in everything", "also_known_as"),
    ("Chest Tightness", "chest feels tight", "also_known_as"),
    ("Wheezing", "whistling sound breathing", "also_known_as"),
    ("Severe Headache", "bad headache one side", "also_known_as"),
    ("Light Sensitivity", "can't stand light", "also_known_as"),
    ("Nausea", "feeling sick", "also_known_as"),
    ("Fatigue", "always tired", "also_known_as"),
    ("Blurred Vision", "can't see clearly", "also_known_as"),
]
for src, tgt, rel in lay_terms:
    G.add_edge(src, tgt, relation=rel)

# Create embeddings for all nodes
from sentence_transformers import SentenceTransformer
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
node_list = list(G.nodes())
node_texts = [str(node) for node in node_list]
node_embeddings = embed_model.encode(node_texts)

# ── Search Functions ────────────────────────────────────────────────────────────
def graph_search_by_symptoms(symptom_list):
    results = {}
    all_diseases = ["Diabetes", "Hypertension", "Influenza", "Migraine", "Asthma", "Depression"]
    for disease in all_diseases:
        disease_symptoms = [tgt for _, tgt, data in G.out_edges(disease, data=True) if data['relation'] == 'has_symptom']
        matched = [ds for us in symptom_list for ds in disease_symptoms if us.lower() in ds.lower() or ds.lower() in us.lower()]
        if not matched:
            continue
        medicines_found = [tgt for _, tgt, data in G.out_edges(disease, data=True) if data['relation'] == 'treated_by']
        medicine_details = {}
        for med in medicines_found:
            medicine_details[med] = {
                'side_effects': [tgt for _, tgt, data in G.out_edges(med, data=True) if data['relation'] == 'side_effect'],
                'interactions': [tgt for _, tgt, data in G.out_edges(med, data=True) if data['relation'] == 'interacts_with']
            }
        complications = [tgt for _, tgt, data in G.out_edges(disease, data=True) if data['relation'] == 'causes']
        results[disease] = {'match_score': len(matched), 'matched_symptoms': matched, 'medicines': medicine_details, 'complications': complications}
    return dict(sorted(results.items(), key=lambda x: x[1]['match_score'], reverse=True))

def graph_search_medicine(medicine_name):
    result = {}
    for node in G.nodes():
        if medicine_name.lower() in node.lower():
            result[node] = {
                'used_for': [src for src, tgt, d in G.in_edges(node, data=True) if d['relation'] == 'treated_by'],
                'side_effects': [tgt for _, tgt, d in G.out_edges(node, data=True) if d['relation'] == 'side_effect'],
                'interactions': [tgt for _, tgt, d in G.out_edges(node, data=True) if d['relation'] == 'interacts_with']
            }
    return result

def real_rag_search(query, top_k=5):
    query_embedding = embed_model.encode([query])[0]
    similarities = []
    for i, node in enumerate(node_list):
        similarity = np.dot(query_embedding, node_embeddings[i]) / (np.linalg.norm(query_embedding) * np.linalg.norm(node_embeddings[i]))
        similarities.append((similarity, node))
    similarities.sort(reverse=True)
    top_nodes = [node for _, node in similarities[:top_k]]
    results = {}
    for node in top_nodes:
        connections = {'similar_to_query': node, 'connected_to': []}
        for _, neighbor, data in G.out_edges(node, data=True):
            connections['connected_to'].append({'node': neighbor, 'relation': data['relation']})
        for src, _, data in G.in_edges(node, data=True):
            connections['connected_to'].append({'node': src, 'relation': f"is_{data['relation']}_of"})
        results[node] = connections
    return results

def medical_agent_v2(user_query):
    SYSTEM_PROMPT = """You are an expert medical assistant AI powered by a medical knowledge graph with real vector embeddings.

When a user describes symptoms:
1. ALWAYS use search_by_symptoms tool first
2. ALWAYS use real_vector_search tool to find semantically similar conditions
3. Combine both results to give a comprehensive answer
4. Rank diseases by likelihood
5. If they ask about a medicine use search_medicine_info

Always structure your answer like this:

🔍 SYMPTOM ANALYSIS
 POSSIBLE CONDITIONS (ranked by likelihood)
 TREATMENT OPTIONS
 IMPORTANT WARNINGS
 RECOMMENDATION

Always end with: Please consult a real doctor for diagnosis."""

    from google.genai import types
    tools = [types.Tool(function_declarations=[
        types.FunctionDeclaration(name="search_by_symptoms", description="Search medical knowledge graph using symptoms — keyword matching.",
            parameters=types.Schema(type="OBJECT", properties={"symptoms": types.Schema(type="ARRAY", items=types.Schema(type="STRING"), description="List of symptoms")}, required=["symptoms"])),
        types.FunctionDeclaration(name="real_vector_search", description="Search using vector embeddings — finds by MEANING not keywords.",
            parameters=types.Schema(type="OBJECT", properties={"query": types.Schema(type="STRING", description="Natural language symptom description")}, required=["query"])),
        types.FunctionDeclaration(name="search_medicine_info", description="Get detailed info about a medicine — side effects, interactions.",
            parameters=types.Schema(type="OBJECT", properties={"medicine_name": types.Schema(type="STRING", description="Medicine name")}, required=["medicine_name"]))
    ])]

    messages = [types.Content(role="user", parts=[types.Part(text=user_query)])]

    for step in range(8):
        response = client.models.generate_content(model="gemini-2.5-flash-lite", contents=messages, config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, tools=tools))
        has_tool_call = False
        for part in response.candidates[0].content.parts:
            if part.function_call is not None and part.function_call.name:
                has_tool_call = True
                tool_name = part.function_call.name
                tool_args = dict(part.function_call.args)
                if tool_name == 'search_by_symptoms': result = graph_search_by_symptoms(tool_args.get('symptoms', []))
                elif tool_name == 'real_vector_search': result = real_rag_search(tool_args.get('query', ''))
                elif tool_name == 'search_medicine_info': result = graph_search_medicine(tool_args.get('medicine_name', ''))
                else: result = {}
                messages.append(response.candidates[0].content)
                messages.append(types.Content(role="tool", parts=[types.Part(function_response=types.FunctionResponse(name=tool_name, response={"result": json.dumps(result, indent=2)}))]))
                break
        if not has_tool_call:
            return response.text
    return 'Could not complete analysis.'

# ── Gradio UI ──────────────────────────────────────────────────────────────────
def draw_graph():
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.patch.set_facecolor('#161c2e')
    ax.set_facecolor('#161c2e')
    type_map = {}
    for u, v, d in G.edges(data=True):
        rel = d.get('relation', '')
        if rel == 'treated_by': type_map[u] = 'disease'; type_map[v] = 'medicine'
        elif rel == 'has_symptom': type_map[u] = 'disease'; type_map[v] = 'symptom'
        elif rel == 'side_effect': type_map[v] = 'side_effect'
    for n in G.nodes():
        if n not in type_map: type_map[n] = 'symptom'
    color_map2 = {'disease': '#ff4d6d', 'medicine': '#00ffb2', 'symptom': '#7b61ff', 'side_effect': '#ffbe0b'}
    node_colors = [color_map2.get(type_map.get(n, 'symptom'), '#7b61ff') for n in G.nodes()]
    node_sizes = [800 if type_map.get(n) == 'disease' else 500 if type_map.get(n) == 'medicine' else 300 for n in G.nodes()]
    pos = nx.spring_layout(G, k=2.5, iterations=50, seed=42)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#ffffff15', arrows=True, arrowsize=8, width=0.8)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes, alpha=0.95)
    nx.draw_networkx_labels(G, pos, labels={n: n for n in G.nodes() if type_map.get(n) == 'disease'}, ax=ax, font_size=9, font_color='#ffffff', font_weight='bold')
    ax.legend(handles=[
        mpatches.Patch(color='#ff4d6d', label='Disease'),
        mpatches.Patch(color='#00ffb2', label='Medicine'),
        mpatches.Patch(color='#7b61ff', label='Symptom'),
        mpatches.Patch(color='#ffbe0b', label='Side Effect'),
    ], loc='upper left', facecolor='#0f1320', edgecolor='#ffffff1a', labelcolor='#dde3f0', fontsize=8)
    ax.set_title(f'Medical Knowledge Graph  ·  {G.number_of_nodes()} nodes  ·  {G.number_of_edges()} edges', color='#5a6280', fontsize=9, pad=10)
    ax.axis('off')
    plt.tight_layout()
    return fig

def run_agent_ui(query):
    if not query.strip(): return "❌ Please enter your symptoms!"
    try: return medical_agent_v2(query)
    except Exception as e: return f"Error: {str(e)}"

custom_css = """
body, .gradio-container { background: #080b12 !important; }
footer { display: none !important; }
textarea, input[type='text'] { background: #161c2e !important; border: 1px solid #ffffff1a !important; border-radius: 12px !important; color: #dde3f0 !important; }
.gr-button-primary { background: linear-gradient(135deg, #00ffb2, #00cc8f) !important; border: none !important; border-radius: 12px !important; color: #040810 !important; font-weight: 600 !important; }
"""

with gr.Blocks(css=custom_css, title="MedGraph AI") as demo:
    gr.HTML(f"""
    <div style="background:linear-gradient(90deg,#080b12,#0d1220);border-bottom:1px solid #ffffff0f;padding:16px 28px;display:flex;align-items:center;justify-content:space-between;margin-bottom:16px">
      <div style="display:flex;align-items:center;gap:12px">
        <div style="width:34px;height:34px;border-radius:9px;background:linear-gradient(135deg,#00ffb2,#7b61ff);display:flex;align-items:center;justify-content:center;font-size:16px">⚕</div>
        <div>
          <div style="font-size:15px;font-weight:600;color:#dde3f0">MedGraph AI</div>
          <div style="font-size:10px;color:#5a6280;font-family:monospace">Graph RAG · Vector Embeddings · Agentic AI · Gemini</div>
        </div>
      </div>
      <div style="display:flex;gap:10px">
        <span style="font-family:monospace;font-size:10px;padding:4px 10px;border-radius:20px;border:1px solid #00ffb260;color:#00ffb2">{G.number_of_nodes()} nodes · {G.number_of_edges()} edges</span>
        <span style="font-family:monospace;font-size:10px;padding:4px 10px;border-radius:20px;border:1px solid #7b61ff60;color:#7b61ff">Real Embeddings </span>
      </div>
    </div>
    """)
    with gr.Row():
        with gr.Column(scale=1):
            graph_plot = gr.Plot(label="KNOWLEDGE GRAPH", show_label=True)
            demo.load(fn=draw_graph, outputs=graph_plot)
        with gr.Column(scale=1):
            query_input = gr.Textbox(lines=5, placeholder="Describe your symptoms...\ne.g. persistent cough with blood, chest pain, weight loss", label="DESCRIBE YOUR SYMPTOMS")
            run_btn = gr.Button("Run Medical Analysis ↗", variant="primary")
            gr.Examples(examples=[
                ["Persistent cough with blood, chest pain and weight loss."],
                ["Severe headache, nausea and sensitivity to light."],
                ["Always tired, very thirsty, blurry vision."],
                ["Wheezing, chest tightness, shortness of breath."],
                ["Persistent sadness, sleep problems, loss of interest."],
                ["Side effects of Metformin?"],
            ], inputs=query_input, label="QUICK EXAMPLES")
            output = gr.Textbox(lines=18, label="AI MEDICAL REPORT", interactive=False)
    run_btn.click(fn=run_agent_ui, inputs=query_input, outputs=output)

demo.launch()