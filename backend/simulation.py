import os
import json
import networkx as nx
import chromadb
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

# ── LLM Configuration ───────────────────────────────────
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
    model=os.getenv("OPENAI_MODEL_NAME"),
    temperature=0.7
)

# ── ChromaDB Memory ──────────────────────────────────────
chroma_client = chromadb.Client()
memory = chroma_client.get_or_create_collection(name="swarm_memory")

# ── Knowledge Graph ──────────────────────────────────────
graph = nx.DiGraph()

# ── State Definition ─────────────────────────────────────
class SimulationState(TypedDict):
    round_num: int
    max_rounds: int
    consensus_reached: bool
    consensus_score: float
    developer_position: str
    devops_position: str
    pm_position: str
    pm_previous_position: str
    history: list
    final_report: str

# ── Memory Functions ─────────────────────────────────────

def save_memory(agent: str, round_num: int, content: str):
    doc_id = f"{agent}_round_{round_num}"
    memory.add(
        documents=[content],
        metadatas=[{"agent": agent, "round": round_num}],
        ids=[doc_id]
    )

def get_memory_context(query: str, n: int = 6) -> str:
    try:
        results = memory.query(query_texts=[query], n_results=n)
        if results["documents"] and results["documents"][0]:
            docs = []
            for doc, meta in zip(
                results["documents"][0],
                results["metadatas"][0]
            ):
                docs.append(
                    f"[{meta['agent']} - Round {meta['round']}]:\n{doc}"
                )
            return "\n---\n".join(docs)
        return "No previous context."
    except Exception:
        return "No previous context."

# ── Graph Update ─────────────────────────────────────────

def update_knowledge_graph(agent: str, concepts: list, round_num: int):
    graph.add_node(agent, type="agent")
    for concept in concepts:
        graph.add_node(concept, type="concept")
        if graph.has_edge(agent, concept):
            graph[agent][concept]["weight"] += 1
        else:
            graph.add_edge(agent, concept, round=round_num, weight=1)

# ── Consensus Detection ──────────────────────────────────

def check_consensus(state: SimulationState) -> float:
    """
    Détecte le consensus en comparant la position actuelle
    du PM avec sa position précédente.
    Score entre 0 (aucun consensus) et 1 (consensus total).
    """
    current = state["pm_position"].lower()
    previous = state["pm_previous_position"].lower()

    if not previous or previous == "":
        return 0.0

    keywords = [
        "aws", "azure", "gcp", "eks", "aks", "gke",
        "migrate", "recommend", "cost", "timeline"
    ]

    current_hits = set(k for k in keywords if k in current)
    previous_hits = set(k for k in keywords if k in previous)

    if not current_hits or not previous_hits:
        return 0.0

    intersection = current_hits & previous_hits
    union = current_hits | previous_hits
    score = len(intersection) / len(union)

    print(f"  🎯 Consensus score: {score:.2f} "
          f"(shared keywords: {intersection})")
    return score

# ── Agent Nodes ──────────────────────────────────────────

def developer_node(state: SimulationState) -> SimulationState:
    round_num = state["round_num"]
    print(f"\n  🐍 [Round {round_num}] Python Developer thinking...")

    context = get_memory_context(
        "Python migration OpenShift EKS AKS GKE vendor lock-in"
    )

    if round_num == 1:
        instruction = """Analyze the technical feasibility of migrating 
        Python apps (Django, FastAPI, Celery) from OpenShift to AWS, Azure, 
        or GCP. Cover: technical challenges, best managed Kubernetes service 
        (EKS vs AKS vs GKE), vendor lock-in risk, required code changes. 
        Give a clear recommendation. Max 200 words."""
    else:
        instruction = f"""Round {round_num}: Review the previous debate and 
        REFINE your technical position. Have you changed your mind? 
        What new argument can you add? Address concerns raised by DevOps 
        and the PM. Max 200 words."""

    messages = [
        SystemMessage(content="""You are a Senior Python Developer with 10 
        years of experience maintaining Django, FastAPI, and Celery apps on 
        OpenShift. You care about performance and avoiding vendor lock-in."""),
        HumanMessage(content=f"""Previous debate context:
{context}

{instruction}""")
    ]

    response = llm.invoke(messages)
    content = response.content

    save_memory("python_developer", round_num, content)
    update_knowledge_graph(
        "python_developer",
        ["EKS", "AKS", "GKE", "Django", "FastAPI",
         "Celery", "vendor lock-in", "containerization"],
        round_num
    )

    print(f"  ✅ Developer position saved (round {round_num})")
    return {**state, "developer_position": content}


def devops_node(state: SimulationState) -> SimulationState:
    round_num = state["round_num"]
    print(f"\n  🔧 [Round {round_num}] DevOps Engineer thinking...")

    context = get_memory_context(
        "infrastructure migration ArgoCD Helm Prometheus Grafana downtime"
    )
    dev_position = state["developer_position"]

    if round_num == 1:
        instruction = """Evaluate infrastructure implications: OpenShift vs 
        EKS/AKS/GKE, migration timeline, CI/CD impact (ArgoCD, Helm), 
        monitoring migration (Prometheus, Grafana), hidden operational costs. 
        Max 200 words."""
    else:
        instruction = f"""Round {round_num}: Review all previous arguments. 
        What do you agree or disagree with? Refine your infrastructure 
        assessment. Add new operational insights. Max 200 words."""

    messages = [
        SystemMessage(content="""You are a DevOps Engineer managing an 
        OpenShift cluster with ArgoCD, Helm, Prometheus and Grafana. 
        You have hands-on experience with EKS, AKS, and GKE. You are 
        concerned about migration effort and hidden costs."""),
        HumanMessage(content=f"""Developer's position this round:
{dev_position}

Previous debate context:
{context}

{instruction}""")
    ]

    response = llm.invoke(messages)
    content = response.content

    save_memory("devops_engineer", round_num, content)
    update_knowledge_graph(
        "devops_engineer",
        ["ArgoCD", "Helm", "Prometheus", "Grafana",
         "migration timeline", "downtime risk", "CI/CD", "OpenShift"],
        round_num
    )

    print(f"  ✅ DevOps position saved (round {round_num})")
    return {**state, "devops_position": content}


def pm_node(state: SimulationState) -> SimulationState:
    round_num = state["round_num"]
    max_rounds = state["max_rounds"]
    print(f"\n  📊 [Round {round_num}] Project Manager thinking...")

    context = get_memory_context(
        "cloud migration cost savings recommendation board timeline budget"
    )
    dev_position = state["developer_position"]
    devops_position = state["devops_position"]
    previous_pm = state["pm_position"]

    if round_num == max_rounds:
        instruction = """FINAL ROUND: Synthesize ALL arguments from ALL rounds.
        Give your DEFINITIVE recommendation to the board:
        1. Final cloud provider choice and justification
        2. Migration roadmap (3 phases with timeline)
        3. Total cost estimate vs expected savings
        4. Top 3 risks and mitigations
        5. One clear concluding sentence for the board.
        Max 250 words."""
    elif round_num == 1:
        instruction = """Based on developer and DevOps inputs, give your 
        initial strategic assessment: migrate or not? Which cloud provider? 
        Estimated cost vs savings? Top 3 risks? Max 200 words."""
    else:
        instruction = f"""Round {round_num}: Refine your strategic position. 
        What changed from your previous position? Update cost and risk 
        assessment based on new arguments. Max 200 words."""

    messages = [
        SystemMessage(content="""You are a Project Manager with a 500,000 
        euros budget. OpenShift costs 120,000 euros/year. Goal: reduce costs 
        by 30% and improve scalability within 18 months. You report to the CTO 
        and need a clear recommendation for the board."""),
        HumanMessage(content=f"""Developer position (round {round_num}):
{dev_position}

DevOps position (round {round_num}):
{devops_position}

Your previous position:
{previous_pm if previous_pm else "This is your first position."}

Full debate context:
{context}

{instruction}""")
    ]

    response = llm.invoke(messages)
    content = response.content

    save_memory("project_manager", round_num, content)
    update_knowledge_graph(
        "project_manager",
        ["budget 500k", "cost reduction 30%", "18 months",
         "board recommendation", "ROI", "risk mitigation"],
        round_num
    )

    history = state["history"] + [{
        "round": round_num,
        "developer": dev_position,
        "devops": devops_position,
        "pm": content
    }]

    print(f"  ✅ PM position saved (round {round_num})")
    return {
        **state,
        "pm_previous_position": state["pm_position"],
        "pm_position": content,
        "history": history
    }


def consensus_node(state: SimulationState) -> SimulationState:
    """Vérifie si un consensus est atteint."""
    round_num = state["round_num"]
    max_rounds = state["max_rounds"]

    print(f"\n  🔍 [Round {round_num}] Checking consensus...")

    score = check_consensus(state)
    consensus_reached = score >= 0.75 or round_num >= max_rounds

    if consensus_reached:
        if round_num >= max_rounds:
            print(f"  🏁 Max rounds reached ({max_rounds}). "
                  f"Generating final report...")
        else:
            print(f"  🤝 Consensus reached at round {round_num} "
                  f"(score: {score:.2f}) !")
    else:
        print(f"  🔄 No consensus yet. "
              f"Moving to round {round_num + 1}...")

    return {
        **state,
        "consensus_reached": consensus_reached,
        "consensus_score": score,
        "round_num": round_num + 1
    }


def final_report_node(state: SimulationState) -> SimulationState:
    """Génère le rapport final et le résumé du graphe."""
    print(f"\n  📋 Generating final report...")

    final = state["pm_position"]

    print("\n" + "="*60)
    print("📋 FINAL STRATEGIC RECOMMENDATION")
    print("="*60)
    print(final)

    print(f"\n📊 KNOWLEDGE GRAPH SUMMARY")
    print(f"   Nodes : {graph.number_of_nodes()}")
    print(f"   Edges : {graph.number_of_edges()}")
    concepts = [
        (n, graph.in_degree(n))
        for n, d in graph.nodes(data=True)
        if d.get("type") == "concept"
    ]
    concepts.sort(key=lambda x: x[1], reverse=True)
    print(f"   Top concepts:")
    for concept, degree in concepts[:8]:
        print(f"     - {concept} (mentioned {degree}x)")

    with open("simulation_results.json", "w") as f:
        json.dump({
            "total_rounds": state["round_num"] - 1,
            "consensus_score": state["consensus_score"],
            "final_recommendation": final,
            "history": state["history"],
            "graph": {
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges()
            }
        }, f, indent=2)

    print("\n💾 Results saved to simulation_results.json")
    return {**state, "final_report": final}

# ── Routing Logic ─────────────────────────────────────────

def should_continue(state: SimulationState) -> str:
    if state["consensus_reached"]:
        return "final_report"
    return "developer"

# ── Build LangGraph ──────────────────────────────────────

workflow = StateGraph(SimulationState)

workflow.add_node("developer", developer_node)
workflow.add_node("devops", devops_node)
workflow.add_node("pm", pm_node)
workflow.add_node("consensus_check", consensus_node)
workflow.add_node("final_report", final_report_node)

workflow.set_entry_point("developer")

workflow.add_edge("developer", "devops")
workflow.add_edge("devops", "pm")
workflow.add_edge("pm", "consensus_check")

workflow.add_conditional_edges(
    "consensus_check",
    should_continue,
    {
        "developer": "developer",
        "final_report": "final_report"
    }
)

workflow.add_edge("final_report", END)

app = workflow.compile()

# ── Run Simulation ───────────────────────────────────────

print("\n" + "="*60)
print("🚀 SWARM INTELLIGENCE SIMULATION — PHASE 3")
print("   Topic  : OpenShift → Cloud Migration (AWS/Azure/GCP)")
print("   Engine : LangGraph")
print("   Max rounds: 5 | Consensus threshold: 0.75")
print("="*60)

initial_state: SimulationState = {
    "round_num": 1,
    "max_rounds": 5,
    "consensus_reached": False,
    "consensus_score": 0.0,
    "developer_position": "",
    "devops_position": "",
    "pm_position": "",
    "pm_previous_position": "",
    "history": [],
    "final_report": ""
}

result = app.invoke(initial_state)

print(f"\n✅ Simulation complete in "
      f"{result['round_num'] - 1} rounds")
