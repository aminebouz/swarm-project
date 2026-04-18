import os
import json
import networkx as nx
import chromadb
from crewai import Agent, Task, Crew
from dotenv import load_dotenv

load_dotenv()

# ── Initialisation ChromaDB (mémoire locale) ────────────
chroma_client = chromadb.Client()
memory = chroma_client.get_or_create_collection(name="swarm_memory")

# ── Initialisation du graphe de connaissance ────────────
graph = nx.DiGraph()

# ── Fonctions mémoire ───────────────────────────────────

def save_to_memory(agent_name: str, round_num: int, content: str):
    """Sauvegarde la réponse d'un agent dans ChromaDB."""
    doc_id = f"{agent_name}_round_{round_num}"
    memory.add(
        documents=[content],
        metadatas=[{"agent": agent_name, "round": round_num}],
        ids=[doc_id]
    )
    print(f"  💾 Memory saved: {doc_id}")

def get_memory(agent_name: str, query: str, n_results: int = 3) -> str:
    """Récupère les souvenirs pertinents d'un agent."""
    try:
        results = memory.query(
            query_texts=[query],
            n_results=n_results,
            where={"agent": agent_name}
        )
        if results["documents"] and results["documents"][0]:
            return "\n---\n".join(results["documents"][0])
        return "No previous memory found."
    except Exception:
        return "No previous memory found."

def get_all_memory(query: str, n_results: int = 5) -> str:
    """Récupère les souvenirs de tous les agents."""
    try:
        results = memory.query(
            query_texts=[query],
            n_results=n_results
        )
        if results["documents"] and results["documents"][0]:
            docs = []
            for doc, meta in zip(
                results["documents"][0],
                results["metadatas"][0]
            ):
                docs.append(
                    f"[{meta['agent']} - Round {meta['round']}]: {doc}"
                )
            return "\n---\n".join(docs)
        return "No memory found."
    except Exception:
        return "No memory found."

# ── Fonctions graphe ─────────────────────────────────────

def update_graph(agent_name: str, concepts: list, round_num: int):
    """Ajoute les concepts et relations au graphe."""
    graph.add_node(agent_name, type="agent")
    for concept in concepts:
        graph.add_node(concept, type="concept")
        graph.add_edge(
            agent_name,
            concept,
            round=round_num,
            weight=1
        )
    print(f"  🔗 Graph updated: {len(concepts)} concepts added for {agent_name}")

def print_graph_summary():
    """Affiche un résumé du graphe."""
    print(f"\n📊 KNOWLEDGE GRAPH SUMMARY")
    print(f"   Nodes : {graph.number_of_nodes()}")
    print(f"   Edges : {graph.number_of_edges()}")
    print(f"   Agents: {[n for n,d in graph.nodes(data=True) if d.get('type')=='agent']}")
    print(f"   Top concepts:")
    concepts = [
        (n, graph.in_degree(n))
        for n, d in graph.nodes(data=True)
        if d.get("type") == "concept"
    ]
    concepts.sort(key=lambda x: x[1], reverse=True)
    for concept, degree in concepts[:8]:
        print(f"     - {concept} (mentioned {degree}x)")

# ── Définition des agents ────────────────────────────────

python_dev = Agent(
    role="Senior Python Developer",
    goal="Evaluate the technical feasibility of migrating Python applications from OpenShift to a private cloud",
    backstory="""You are a senior Python developer with 10 years of experience.
    You maintain Python applications (Django, FastAPI, Celery) running on OpenShift.
    You care about performance, developer experience, and avoiding vendor lock-in.
    You have experience with AWS Lambda, GCP Cloud Run, and Azure App Service.""",
    verbose=False
)

devops = Agent(
    role="DevOps Engineer",
    goal="Assess infrastructure and operational implications of migrating from OpenShift to AWS, Azure, or GCP",
    backstory="""You are a DevOps engineer managing the OpenShift cluster, ArgoCD
    pipelines, Helm charts, and monitoring stack (Prometheus, Grafana).
    You have hands-on experience with EKS, AKS, and GKE.
    You are concerned about migration effort, downtime risk, and hidden costs.""",
    verbose=False
)

project_manager = Agent(
    role="Project Manager",
    goal="Make the final strategic recommendation on cloud migration for the CTO and the board",
    backstory="""You are a project manager with a 500,000 euros budget and a team
    of 8 people. The current OpenShift cluster costs 120,000 euros/year.
    Your goal is to reduce costs by 30% and improve scalability within 18 months.
    You need a clear, justified recommendation to present to the board.""",
    verbose=False
)

# ── Simulation multi-rounds ──────────────────────────────

ROUNDS = 3
round_results = {}

print("\n" + "="*60)
print("🚀 SWARM INTELLIGENCE SIMULATION — PHASE 2")
print("   Topic: OpenShift → Cloud Migration (AWS/Azure/GCP)")
print("   Rounds:", ROUNDS)
print("="*60 + "\n")

for round_num in range(1, ROUNDS + 1):

    print(f"\n{'='*60}")
    print(f"  ROUND {round_num}/{ROUNDS}")
    print(f"{'='*60}\n")

    # ── Contexte mémoire pour ce round ──────────────────
    if round_num == 1:
        memory_context = "This is the first round. Give your initial position."
    else:
        memory_context = get_all_memory(
            query="cloud migration OpenShift AWS Azure GCP",
            n_results=6
        )

    # ── Tâche développeur ────────────────────────────────
    dev_task = Task(
        description=f"""ROUND {round_num}/{ROUNDS} — Cloud Migration Analysis

Previous debate context:
{memory_context}

Your task:
{"Analyze the technical feasibility of migrating Python apps (Django, FastAPI, Celery) from OpenShift to AWS, Azure, or GCP. Cover: technical challenges, best managed Kubernetes service (EKS vs AKS vs GKE), vendor lock-in risk, required code changes." if round_num == 1 else f"You are in round {round_num}. Review what was said before and REFINE your position. Have you changed your mind on any point? What new technical argument can you add? Address the DevOps and PM concerns raised in previous rounds."}

Be specific. Max 250 words.""",
        expected_output=f"Round {round_num} technical analysis from the Python developer",
        agent=python_dev
    )

    # ── Tâche DevOps ─────────────────────────────────────
    devops_task = Task(
        description=f"""ROUND {round_num}/{ROUNDS} — Infrastructure Assessment

Previous debate context:
{memory_context}

Your task:
{"Evaluate infrastructure implications: OpenShift vs EKS/AKS/GKE comparison, migration timeline, CI/CD impact (ArgoCD, Helm), monitoring stack migration, hidden operational costs." if round_num == 1 else f"You are in round {round_num}. Review the previous arguments. What do you agree or disagree with? Refine your infrastructure assessment based on what the developer and PM said. Add new operational insights."}

Be specific. Max 250 words.""",
        expected_output=f"Round {round_num} infrastructure assessment from the DevOps engineer",
        agent=devops
    )

    # ── Tâche PM ─────────────────────────────────────────
    pm_task = Task(
        description=f"""ROUND {round_num}/{ROUNDS} — Strategic Decision

Previous debate context:
{memory_context}

Your task:
{"Based on developer and DevOps inputs, give your initial strategic assessment: should we migrate? Which cloud provider? What is the estimated cost vs savings? What are the top 3 risks?" if round_num == 1 else f"You are in round {round_num}. {'This is the FINAL round. Synthesize all arguments from all rounds and give your DEFINITIVE recommendation to the board. Include: final cloud provider choice, migration roadmap (3 phases), total cost estimate, top 3 risks and mitigations, and one clear concluding sentence.' if round_num == ROUNDS else 'Refine your strategic position. What new information from the developer and DevOps changed your view? Update your cost and risk assessment.'}"}

Be specific. Max 250 words.""",
        expected_output=f"Round {round_num} strategic recommendation from the Project Manager",
        agent=project_manager
    )

    # ── Exécution du round ───────────────────────────────
    crew = Crew(
        agents=[python_dev, devops, project_manager],
        tasks=[dev_task, devops_task, pm_task],
        verbose=False
    )

    print(f"⏳ Running round {round_num}...")
    result = crew.kickoff()

    # ── Sauvegarde des résultats dans la mémoire ─────────
    tasks_output = crew.tasks_output if hasattr(crew, 'tasks_output') else []

    dev_output = tasks_output[0].raw if len(tasks_output) > 0 else str(result)
    devops_output = tasks_output[1].raw if len(tasks_output) > 1 else str(result)
    pm_output = tasks_output[2].raw if len(tasks_output) > 2 else str(result)

    save_to_memory("python_developer", round_num, dev_output)
    save_to_memory("devops_engineer", round_num, devops_output)
    save_to_memory("project_manager", round_num, pm_output)

    # ── Mise à jour du graphe ────────────────────────────
    update_graph("python_developer", [
        "Python apps", "vendor lock-in", "EKS", "AKS", "GKE",
        "Django", "FastAPI", "Celery", "containerization"
    ], round_num)

    update_graph("devops_engineer", [
        "ArgoCD", "Helm", "Prometheus", "Grafana",
        "migration timeline", "downtime risk", "CI/CD", "OpenShift"
    ], round_num)

    update_graph("project_manager", [
        "budget 500k", "cost reduction 30%", "18 months",
        "board recommendation", "ROI", "risk mitigation"
    ], round_num)

    round_results[round_num] = {
        "developer": dev_output,
        "devops": devops_output,
        "pm": pm_output
    }

    print(f"\n✅ Round {round_num} completed")
    print(f"\n--- PM Position (Round {round_num}) ---")
    print(pm_output[:500] + "..." if len(pm_output) > 500 else pm_output)

# ── Rapport final ────────────────────────────────────────

print("\n" + "="*60)
print("📋 FINAL STRATEGIC RECOMMENDATION")
print("="*60)
print(round_results[ROUNDS]["pm"])

print_graph_summary()

# ── Sauvegarde JSON ──────────────────────────────────────
with open("simulation_results.json", "w") as f:
    json.dump(round_results, f, indent=2)
print("\n💾 Full results saved to simulation_results.json")
