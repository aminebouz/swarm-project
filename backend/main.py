import os
import json
import asyncio
import networkx as nx
import chromadb
from typing import TypedDict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── State ────────────────────────────────────────────────

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

# ── WebSocket Manager ────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        self.active.remove(ws)

    async def broadcast(self, msg: dict):
        for ws in self.active:
            try:
                await ws.send_json(msg)
            except Exception:
                pass

manager = ConnectionManager()

# ── Graph Data ───────────────────────────────────────────

graph_data = {"nodes": [], "edges": []}
knowledge_graph = nx.DiGraph()

def update_knowledge_graph(agent: str, concepts: list, round_num: int):
    knowledge_graph.add_node(agent, type="agent")
    for concept in concepts:
        knowledge_graph.add_node(concept, type="concept")
        if knowledge_graph.has_edge(agent, concept):
            knowledge_graph[agent][concept]["weight"] += 1
        else:
            knowledge_graph.add_edge(
                agent, concept, round=round_num, weight=1
            )

    graph_data["nodes"] = [
        {"id": n, "type": d.get("type", "concept")}
        for n, d in knowledge_graph.nodes(data=True)
    ]
    graph_data["edges"] = [
        {"source": u, "target": v, "weight": d.get("weight", 1)}
        for u, v, d in knowledge_graph.edges(data=True)
    ]

# ── Simulation ───────────────────────────────────────────

async def run_simulation(max_rounds: int, topic: str):
    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
        model=os.getenv("OPENAI_MODEL_NAME"),
        temperature=0.7
    )

    chroma_client = chromadb.Client()
    try:
        chroma_client.delete_collection("swarm_memory")
    except Exception:
        pass
    memory = chroma_client.get_or_create_collection(name="swarm_memory")
    knowledge_graph.clear()

    def save_memory(agent, round_num, content):
        doc_id = f"{agent}_round_{round_num}"
        memory.add(
            documents=[content],
            metadatas=[{"agent": agent, "round": round_num}],
            ids=[doc_id]
        )

    def get_memory_context(query, n=6):
        try:
            results = memory.query(query_texts=[query], n_results=n)
            if results["documents"] and results["documents"][0]:
                docs = []
                for doc, meta in zip(
                    results["documents"][0],
                    results["metadatas"][0]
                ):
                    docs.append(
                        f"[{meta['agent']} - Round {meta['round']}]:"
                        f"\n{doc}"
                    )
                return "\n---\n".join(docs)
            return "No previous context."
        except Exception:
            return "No previous context."

    def check_consensus(state):
        current = state["pm_position"].lower()
        previous = state["pm_previous_position"].lower()
        if not previous:
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
        return len(intersection) / len(union)

    async def developer_node(state):
        round_num = state["round_num"]
        await manager.broadcast({
            "type": "log",
            "agent": "developer",
            "round": round_num,
            "message": f"[Round {round_num}] 🐍 Python Developer thinking..."
        })
        context = get_memory_context(
            "Python migration OpenShift EKS AKS GKE"
        )
        instruction = (
            f"Analyze the technical feasibility of migrating Python apps "
            f"(Django, FastAPI, Celery) from OpenShift for this topic: "
            f"{topic}. Cover EKS vs AKS vs GKE, vendor lock-in, code "
            f"changes needed. Max 200 words."
            if round_num == 1 else
            f"Round {round_num}: Refine your technical position based on "
            f"previous debate. Have you changed your mind? Max 200 words."
        )
        messages = [
            SystemMessage(content=(
                "You are a Senior Python Developer maintaining Django, "
                "FastAPI, and Celery apps on OpenShift."
            )),
            HumanMessage(
                content=f"Context:\n{context}\n\n{instruction}"
            )
        ]
        response = llm.invoke(messages)
        content = response.content
        save_memory("python_developer", round_num, content)
        update_knowledge_graph(
            "python_developer",
            ["EKS", "AKS", "GKE", "Django",
             "FastAPI", "vendor lock-in"],
            round_num
        )
        await manager.broadcast({
            "type": "agent_result",
            "agent": "developer",
            "round": round_num,
            "content": content
        })
        await manager.broadcast({"type": "graph", "data": graph_data})
        return {**state, "developer_position": content}

    async def devops_node(state):
        round_num = state["round_num"]
        await manager.broadcast({
            "type": "log",
            "agent": "devops",
            "round": round_num,
            "message": f"[Round {round_num}] 🔧 DevOps Engineer thinking..."
        })
        context = get_memory_context(
            "infrastructure ArgoCD Helm Prometheus migration"
        )
        instruction = (
            f"Evaluate infrastructure implications for: {topic}. "
            f"Cover OpenShift vs EKS/AKS/GKE, migration timeline, "
            f"CI/CD impact, monitoring. Max 200 words."
            if round_num == 1 else
            f"Round {round_num}: Refine your infrastructure assessment. "
            f"Max 200 words."
        )
        messages = [
            SystemMessage(content=(
                "You are a DevOps Engineer managing OpenShift with "
                "ArgoCD, Helm, Prometheus, Grafana."
            )),
            HumanMessage(content=(
                f"Developer:\n{state['developer_position']}\n\n"
                f"Context:\n{context}\n\n{instruction}"
            ))
        ]
        response = llm.invoke(messages)
        content = response.content
        save_memory("devops_engineer", round_num, content)
        update_knowledge_graph(
            "devops_engineer",
            ["ArgoCD", "Helm", "Prometheus",
             "CI/CD", "OpenShift", "downtime"],
            round_num
        )
        await manager.broadcast({
            "type": "agent_result",
            "agent": "devops",
            "round": round_num,
            "content": content
        })
        await manager.broadcast({"type": "graph", "data": graph_data})
        return {**state, "devops_position": content}

    async def pm_node(state):
        round_num = state["round_num"]
        await manager.broadcast({
            "type": "log",
            "agent": "pm",
            "round": round_num,
            "message": f"[Round {round_num}] 📊 Project Manager thinking..."
        })
        context = get_memory_context(
            "cost savings recommendation board budget"
        )
        instruction = (
            "FINAL ROUND: Give DEFINITIVE recommendation: cloud provider, "
            "3-phase roadmap, total cost vs savings, top 3 risks. "
            "Max 250 words."
            if round_num == state["max_rounds"] else
            (
                f"Initial assessment for: {topic}. Migrate or not? "
                f"Which cloud? Cost estimate? Top 3 risks? Max 200 words."
                if round_num == 1 else
                f"Round {round_num}: Refine your position. "
                f"Max 200 words."
            )
        )
        messages = [
            SystemMessage(content=(
                "You are a Project Manager with 500,000 euros budget. "
                "OpenShift costs 120,000 euros/year. Goal: reduce costs "
                "30% in 18 months."
            )),
            HumanMessage(content=(
                f"Developer:\n{state['developer_position']}\n\n"
                f"DevOps:\n{state['devops_position']}\n\n"
                f"Previous PM:\n"
                f"{state['pm_position'] or 'First round.'}\n\n"
                f"Context:\n{context}\n\n{instruction}"
            ))
        ]
        response = llm.invoke(messages)
        content = response.content
        save_memory("project_manager", round_num, content)
        update_knowledge_graph(
            "project_manager",
            ["budget", "cost reduction", "18 months", "ROI"],
            round_num
        )
        history = state["history"] + [{
            "round": round_num,
            "developer": state["developer_position"],
            "devops": state["devops_position"],
            "pm": content
        }]
        await manager.broadcast({
            "type": "agent_result",
            "agent": "pm",
            "round": round_num,
            "content": content
        })
        await manager.broadcast({"type": "graph", "data": graph_data})
        return {
            **state,
            "pm_previous_position": state["pm_position"],
            "pm_position": content,
            "history": history
        }

    async def consensus_node(state):
        round_num = state["round_num"]
        score = check_consensus(state)
        consensus = score >= 0.75 or round_num >= state["max_rounds"]
        msg = (
            f"🤝 Consensus reached! Score: {score:.2f}"
            if consensus and score >= 0.75 else
            f"🏁 Max rounds reached ({state['max_rounds']})"
            if consensus else
            f"🔄 No consensus yet (score: {score:.2f}). "
            f"Round {round_num + 1} starting..."
        )
        await manager.broadcast({
            "type": "consensus",
            "score": score,
            "reached": consensus,
            "message": msg
        })
        return {
            **state,
            "consensus_reached": consensus,
            "consensus_score": score,
            "round_num": round_num + 1
        }

    async def final_report_node(state):
        final = state["pm_position"]
        await manager.broadcast({
            "type": "final_report",
            "content": final,
            "graph": graph_data,
            "total_rounds": state["round_num"] - 1,
            "consensus_score": state["consensus_score"]
        })
        return {**state, "final_report": final}

    def should_continue(state):
        return (
            "final_report"
            if state["consensus_reached"]
            else "developer"
        )

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
        {"developer": "developer", "final_report": "final_report"}
    )
    workflow.add_edge("final_report", END)
    simulation_app = workflow.compile()

    initial_state: SimulationState = {
        "round_num": 1,
        "max_rounds": max_rounds,
        "consensus_reached": False,
        "consensus_score": 0.0,
        "developer_position": "",
        "devops_position": "",
        "pm_position": "",
        "pm_previous_position": "",
        "history": [],
        "final_report": ""
    }

    await simulation_app.ainvoke(initial_state)

# ── Routes ───────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    await ws.send_json({"type": "connected", "message": "Ready"})
    try:
        while True:
            data = await ws.receive_json()
            if data.get("action") == "start":
                max_rounds = data.get("max_rounds", 3)
                topic = data.get("topic", "OpenShift to cloud migration")
                await manager.broadcast({
                    "type": "log",
                    "agent": "system",
                    "round": 0,
                    "message": (
                        f"🚀 Simulation started — "
                        f"Topic: {topic} | Max rounds: {max_rounds}"
                    )
                })
                asyncio.create_task(
                    run_simulation(max_rounds, topic)
                )
    except WebSocketDisconnect:
        manager.disconnect(ws)
