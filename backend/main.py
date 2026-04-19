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

class SimulationState(TypedDict):
    round_num: int
    max_rounds: int
    consensus_reached: bool
    consensus_score: float
    positions: dict
    previous_positions: dict
    history: list
    final_report: str

class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, msg: dict):
        for ws in self.active:
            try:
                await ws.send_json(msg)
            except Exception:
                pass

manager = ConnectionManager()
graph_data = {"nodes": [], "edges": []}
knowledge_graph = nx.DiGraph()

def update_knowledge_graph(agent_name: str, concepts: list, round_num: int):
    knowledge_graph.add_node(agent_name, type="agent")
    for concept in concepts:
        knowledge_graph.add_node(concept, type="concept")
        if knowledge_graph.has_edge(agent_name, concept):
            knowledge_graph[agent_name][concept]["weight"] += 1
        else:
            knowledge_graph.add_edge(
                agent_name, concept, round=round_num, weight=1
            )
    graph_data["nodes"] = [
        {"id": n, "type": d.get("type", "concept")}
        for n, d in knowledge_graph.nodes(data=True)
    ]
    graph_data["edges"] = [
        {"source": u, "target": v, "weight": d.get("weight", 1)}
        for u, v, d in knowledge_graph.edges(data=True)
    ]

def check_consensus(positions: dict, previous_positions: dict) -> float:
    if not previous_positions:
        return 0.0
    scores = []
    keywords = [
        "recommend", "agree", "conclude", "decision",
        "should", "must", "will", "propose", "suggest"
    ]
    for agent_id in positions:
        if agent_id not in previous_positions:
            continue
        current = positions[agent_id].lower()
        previous = previous_positions[agent_id].lower()
        current_hits = set(k for k in keywords if k in current)
        previous_hits = set(k for k in keywords if k in previous)
        if not current_hits or not previous_hits:
            scores.append(0.0)
            continue
        intersection = current_hits & previous_hits
        union = current_hits | previous_hits
        scores.append(len(intersection) / len(union))
    return sum(scores) / len(scores) if scores else 0.0

async def run_simulation(
    max_rounds: int,
    topic: str,
    agents: list
):
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
    graph_data["nodes"] = []
    graph_data["edges"] = []

    def save_memory(agent_id: str, round_num: int, content: str):
        doc_id = f"{agent_id}_round_{round_num}"
        memory.add(
            documents=[content],
            metadatas=[{"agent": agent_id, "round": round_num}],
            ids=[doc_id]
        )

    def get_memory_context(query: str, n: int = 8) -> str:
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

    def make_agent_node(agent: dict):
        agent_id = agent["id"]
        agent_name = agent["name"]
        agent_profile = agent["profile"]
        agent_description = agent["description"]

        async def agent_node(state: SimulationState):
            round_num = state["round_num"]
            await manager.broadcast({
                "type": "log",
                "agent": agent_id,
                "agent_name": agent_name,
                "round": round_num,
                "message": (
                    f"[Round {round_num}] 🤖 {agent_name} thinking..."
                )
            })

            context = get_memory_context(topic)

            if round_num == 1:
                instruction = (
                    f"This is round 1. Give your initial position on "
                    f"the following topic: {topic}\n\n"
                    f"Your role and perspective: {agent_description}\n\n"
                    f"Be specific, structured and argue from your "
                    f"expertise. Max 250 words."
                )
            elif round_num == state["max_rounds"]:
                instruction = (
                    f"This is the FINAL round ({round_num}/{state['max_rounds']}).\n"
                    f"Topic: {topic}\n\n"
                    f"Synthesize all previous arguments and give your "
                    f"DEFINITIVE final position. Have you changed your "
                    f"mind compared to round 1? What is your conclusion? "
                    f"Max 250 words."
                )
            else:
                other_positions = "\n\n".join([
                    f"[{k}]: {v[:300]}..."
                    for k, v in state["positions"].items()
                    if k != agent_id and v
                ])
                instruction = (
                    f"Round {round_num}/{state['max_rounds']}.\n"
                    f"Topic: {topic}\n\n"
                    f"Other agents' positions this round:\n"
                    f"{other_positions}\n\n"
                    f"Review the debate so far and REFINE your position. "
                    f"Do you agree or disagree with the others? "
                    f"Have you changed your mind on any point? "
                    f"Max 250 words."
                )

            messages = [
                SystemMessage(content=agent_profile),
                HumanMessage(
                    content=(
                        f"Previous debate context:\n{context}\n\n"
                        f"{instruction}"
                    )
                )
            ]

            await asyncio.sleep(1)
            response = llm.invoke(messages)
            content = response.content

            save_memory(agent_id, round_num, content)
            update_knowledge_graph(
                agent_name,
                [w for w in topic.split() if len(w) > 4][:6],
                round_num
            )

            await manager.broadcast({
                "type": "agent_result",
                "agent": agent_id,
                "agent_name": agent_name,
                "round": round_num,
                "content": content
            })
            await manager.broadcast({
                "type": "graph",
                "data": graph_data
            })

            new_positions = {**state["positions"], agent_id: content}
            return {**state, "positions": new_positions}

        agent_node.__name__ = agent_id
        return agent_node

    async def consensus_node(state: SimulationState):
        round_num = state["round_num"]
        score = check_consensus(
            state["positions"],
            state["previous_positions"]
        )
        consensus = (
            score >= 0.75 or round_num >= state["max_rounds"]
        )

        msg = (
            f"🤝 Consensus reached! Score: {score:.2f}"
            if consensus and score >= 0.75 else
            f"🏁 Max rounds reached ({state['max_rounds']}). "
            f"Generating final report..."
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
            "previous_positions": dict(state["positions"]),
            "history": state["history"] + [{
                "round": round_num,
                "positions": dict(state["positions"])
            }],
            "round_num": round_num + 1
        }

    async def final_report_node(state: SimulationState):
        last_positions = state["positions"]
        final = "\n\n".join([
            f"## {agents[i]['name']}\n{last_positions.get(a['id'], '')}"
            for i, a in enumerate(agents)
            if a["id"] in last_positions
        ])

        await manager.broadcast({
            "type": "final_report",
            "content": final,
            "graph": graph_data,
            "total_rounds": state["round_num"] - 1,
            "consensus_score": state["consensus_score"]
        })
        return {**state, "final_report": final}

    def should_continue(state: SimulationState) -> str:
        if state["consensus_reached"]:
            return "final_report"
        return agents[0]["id"]

    # ── Build dynamic graph ──────────────────────────────
    workflow = StateGraph(SimulationState)

    for agent in agents:
        workflow.add_node(agent["id"], make_agent_node(agent))

    workflow.add_node("consensus_check", consensus_node)
    workflow.add_node("final_report", final_report_node)

    workflow.set_entry_point(agents[0]["id"])

    for i in range(len(agents) - 1):
        workflow.add_edge(agents[i]["id"], agents[i + 1]["id"])

    workflow.add_edge(agents[-1]["id"], "consensus_check")

    workflow.add_conditional_edges(
        "consensus_check",
        should_continue,
        {
            agents[0]["id"]: agents[0]["id"],
            "final_report": "final_report"
        }
    )
    workflow.add_edge("final_report", END)

    simulation_app = workflow.compile()

    initial_positions = {a["id"]: "" for a in agents}

    initial_state: SimulationState = {
        "round_num": 1,
        "max_rounds": max_rounds,
        "consensus_reached": False,
        "consensus_score": 0.0,
        "positions": initial_positions,
        "previous_positions": {},
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
                topic = data.get("topic", "")
                agents = data.get("agents", [])

                if not topic or not agents:
                    await ws.send_json({
                        "type": "error",
                        "message": "Topic and agents are required."
                    })
                    continue

                for i, a in enumerate(agents):
                    a["id"] = f"agent_{i}_{a['name'].lower().replace(' ', '_')}"

                await manager.broadcast({
                    "type": "log",
                    "agent": "system",
                    "round": 0,
                    "message": (
                        f"🚀 Simulation started — "
                        f"Topic: {topic} | "
                        f"Agents: {len(agents)} | "
                        f"Max rounds: {max_rounds}"
                    )
                })

                asyncio.create_task(
                    run_simulation(max_rounds, topic, agents)
                )
    except WebSocketDisconnect:
        manager.disconnect(ws)
