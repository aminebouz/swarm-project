from crewai import Agent, Task, Crew
from dotenv import load_dotenv

load_dotenv()

# ── Agents ──────────────────────────────────────────────

ceo = Agent(
    role="Optimistic CEO",
    goal="Push for launching the product as soon as possible",
    backstory="""You are an ambitious startup CEO. You believe the market 
    window is now and waiting means losing to competitors. You focus on 
    opportunities and growth potential.""",
    verbose=True
)

cfo = Agent(
    role="Cautious CFO",
    goal="Protect the company's financial health",
    backstory="""You are a pragmatic CFO. You always ask for data before 
    any decision. You worry about burn rate, LTV/CAC ratio, and runway. 
    You support launches only when the numbers make sense.""",
    verbose=True
)

engineer = Agent(
    role="Skeptical Engineer",
    goal="Ensure technical quality before any launch",
    backstory="""You are the lead engineer. You have seen too many rushed 
    launches that damaged user trust. You insist on fixing the race condition 
    in the payment module before going live.""",
    verbose=True
)

# ── Tâches ──────────────────────────────────────────────

ceo_task = Task(
    description="""Argue why the startup should launch its new product NOW.
    Address market timing, competitor threats, and growth opportunity.
    Be specific and persuasive. Max 200 words.""",
    expected_output="A strong argument for immediate launch",
    agent=ceo
)

cfo_task = Task(
    description="""Respond to the CEO's argument. Analyze the financial risks
    of launching now vs waiting 4 more weeks. Include specific metrics
    like burn rate, runway, and break-even point. Max 200 words.""",
    expected_output="A financial risk assessment with specific numbers",
    agent=cfo
)

engineer_task = Task(
    description="""Given the CEO and CFO arguments, give your final technical
    verdict. Is the product ready? What is the ONE critical bug that must be
    fixed before launch? Propose a compromise solution. Max 200 words.""",
    expected_output="Technical verdict and compromise proposal",
    agent=engineer
)

# ── Crew ────────────────────────────────────────────────

crew = Crew(
    agents=[ceo, cfo, engineer],
    tasks=[ceo_task, cfo_task, engineer_task],
    verbose=True
)

# ── Lancement ───────────────────────────────────────────

print("\n🚀 Starting simulation...\n")
result = crew.kickoff()
print("\n📋 FINAL REPORT:\n")
print(result)
