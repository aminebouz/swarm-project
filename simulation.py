from crewai import Agent, Task, Crew
from dotenv import load_dotenv

load_dotenv()

# ── Agents ──────────────────────────────────────────────

python_dev = Agent(
    role="Senior Python Developer",
    goal="Evaluate the technical feasibility of migrating Python applications from OpenShift to a private cloud",
    backstory="""You are a senior Python developer with 10 years of experience.
    You have built and maintained several Python applications currently running
    on an OpenShift cluster. You know the codebase inside out — the Django APIs,
    the Celery workers, the FastAPI microservices. You care deeply about
    application performance, developer experience, and avoiding vendor lock-in.
    You have some experience with AWS Lambda and GCP Cloud Run but you are
    skeptical about the complexity of a full migration.""",
    verbose=True
)

devops = Agent(
    role="DevOps Engineer",
    goal="Assess the infrastructure, CI/CD, and operational implications of migrating from OpenShift to AWS, Azure, or GCP",
    backstory="""You are a DevOps engineer with deep expertise in containers,
    Kubernetes, and CI/CD pipelines. You currently manage the OpenShift cluster
    and all the Helm charts, ArgoCD pipelines, and monitoring stack (Prometheus,
    Grafana). You have hands-on experience with AWS EKS, Azure AKS, and GCP GKE.
    You know that OpenShift is essentially Kubernetes with Red Hat enterprise
    features on top. You are concerned about the operational cost, the migration
    effort, and the risk of downtime during the transition.""",
    verbose=True
)

project_manager = Agent(
    role="Project Manager",
    goal="Make the final strategic recommendation on whether to migrate to a private cloud and which provider to choose",
    backstory="""You are an experienced project manager with a background in
    cloud transformation projects. You manage a team of 8 people, a budget of
    500,000 euros for this year, and you report to the CTO. You need to balance
    technical quality, cost, timeline, and business risk. The current OpenShift
    cluster costs 120,000 euros per year in licensing and infrastructure.
    The business objective is to reduce costs by 30% and improve scalability
    within 18 months. You are neutral on the cloud provider choice but you
    need a clear, justified recommendation to present to the board.""",
    verbose=True
)

# ── Tâches ──────────────────────────────────────────────

dev_task = Task(
    description="""Analyze the technical feasibility of migrating the current
    Python applications from OpenShift to a private cloud (AWS, Azure, or GCP).

    Cover the following points:
    - What are the main technical challenges of migrating Python apps
      (Django, FastAPI, Celery) from OpenShift to each cloud provider?
    - Which cloud provider offers the best managed Kubernetes service
      for Python workloads (EKS vs AKS vs GKE)?
    - What is the risk of vendor lock-in for each provider?
    - What would need to change in the codebase or architecture?

    Give a clear technical recommendation with justification.
    Max 300 words.""",
    expected_output="Technical feasibility analysis with a recommended cloud provider from a developer perspective",
    agent=python_dev
)

devops_task = Task(
    description="""Based on the developer's analysis, evaluate the infrastructure
    and operational implications of migrating from OpenShift to the cloud.

    Cover the following points:
    - How does OpenShift compare to EKS (AWS), AKS (Azure), and GKE (GCP)
      in terms of operational complexity and migration effort?
    - What is the estimated migration timeline and risk of downtime?
    - How would the current CI/CD pipelines (ArgoCD, Helm) be affected?
    - What would the monitoring stack (Prometheus, Grafana) look like
      on each provider?
    - What are the hidden operational costs beyond the license savings?

    Give a clear infrastructure recommendation with justification.
    Max 300 words.""",
    expected_output="Infrastructure assessment with estimated migration effort and recommended cloud provider from a DevOps perspective",
    agent=devops
)

pm_task = Task(
    description="""Based on the developer's and DevOps engineer's analyses,
    make the final strategic recommendation for the CTO and the board.

    Cover the following points:
    - Should the team migrate from OpenShift to a private cloud, or stay
      on OpenShift? Justify the decision.
    - If migration is recommended, which cloud provider (AWS, Azure, or GCP)
      and why?
    - What is the proposed migration roadmap (phases, timeline, milestones)?
    - What is the estimated total cost of migration vs the expected savings?
    - What are the top 3 risks and how to mitigate them?

    Conclude with a single, clear recommendation sentence for the board.
    Max 300 words.""",
    expected_output="Strategic recommendation report with cloud provider choice, roadmap, cost estimate, and risk mitigation plan",
    agent=project_manager
)

# ── Crew ────────────────────────────────────────────────

crew = Crew(
    agents=[python_dev, devops, project_manager],
    tasks=[dev_task, devops_task, pm_task],
    verbose=True
)

# ── Lancement ───────────────────────────────────────────

print("\n🚀 Starting cloud migration debate...\n")
result = crew.kickoff()
print("\n📋 FINAL STRATEGIC RECOMMENDATION:\n")
print(result)
