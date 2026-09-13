CPU SCHEDULER RL
=================

A CPU Scheduling Simulator, Reinforcement Learning Platform, and
Benchmarking Framework for Classical Operating System Schedulers.

PROJECT OVERVIEW
----------------

CPU Scheduler RL is an end-to-end systems and artificial intelligence
project that evaluates classical CPU scheduling algorithms against a
custom Reinforcement Learning (RL) agent.

The platform combines discrete-time CPU scheduling simulation,
Gymnasium-based reinforcement learning, PPO training, reproducible
benchmarking, and LLM-powered performance analysis through Groq.

Users can execute individual scheduling algorithms, configure
Round Robin time quantum, train and evaluate an RL policy, and compare
all approaches using consistent performance metrics.

The project is designed for local, CPU-based execution using free and
open-source software, with optional Groq API integration for intelligent
workload and benchmark analysis.

KEY FEATURES
------------

1. Discrete-time CPU scheduling simulation.
2. Dynamic process arrivals and ready queue management.
3. Classical scheduling algorithms:
   - First Come, First Served (FCFS)
   - Shortest Job First (SJF)
   - Shortest Remaining Time First (SRTF)
   - Priority Scheduling
   - Round Robin (RR)
4. Configurable Round Robin time quantum through the CLI.
5. Gymnasium-compatible reinforcement learning environment.
6. PPO-based RL agent using Stable-Baselines3.
7. Randomized workload generation for RL training.
8. Deterministic evaluation and reproducible experiments.
9. Action validation and fallback scheduling during inference.
10. Gantt chart visualization using Matplotlib.
11. Structured terminal results using tabulate.
12. CSV export of scheduling metrics.
13. Comparative benchmarking of classical and RL schedulers.
14. Groq LLM integration using openai/gpt-oss-120b.
15. Automatic Mock Mode when Groq credentials are unavailable.
16. Unit tests for environment, scheduling, and LLM functionality.
17. CPU-friendly local training and experimentation.

PROBLEM STATEMENT
-----------------

Classical CPU scheduling algorithms use predefined heuristics to
determine which process should receive CPU time.

Although these algorithms are effective for many workloads, their
performance depends on workload characteristics such as burst times,
arrival patterns, priorities, and process contention.

This project investigates whether a Reinforcement Learning agent can
learn scheduling policies that perform competitively with classical
algorithms under specific workload distributions and optimization
objectives.

The system provides a common simulation environment and evaluation
pipeline so that all algorithms can be compared fairly.

PROJECT GOALS
-------------

1. Simulate CPU scheduling accurately.
2. Implement and validate classical scheduling algorithms.
3. Train an RL policy using randomized workloads.
4. Evaluate the learned policy on unseen workloads.
5. Compare scheduling performance using consistent metrics.
6. Analyze experiment results using an optional LLM-powered analyst.
7. Provide a reproducible, professional research and engineering
   workflow.

ARCHITECTURE
------------

The platform consists of the following major components:

1. Process and Workload Layer

   Generates or loads processes with arrival time, burst time,
   priority, and remaining execution time.

2. Simulation Engine

   Advances CPU execution tick by tick, manages the ready queue,
   handles process arrivals and completion, and records scheduling
   events.

3. Classical Scheduler Layer

   Provides FCFS, SJF, SRTF, Priority, and Round Robin implementations.

4. Reinforcement Learning Layer

   Defines the CPU scheduling problem as a Gymnasium MDP and trains
   a PPO agent to select processes for execution.

5. Benchmarking Layer

   Runs all algorithms on identical workloads and calculates
   comparable performance metrics.

6. Visualization and Export Layer

   Generates Gantt charts, terminal tables, CSV reports, and
   comparative performance plots.

7. Groq LLM Analyst

   Uses openai/gpt-oss-120b through Groq to analyze workloads,
   interpret benchmark results, and explain algorithmic behavior.

HIGH-LEVEL PIPELINE
-------------------

User CLI Input
      |
      v
Workload Generator or JSON Workload Loader
      |
      v
Common CPU Simulation Engine
      |
      +-------------------------------+
      |                               |
      v                               v
Classical Scheduling Algorithms       RL Scheduling Environment
      |                               |
      |                               v
      |                         PPO Training
      |                               |
      |                               v
      |                         Trained RL Policy
      |                               |
      +---------------+---------------+
                      |
                      v
              Benchmark Engine
                      |
                      v
       Metrics, Gantt Charts, CSV Reports
                      |
                      v
             Optional Groq Analysis
                      |
                      v
               Final Results

SUPPORTED SCHEDULING ALGORITHMS
-------------------------------

1. FIRST COME, FIRST SERVED (FCFS)

   Type:
   Non-preemptive.

   Description:
   Processes are executed in the order in which they arrive in the
   ready queue.

   Characteristics:
   - Simple implementation.
   - Fair according to arrival order.
   - Can cause convoy effects.
   - Long processes may delay short processes.

2. SHORTEST JOB FIRST (SJF)

   Type:
   Non-preemptive.

   Description:
   Among the available processes, the process with the shortest total
   CPU burst time is selected.

   Characteristics:
   - Can minimize average waiting time under ideal assumptions.
   - Requires knowledge or estimation of burst time.
   - May cause starvation of long processes.

3. SHORTEST REMAINING TIME FIRST (SRTF)

   Type:
   Preemptive.

   Description:
   The process with the smallest remaining CPU execution time is
   selected. A running process may be preempted when a newly arrived
   process has a shorter remaining time.

   Characteristics:
   - Preemptive version of SJF.
   - Can improve response and waiting time.
   - Requires frequent scheduling decisions.
   - May increase context switches.

4. PRIORITY SCHEDULING

   Type:
   Non-preemptive.

   Description:
   The scheduler selects the ready process with the highest priority.

   Priority Convention:
   The project must document whether a smaller numerical value or
   larger numerical value represents higher priority.

   Characteristics:
   - Useful for workload differentiation.
   - Can cause starvation without aging.
   - Behavior depends on the priority convention.

5. ROUND ROBIN (RR)

   Type:
   Preemptive.

   Description:
   Each ready process receives CPU time for a fixed time quantum.
   When the quantum expires, the process is requeued if it still has
   remaining execution time.

   Configurable Parameter:
   Time quantum.

   Example:
   Quantum = 4

   Command:
   python inference.py --algorithm rr --quantum 4 --processes 5

   Characteristics:
   - Suitable for time-sharing systems.
   - Provides responsive scheduling.
   - Performance depends heavily on the time quantum.
   - Small quantums may increase context switches.
   - Large quantums make Round Robin behave more like FCFS.

SIMULATION ENGINE
-----------------

The platform uses a discrete-time CPU simulation engine.

Instead of relying only on analytical formulas, the simulator advances
execution tick by tick and maintains the actual state of every process.

PROCESS ATTRIBUTES
------------------

Each process contains:

1. Process ID (PID)
2. Arrival Time
3. Burst Time
4. Remaining Time
5. Priority
6. Start Time
7. Completion Time
8. Waiting Time
9. Response Time
10. Completion Status

SIMULATION RESPONSIBILITIES
---------------------------

The simulation engine handles:

1. CPU clock advancement.
2. Dynamic process arrivals.
3. Ready queue management.
4. Process selection.
5. Process execution.
6. Preemption.
7. Process completion.
8. CPU idle periods.
9. Context-switch tracking.
10. Gantt chart event generation.
11. Per-process metric calculation.
12. Aggregate workload metrics.

The engine provides a common result format so that every scheduling
algorithm can be evaluated consistently.

CORE PERFORMANCE METRICS
------------------------

1. WAITING TIME

   Waiting time represents the total time a process spends waiting
   in the ready queue.

   Formula:

   Waiting Time = Turnaround Time - Burst Time

2. TURNAROUND TIME

   Turnaround time is the total time from process arrival until
   process completion.

   Formula:

   Turnaround Time = Completion Time - Arrival Time

3. RESPONSE TIME

   Response time is the time between process arrival and its first
   allocation of CPU time.

   Formula:

   Response Time = First Start Time - Arrival Time

4. MAKESPAN

   Makespan is the total elapsed simulation time required to complete
   the workload.

   It is measured from the beginning of the simulation until the
   final process completes.

5. CONTEXT SWITCHES

   Context switches measure the number of times CPU execution changes
   from one process ID to another.

   The count is derived from generated Gantt chart execution events
   according to the project's documented counting convention.

6. CPU UTILIZATION

   CPU utilization represents the proportion of elapsed simulation
   time during which the CPU is executing a process.

   Formula:

   CPU Utilization = Busy CPU Time / Total Elapsed Time

7. THROUGHPUT

   Throughput represents the number of completed processes per unit
   of elapsed simulation time.

REINFORCEMENT LEARNING ENVIRONMENT
----------------------------------

The scheduling problem is formulated as a Markov Decision Process
using the Gymnasium API.

The RL environment is implemented through a randomized CPU workload
environment.

OBSERVATION SPACE
-----------------

The observation is a normalized NumPy float32 matrix of shape:

(max_processes, 5)

Each process row contains the following five features:

1. is_present

   Indicates whether the process slot is active.

2. is_ready

   Indicates whether the process has arrived and still has
   remaining execution time.

3. remaining_time

   The number of CPU ticks required to complete the process.

4. priority

   The static priority rank of the process.

5. wait_time

   The accumulated waiting time up to the current simulation tick.

The observation is normalized to improve neural network training
stability.

ACTION SPACE
------------

The action space is:

Discrete(max_processes)

Each integer action represents a process index.

For example:

Action 0 selects process slot 0.
Action 1 selects process slot 1.
Action 2 selects process slot 2.

The environment validates whether the selected process is eligible
for execution.

INVALID ACTION HANDLING
-----------------------

An invalid action may occur when the RL agent selects:

1. A process that has not arrived.
2. A process that has already completed.
3. A process with no remaining execution time.
4. An inactive process slot.

The environment applies a strong penalty to invalid actions.

During deterministic inference, a fallback policy selects the first
available valid process if the trained model produces an invalid
selection.

This prevents invalid scheduling decisions from causing infinite
loops or CPU simulation freezes.

REWARD FUNCTION
---------------

The current reward design uses a dense negative penalty.

For each simulation tick:

Reward = -Ready Queue Size

This penalizes the number of processes waiting in the ready queue.

An additional invalid-action penalty is applied:

Invalid Action Penalty = -10

The reward function is designed to encourage the agent to reduce
queue waiting and make valid scheduling decisions.

The reward is a training objective and should be interpreted separately
from the final benchmark metrics.

RL AGENT AND TRAINING
---------------------

The platform uses Proximal Policy Optimization (PPO) from
Stable-Baselines3.

MODEL ARCHITECTURE
------------------

Policy Type:
Multi-Layer Perceptron (MlpPolicy)

Neural Network Architecture:
Two hidden layers with 128 neurons each.

Configuration:

net_arch = [128, 128]

The network receives the flattened scheduling observation and
produces the policy and value estimates required by PPO.

TRAINING ENVIRONMENT
--------------------

The RL agent trains using a RandomizedCPUEnv.

A fresh randomized workload is generated on every environment reset.

This prevents the agent from simply memorizing one fixed schedule
and encourages learning scheduling behavior across different
workload configurations.

CURRENT TRAINING CONFIGURATION
------------------------------

Algorithm:
PPO

Learning Rate:
0.0003

Rollout Buffer Size:
1024 steps

Batch Size:
64

Entropy Coefficient:
0.01

Policy Network:
Two hidden layers of 128 neurons each.

The training configuration can be adjusted for larger experiments
and improved convergence.

CPU-FRIENDLY TRAINING
---------------------

The project is designed to run on a normal Windows laptop using
CPU-only execution.

Training timesteps can be adjusted depending on available resources.

Suggested experiments:

25,000 timesteps:
Quick development experiment.

50,000 timesteps:
Base training experiment.

100,000 timesteps:
Extended training experiment.

250,000 timesteps:
Longer training experiment for deeper convergence.

Training performance depends on workload complexity, observation
design, reward shaping, hardware, and hyperparameters.

The project does not assume that additional training will always
produce better scheduling performance.

GROQ LLM ANALYST
----------------

The platform includes an optional LLM-powered analysis component
using Groq.

Provider:
Groq

Model:
openai/gpt-oss-120b

The LLM analyst receives workload characteristics and comparative
benchmark metrics and produces an expert-style interpretation.

POSSIBLE ANALYSIS TASKS
-----------------------

1. Explain why FCFS performed well or poorly.
2. Analyze the effect of Round Robin time quantum.
3. Explain SJF and SRTF behavior for the current workload.
4. Interpret priority scheduling outcomes.
5. Compare RL performance with classical baselines.
6. Identify workload characteristics affecting performance.
7. Summarize benchmark results.
8. Suggest possible experiment improvements.

ARCHITECTURAL DISTINCTION
-------------------------

The Groq LLM analyst and the local RL policy are separate components.

The PPO agent is responsible for learning scheduling behavior inside
the simulation environment.

The Groq LLM is responsible for higher-level analysis and reasoning
about workload and benchmark results.

The LLM is not automatically the CPU scheduling policy.

If LLM-based scheduling decisions are implemented in the future,
they must use a strict action schema, validate every action, and be
benchmarked separately from the PPO agent.

FAULT TOLERANCE
---------------

The Groq integration supports Mock Mode.

Mock Mode is automatically used when API credentials are absent or
invalid, allowing local experiments and telemetry workflows to
continue without crashing.

The Groq API key must never be hardcoded.

Environment variable:

GROQ_API_KEY

The model name should remain configurable, with
openai/gpt-oss-120b as the default.

DYNAMIC EXPERIMENT PIPELINE
---------------------------

The experiment.py script provides an automated end-to-end experiment.

The pipeline:

1. Randomizes the workload size between 10 and 30 processes.
2. Generates a fresh workload.
3. Instantiates a PPO agent.
4. Trains the agent for a user-defined number of timesteps.
5. Runs classical scheduling algorithms.
6. Runs the trained RL scheduler.
7. Calculates performance metrics.
8. Sorts benchmark outcomes by performance.
9. Exports experiment results.
10. Optionally triggers Groq LLM analysis.

The pipeline supports quick experiments and longer training runs.

COMMAND-LINE INTERFACE
----------------------

The CLI is exposed through inference.py.

The user can select a scheduling algorithm and configure execution
parameters directly from the terminal.

ENVIRONMENT ACTIVATION
----------------------

Windows PowerShell:

.\.venv\Scripts\Activate.ps1

macOS/Linux:

source .venv/bin/activate

DYNAMIC EXPERIMENT COMMANDS
---------------------------

1. Standard dynamic experiment:

   python experiment.py

   Generates a random workload, trains the RL agent, benchmarks
   the algorithms, and triggers optional Groq analysis.

2. High-training experiment:

   python experiment.py --timesteps 250000

   Runs the dynamic experiment with 250,000 training timesteps.

3. Disable LLM analysis:

   python experiment.py --no-analyze

   Runs the experiment without triggering Groq analysis.

4. Reproducible experiment:

   python experiment.py --seed 42

   Uses a fixed random seed for reproducibility.

STANDALONE RL TRAINING
----------------------

1. Train a base model:

   python train.py

   Default configuration:
   50,000 timesteps.
   Maximum 5 processes.

2. Train with custom timesteps and process capacity:

   python train.py --timesteps 100000 --processes 10

3. Train with a custom output path and random seed:

   python train.py --timesteps 50000 --processes 8 --output models/my_custom_model --seed 123

The trained model is saved for later inference and benchmarking.

INDIVIDUAL ALGORITHM EXECUTION
------------------------------

1. FCFS:

   python inference.py --algorithm fcfs --processes 5

2. SJF:

   python inference.py --algorithm sjf --processes 5

3. SRTF:

   python inference.py --algorithm srtf --processes 5

4. Priority Scheduling:

   python inference.py --algorithm priority --processes 5

5. Round Robin with quantum 4:

   python inference.py --algorithm rr --quantum 4 --processes 5

6. Trained RL Agent:

   python inference.py --algorithm rl --model models/ppo_scheduler.zip --processes 5

BENCHMARKING COMMANDS
---------------------

1. Full comparative benchmark:

   python inference.py --algorithm benchmark --processes 5

   Runs classical algorithms and the RL agent on comparable workloads.

2. Benchmark with Groq analysis:

   python inference.py --algorithm benchmark --processes 5 --analyze

3. Export metrics and Gantt chart:

   python inference.py --algorithm sjf --processes 5 --csv results/sjf_metrics.csv --gantt results/sjf_chart.png --no-display

4. Load a custom workload:

   python inference.py --algorithm sjf --workload workloads/sample.json

BENCHMARKING METHODOLOGY
------------------------

All algorithms should be evaluated using identical workloads for
fair comparison.

The benchmark engine should support:

1. Fixed random seeds.
2. Configurable workload counts.
3. Separate training and evaluation workloads.
4. Raw per-workload results.
5. Aggregate metrics.
6. Mean and median performance.
7. Standard deviation.
8. Relevant percentiles.
9. CSV export.
10. Comparative plots.
11. RL-versus-classical analysis.

The RL agent should be evaluated on workloads that were not used
during training when measuring generalization.

The project does not assume that RL will outperform every classical
algorithm.

The results should identify the conditions under which each
algorithm performs well or poorly.

VISUALIZATION AND DATA EXPORT
-----------------------------

The platform provides visual and structured output for experiment
interpretation.

1. Terminal Tables

   Uses tabulate to display scheduling metrics in a readable format.

2. Gantt Charts

   Uses Matplotlib to display process execution intervals over time.

3. CSV Reports

   Exports per-process and aggregate metrics to CSV files.

4. Comparative Plots

   Displays differences in scheduling performance across algorithms.

5. LLM Reports

   Optionally generates natural-language explanations of benchmark
   results using Groq.

RECOMMENDED PROJECT STRUCTURE
-----------------------------
```
cpu-scheduler-rl/
|
+-- server/
|   +-- __init__.py
|   +-- app.py
|   +-- tests/
|       +-- test_env.py
|
+-- src/
|   +-- scheduler/
|       +-- __init__.py
|       +-- models.py
|       +-- workload.py
|       +-- metrics.py
|       |
|       +-- algorithms/
|       |   +-- __init__.py
|       |   +-- base.py
|       |   +-- fcfs.py
|       |   +-- sjf.py
|       |   +-- srtf.py
|       |   +-- priority.py
|       |   +-- round_robin.py
|       |
|       +-- simulation/
|       |   +-- __init__.py
|       |   +-- engine.py
|       |   +-- gantt.py
|       |
|       +-- rl/
|       |   +-- __init__.py
|       |   +-- environment.py
|       |   +-- agent.py
|       |   +-- train.py
|       |
|       +-- llm/
|       |   +-- __init__.py
|       |   +-- groq_agent.py
|       |   +-- prompts.py
|       |
|       +-- benchmark.py
|       +-- cli.py
|
+-- tests/
|   +-- test_algorithms.py
|   +-- test_env.py
|   +-- test_metrics.py
|   +-- test_rl.py
|   +-- test_llm_agent.py
|
+-- scripts/
|   +-- benchmark.py
|
+-- models/
+-- workloads/
+-- results/
+-- .env.example
+-- inference.py
+-- train.py
+-- experiment.py
+-- agent.py
+-- env.py
+-- grader.py
+-- tasks.py
+-- openenv.yaml
+-- pyproject.toml
+-- requirements.txt
+-- Dockerfile
+-- README.md
```
EXISTING REPOSITORY INTEGRATION
--------------------------------

The repository contains existing files including:

server/app.py
agent.py
env.py
grader.py
inference.py
tasks.py
openenv.yaml
pyproject.toml
requirements.txt
Dockerfile
README.md

These files should be inspected before major architectural changes.

Existing OpenEnv-related functionality should be preserved or adapted
where practical.

The project should avoid unnecessary duplication between root-level
scripts and the modular scheduler package.

TESTING
-------

Run the complete test suite using:

pytest -v

The test suite covers:

1. Gymnasium environment compliance.
2. Environment reset behavior.
3. Environment step behavior.
4. Observation shape and data types.
5. Action validation.
6. Classical algorithm correctness.
7. Process arrival handling.
8. Preemption behavior.
9. Round Robin time quantum.
10. Metric calculations.
11. Gantt timeline generation.
12. RL model loading.
13. Groq Mock Mode.
14. CLI argument validation.

Tests should use deterministic workloads where exact expected
results can be calculated manually.

INSTALLATION
------------

1. Clone the repository.

2. Navigate into the project directory.

3. Create a Python virtual environment:

   python -m venv .venv

4. Activate the environment.

   Windows PowerShell:
   .\.venv\Scripts\Activate.ps1

   macOS/Linux:
   source .venv/bin/activate

5. Install dependencies:

   pip install -r requirements.txt

6. Configure optional Groq credentials using environment variables.

7. Run the test suite:

   pytest -v

8. Execute an individual scheduler:

   python inference.py --algorithm fcfs --processes 5

FREE AND LOCAL-FIRST DESIGN
---------------------------

The project is designed to operate using free and open-source
software.

Core simulation and RL training do not require paid APIs.

The Groq LLM integration is optional and depends on available API
access, quotas, and rate limits.

When Groq is unavailable, Mock Mode allows local development and
benchmarking to continue.

The system does not require paid cloud GPUs or paid hosting.

LIMITATIONS
-----------

1. The initial RL environment uses a fixed maximum process capacity.
2. The observation space is based on a fixed-size process matrix.
3. RL performance depends on reward design and workload distribution.
4. Classical algorithms may outperform RL on many simple workloads.
5. The current simulation focuses on CPU-bound process execution.
6. Context-switch overhead may not fully represent real hardware
   scheduling costs.
7. The environment does not necessarily model I/O bursts or
   multi-core CPU scheduling.
8. LLM analysis depends on API availability and response quality.
9. Training time increases with workload complexity and timesteps.

FUTURE IMPROVEMENTS
-------------------

1. Multi-core CPU scheduling.
2. Context-switch execution overhead.
3. I/O-bound processes and CPU/I/O burst patterns.
4. Dynamic workload arrivals.
5. Priority aging to reduce starvation.
6. Variable-size process observations.
7. More advanced action masking.
8. Alternative RL algorithms such as DQN or A2C.
9. Multi-objective reward optimization.
10. Interactive scheduling dashboard.
11. Advanced benchmark statistics.
12. Hyperparameter optimization.
13. Statistical significance testing.
14. LLM-assisted experiment planning.
15. Scheduling policy explainability.
16. More realistic operating-system workload models.

RESEARCH AND ENGINEERING VALUE
------------------------------

This project demonstrates practical knowledge of:

1. Operating system scheduling algorithms.
2. Discrete-event and discrete-time simulation.
3. Queue management and process state transitions.
4. Reinforcement learning environments.
5. Markov Decision Processes.
6. PPO policy optimization.
7. Neural network-based decision-making.
8. Reproducible experimentation.
9. Benchmark design.
10. Performance metrics and visualization.
11. LLM integration with structured outputs.
12. Python software architecture.
13. Testing and debugging.
14. CLI application development.

RESULT INTERPRETATION
---------------------

The goal is not to assume that reinforcement learning is universally
better than classical scheduling.

Instead, the project evaluates whether a learned policy can discover
useful scheduling behavior under defined workload distributions.

A meaningful experiment should explain:

1. Which algorithm achieved the lowest average waiting time.
2. Which algorithm achieved the best response time.
3. How Round Robin quantum affected performance.
4. How SRTF responded to short jobs.
5. Whether the RL agent generalized to unseen workloads.
6. Where classical algorithms outperformed the RL policy.
7. Whether RL offered a useful trade-off between metrics.
8. How workload characteristics influenced the results.

All benchmark values must be generated from actual program execution.

LICENSE
-------

Add an appropriate open-source license to the repository.

ACKNOWLEDGEMENTS
-----------------

This project uses open-source technologies including Python,
Gymnasium, Stable-Baselines3, PyTorch, NumPy, Pandas, Matplotlib,
Pytest, and Groq's API.

AUTHOR
------

Ritam

PROJECT STATUS
--------------

An end-to-end CPU scheduling simulation and RL benchmarking platform
with classical scheduling baselines, PPO training, CLI inference,
visualization, reproducible experiments, and optional Groq-powered
analysis.