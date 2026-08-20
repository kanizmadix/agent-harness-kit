"use strict";

const DEMO_INTERVAL_MS = 1050;

const scenarios = {
  launch: {
    objective: "Plan a privacy-first product launch",
    input: "Create an evidence-backed launch brief for a small B2B team.",
    research: "3 market signals and 2 customer concerns collected",
    analysis: "Privacy posture selected as the primary differentiator",
    draft: "Launch narrative, channels, and success measures assembled",
    output: "Lead with verifiable privacy defaults, support the claim with a short trust brief, and launch through customer education plus targeted partner outreach. Measure qualified demos, trust-page engagement, and activation—not impressions alone."
  },
  brief: {
    objective: "Draft an AI adoption executive brief",
    input: "Summarize a low-risk 90-day adoption path for leadership.",
    research: "High-value workflows and adoption constraints mapped",
    analysis: "Two reversible pilots prioritized by impact and risk",
    draft: "Executive recommendation and 30/60/90 plan assembled",
    output: "Start with two measured, human-reviewed pilots: internal knowledge retrieval and first-draft support. Establish owners, quality baselines, access controls, and a stop condition before expanding at day 90."
  },
  incident: {
    objective: "Create a service incident update",
    input: "Prepare a clear stakeholder update from simulated incident facts.",
    research: "Timeline, customer impact, and mitigation facts collected",
    analysis: "Confirmed facts separated from open questions",
    draft: "Plain-language status update and next checkpoint assembled",
    output: "Service has recovered and monitoring remains elevated. The issue affected request latency, with no evidence of data loss. The team is validating the contributing change and will publish the next update after review."
  }
};

const elements = {
  shell: document.querySelector(".demo-shell"),
  objective: document.querySelector("#objective"),
  start: document.querySelector("#start-demo"),
  pause: document.querySelector("#pause-demo"),
  reset: document.querySelector("#reset-demo"),
  runState: document.querySelector("#run-state"),
  stepCount: document.querySelector("#step-count"),
  context: document.querySelector("#context-json"),
  announcer: document.querySelector("#demo-announcer"),
  log: document.querySelector("#event-log"),
  eventCount: document.querySelector("#event-count"),
  final: document.querySelector("#final-output"),
  finalText: document.querySelector("#final-text"),
  svg: document.querySelector("#flow-svg"),
  copy: document.querySelector("[data-copy]")
};

const agentNodes = Object.fromEntries(
  ["supervisor", "researcher", "analyst", "writer", "memory"].map((name) => [
    name,
    document.querySelector(`#node-${name}`)
  ])
);

let state;
let timer = null;

function freshState() {
  const scenario = scenarios[elements.objective.value];
  return {
    running: false,
    paused: false,
    step: 0,
    eventCount: 0,
    context: {
      session_id: "demo-local-001",
      messages: [],
      scratchpad: { objective: scenario.objective },
      artifacts: {},
      tools_state: {}
    }
  };
}

function setNodeStatus(name, status) {
  const node = agentNodes[name];
  node.dataset.status = status;
  node.querySelector("[data-agent-status]").textContent = status === "complete" ? "complete" : status;
}

function setAllNodes(status = "idle") {
  Object.keys(agentNodes).forEach((name) => setNodeStatus(name, name === "memory" && status === "idle" ? "waiting" : status));
}

function addEvent(source, message) {
  state.eventCount += 1;
  const item = document.createElement("li");
  const label = document.createElement("strong");
  label.textContent = source;
  item.append(label, document.createTextNode(` · ${message}`));
  elements.log.append(item);
  elements.log.scrollTop = elements.log.scrollHeight;
  elements.eventCount.textContent = `${state.eventCount} ${state.eventCount === 1 ? "event" : "events"}`;
}

function renderContext() {
  elements.context.textContent = JSON.stringify(state.context, null, 2);
  elements.stepCount.textContent = String(state.step);
}

function updateControls(mode) {
  elements.shell.dataset.demoState = mode;
  elements.runState.textContent = mode === "running" ? "Running" : mode === "paused" ? "Paused" : mode === "complete" ? "Complete" : "Ready";
  elements.start.textContent = mode === "paused" ? "Resume" : "Start";
  elements.start.disabled = mode === "running" || mode === "complete";
  elements.pause.disabled = mode !== "running";
  elements.objective.disabled = mode === "running" || mode === "paused";
}

function pauseSvg() {
  if (elements.svg && typeof elements.svg.pauseAnimations === "function") elements.svg.pauseAnimations();
}

function resumeSvg() {
  if (elements.svg && typeof elements.svg.unpauseAnimations === "function") elements.svg.unpauseAnimations();
}

function resetSvg() {
  if (!elements.svg) return;
  pauseSvg();
  if (typeof elements.svg.setCurrentTime === "function") elements.svg.setCurrentTime(0);
}

function runStep() {
  const scenario = scenarios[elements.objective.value];
  state.step += 1;

  switch (state.step) {
    case 1:
      setAllNodes("idle");
      setNodeStatus("supervisor", "active");
      state.context.messages.push({ role: "user", content: scenario.input });
      state.context.tools_state.supervisor = { status: "routing", next_agent: "researcher" };
      addEvent("Supervisor", "selected Researcher for evidence gathering");
      break;
    case 2:
      setNodeStatus("supervisor", "idle");
      setNodeStatus("researcher", "active");
      setNodeStatus("memory", "active");
      state.context.artifacts.research = scenario.research;
      state.context.tools_state.researcher = { status: "complete", artifact: "research" };
      addEvent("Researcher", scenario.research);
      break;
    case 3:
      setNodeStatus("researcher", "complete");
      setNodeStatus("memory", "complete");
      setNodeStatus("supervisor", "active");
      state.context.tools_state.supervisor = { status: "routing", next_agent: "analyst" };
      addEvent("Supervisor", "merged research and selected Analyst");
      break;
    case 4:
      setNodeStatus("supervisor", "idle");
      setNodeStatus("analyst", "active");
      setNodeStatus("memory", "active");
      state.context.artifacts.analysis = scenario.analysis;
      state.context.tools_state.analyst = { status: "complete", artifact: "analysis" };
      addEvent("Analyst", scenario.analysis);
      break;
    case 5:
      setNodeStatus("analyst", "complete");
      setNodeStatus("memory", "complete");
      setNodeStatus("supervisor", "active");
      state.context.tools_state.supervisor = { status: "routing", next_agent: "writer" };
      addEvent("Supervisor", "merged analysis and selected Writer");
      break;
    case 6:
      setNodeStatus("supervisor", "idle");
      setNodeStatus("writer", "active");
      setNodeStatus("memory", "active");
      state.context.artifacts.final_draft = scenario.draft;
      state.context.tools_state.writer = { status: "complete", artifact: "final_draft" };
      addEvent("Writer", scenario.draft);
      break;
    case 7:
      setNodeStatus("writer", "complete");
      setNodeStatus("memory", "complete");
      setNodeStatus("supervisor", "complete");
      state.context.messages.push({ role: "assistant", content: scenario.output });
      state.context.tools_state.supervisor = { status: "done", next_agent: null };
      addEvent("Supervisor", "returned final output and closed the run");
      elements.finalText.textContent = scenario.output;
      elements.final.hidden = false;
      state.running = false;
      window.clearInterval(timer);
      timer = null;
      pauseSvg();
      updateControls("complete");
      break;
    default:
      return;
  }

  renderContext();
}

function startDemo() {
  if (state.step >= 7) return;
  state.running = true;
  state.paused = false;
  updateControls("running");
  resumeSvg();
  runStep();
  if (state.step < 7) timer = window.setInterval(runStep, DEMO_INTERVAL_MS);
}

function pauseDemo() {
  if (!state.running) return;
  window.clearInterval(timer);
  timer = null;
  state.running = false;
  state.paused = true;
  pauseSvg();
  updateControls("paused");
  addEvent("Harness", `paused after step ${state.step}`);
}

function resetDemo() {
  window.clearInterval(timer);
  timer = null;
  state = freshState();
  setAllNodes();
  elements.log.replaceChildren();
  elements.eventCount.textContent = "0 events";
  elements.final.hidden = true;
  elements.finalText.textContent = "";
  elements.announcer.textContent = "Demo reset";
  resetSvg();
  updateControls("idle");
  renderContext();
}

async function copyInstallCommand() {
  const command = elements.copy.dataset.copy;
  try {
    await navigator.clipboard.writeText(command);
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(elements.copy.previousElementSibling);
    selection.removeAllRanges();
    selection.addRange(range);
  }
  elements.copy.textContent = "Copied";
  window.setTimeout(() => { elements.copy.textContent = "Copy"; }, 1600);
}

elements.start.addEventListener("click", startDemo);
elements.pause.addEventListener("click", pauseDemo);
elements.reset.addEventListener("click", resetDemo);
elements.objective.addEventListener("change", resetDemo);
elements.copy.addEventListener("click", copyInstallCommand);

resetDemo();
